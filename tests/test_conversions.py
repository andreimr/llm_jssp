"""Wire-format conversion tests (no network, no API keys)."""

import json

from optimist.config import provider_for, strip_model_prefix
from optimist.messages import (
    ImageBlock,
    Message,
    PdfBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from optimist.providers.anthropic_provider import (
    messages_to_anthropic,
    supports_adaptive_thinking,
    tools_to_anthropic,
)
from optimist.providers.openrouter_provider import messages_to_openai, tools_to_openai

TOOL = ToolSpec(
    name="run_python",
    description="Run code.",
    input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
)


def sample_conversation() -> list[Message]:
    return [
        Message(
            role="user",
            content=[
                TextBlock("solve this"),
                ImageBlock(media_type="image/png", data_b64="aWpn", name="chart.png"),
                PdfBlock(data_b64="cGRm", name="data.pdf", extracted_text="table: 1 2 3"),
            ],
        ),
        Message(
            role="assistant",
            content=[
                TextBlock("running code"),
                ToolUseBlock(id="t1", name="run_python", input={"code": "print(1)"}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="1", is_error=False)],
        ),
        Message(role="assistant", content=[TextBlock("done: 1")]),
    ]


def test_anthropic_message_conversion():
    wire = messages_to_anthropic(sample_conversation())
    assert [m["role"] for m in wire] == ["user", "assistant", "user", "assistant"]
    first = wire[0]["content"]
    assert first[0] == {"type": "text", "text": "solve this"}
    assert first[1]["type"] == "image"
    assert first[1]["source"]["media_type"] == "image/png"
    assert first[2]["type"] == "document"
    assert first[2]["source"]["media_type"] == "application/pdf"
    assert first[2]["title"] == "data.pdf"
    tool_use = wire[1]["content"][1]
    assert tool_use == {
        "type": "tool_use", "id": "t1", "name": "run_python", "input": {"code": "print(1)"},
    }
    tool_result = wire[2]["content"][0]
    assert tool_result["tool_use_id"] == "t1"
    assert tool_result["is_error"] is False


def test_anthropic_raw_replay_preserved():
    raw_content = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "hi"},
    ]
    msg = Message(
        role="assistant",
        content=[TextBlock("hi")],
        raw={"content": raw_content},
        provider="anthropic",
    )
    wire = messages_to_anthropic([msg])
    assert wire[0]["content"] is raw_content  # replayed verbatim


def test_anthropic_raw_not_replayed_across_providers():
    msg = Message(
        role="assistant",
        content=[TextBlock("hi")],
        raw={"content": [{"type": "text", "text": "hi"}]},
        provider="openrouter",
    )
    wire = messages_to_anthropic([msg])
    assert wire[0]["content"] == [{"type": "text", "text": "hi"}]


def test_anthropic_tool_conversion():
    assert tools_to_anthropic([TOOL])[0]["input_schema"]["type"] == "object"


def test_openai_message_conversion():
    wire = messages_to_openai("be helpful", sample_conversation())
    assert wire[0] == {"role": "system", "content": "be helpful"}
    user = wire[1]
    assert user["role"] == "user"
    kinds = [p["type"] for p in user["content"]]
    assert kinds == ["text", "image_url", "text"]  # pdf becomes extracted text
    assert user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "table: 1 2 3" in user["content"][2]["text"]

    assistant = wire[2]
    assert assistant["role"] == "assistant"
    call = assistant["tool_calls"][0]
    assert call["id"] == "t1"
    assert json.loads(call["function"]["arguments"]) == {"code": "print(1)"}

    tool_msg = wire[3]
    assert tool_msg == {"role": "tool", "tool_call_id": "t1", "content": "1"}
    assert wire[4]["role"] == "assistant"


def test_openai_error_tool_result_flagged():
    messages = [
        Message(role="user", content=[ToolResultBlock(tool_use_id="x", content="bad", is_error=True)])
    ]
    wire = messages_to_openai("s", messages)
    assert wire[1]["content"].startswith("ERROR:")


def test_openai_tool_conversion():
    wire = tools_to_openai([TOOL])[0]
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "run_python"


def test_provider_inference():
    assert provider_for("claude-opus-5") == "anthropic"
    assert provider_for("deepseek/deepseek-v4") == "openrouter"
    assert provider_for("openrouter:foo") == "openrouter"
    assert provider_for("or:foo/bar") == "openrouter"
    assert strip_model_prefix("openrouter:foo/bar") == "foo/bar"
    assert strip_model_prefix("claude-opus-5") == "claude-opus-5"


def test_adaptive_thinking_gate():
    assert supports_adaptive_thinking("claude-opus-5")
    assert supports_adaptive_thinking("claude-fable-5")
    assert not supports_adaptive_thinking("claude-haiku-4-5")
