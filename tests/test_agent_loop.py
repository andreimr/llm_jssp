"""Agent-loop and orchestration tests using a scripted fake provider (no network)."""

from optimist.agents.orchestrator import Session, orchestrator_tools
from optimist.agents.runtime import Agent, AgentRunner, NullUI
from optimist.config import Settings
from optimist.messages import (
    CompletionResult,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from optimist.providers.base import Provider, StreamHandler


class FakeProvider(Provider):
    """Replays a fixed script of assistant turns."""

    name = "fake"

    def __init__(self, script: list[CompletionResult]):
        self.script = list(script)
        self.calls: list[list[Message]] = []

    def complete(self, *, model, system, messages, tools=None, max_tokens=16000, stream=None):
        self.calls.append(list(messages))
        handler = stream or StreamHandler()
        result = self.script.pop(0)
        for block in result.message.content:
            if isinstance(block, TextBlock):
                handler.text(block.text)
        return result

    def list_models(self):
        return []


def assistant_turn(*blocks, stop_reason="end_turn") -> CompletionResult:
    return CompletionResult(
        message=Message(role="assistant", content=list(blocks), provider="fake"),
        stop_reason=stop_reason,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


ECHO_TOOL = ToolSpec(
    name="echo",
    description="Echo input.",
    input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
)


def run_agent(provider, handlers, max_turns=8):
    agent = Agent(name="T", system="test", tools=[ECHO_TOOL], handlers=handlers)
    runner = AgentRunner(provider=provider, model="fake-1", ui=NullUI(), usage=Usage())
    messages = [Message(role="user", content=[TextBlock("go")])]
    return runner.run(agent, messages, max_turns=max_turns), messages


def test_tool_loop_executes_and_feeds_back_results():
    provider = FakeProvider([
        assistant_turn(
            TextBlock("calling"),
            ToolUseBlock(id="1", name="echo", input={"text": "ping"}),
            stop_reason="tool_use",
        ),
        assistant_turn(TextBlock("got: ping")),
    ])
    seen = []

    def echo(tool_input):
        seen.append(tool_input)
        return f"echo: {tool_input['text']}", False

    final, messages = run_agent(provider, {"echo": echo})
    assert final == "got: ping"
    assert seen == [{"text": "ping"}]
    # second provider call saw the tool result
    tool_result_msg = provider.calls[1][-1]
    block = tool_result_msg.content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.content == "echo: ping"
    assert not block.is_error


def test_unknown_tool_returns_error_result_not_crash():
    provider = FakeProvider([
        assistant_turn(
            ToolUseBlock(id="1", name="missing", input={}), stop_reason="tool_use"
        ),
        assistant_turn(TextBlock("recovered")),
    ])
    final, _ = run_agent(provider, {})
    assert final == "recovered"
    block = provider.calls[1][-1].content[0]
    assert block.is_error
    assert "Unknown tool" in block.content


def test_tool_exception_is_contained():
    provider = FakeProvider([
        assistant_turn(ToolUseBlock(id="1", name="echo", input={}), stop_reason="tool_use"),
        assistant_turn(TextBlock("ok")),
    ])

    def bomb(tool_input):
        raise RuntimeError("kaboom")

    final, _ = run_agent(provider, {"echo": bomb})
    assert final == "ok"
    block = provider.calls[1][-1].content[0]
    assert block.is_error
    assert "kaboom" in block.content


def test_turn_cap_stops_infinite_tool_loops():
    looping = [
        assistant_turn(
            ToolUseBlock(id=str(i), name="echo", input={"text": "again"}),
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    provider = FakeProvider(looping)
    final, _ = run_agent(provider, {"echo": lambda i: ("ok", False)}, max_turns=3)
    assert len(provider.calls) == 3


def test_refusal_stops_loop():
    provider = FakeProvider([
        assistant_turn(TextBlock(""), stop_reason="refusal"),
    ])
    final, _ = run_agent(provider, {})
    assert "declined" in final
    assert len(provider.calls) == 1


def test_usage_accumulates():
    provider = FakeProvider([
        assistant_turn(ToolUseBlock(id="1", name="echo", input={}), stop_reason="tool_use"),
        assistant_turn(TextBlock("done")),
    ])
    agent = Agent(name="T", system="s", tools=[ECHO_TOOL],
                  handlers={"echo": lambda i: ("ok", False)})
    usage = Usage()
    runner = AgentRunner(provider=provider, model="fake-1", ui=NullUI(), usage=usage)
    runner.run(agent, [Message(role="user", content=[TextBlock("go")])])
    assert usage.input_tokens == 20
    assert usage.output_tokens == 10


# --- Session-level handler tests (direct solver paths, no LLM involved) -----


def make_session() -> Session:
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp(prefix="optimist-test-ws-"))
    return Session(settings=Settings(workspace_root=root))


def test_session_solve_linear_handler():
    session = make_session()
    result, is_error = session._handle_solve_linear(
        {
            "variables": [{"name": "x", "lower_bound": 0, "upper_bound": 10}],
            "constraints": [],
            "objective": {"sense": "maximize", "coefficients": {"x": 2}},
        }
    )
    assert not is_error
    assert '"status": "OPTIMAL"' in result
    assert '"objective_value": 20.0' in result


def test_session_solve_linear_rejects_bad_spec():
    session = make_session()
    result, is_error = session._handle_solve_linear({"variables": []})
    assert is_error
    assert "Invalid linear model spec" in result


def test_session_run_python_handler():
    session = make_session()
    result, is_error = session._handle_run_python({"code": "print('42!')"})
    assert not is_error
    assert "42!" in result


def test_session_handlers_validate_required_fields():
    session = make_session()
    assert session._handle_formulate({})[1] is True
    assert session._handle_solve({})[1] is True
    assert session._handle_verify({})[1] is True
    assert session._handle_run_python({"code": "  "})[1] is True


def test_orchestrator_tool_schemas_are_wellformed():
    tools = orchestrator_tools()
    names = {t.name for t in tools}
    assert names == {
        "formulate_problem", "solve_problem", "verify_solution",
        "solve_linear_model", "run_python",
    }
    for t in tools:
        assert t.description
        assert t.input_schema.get("type") == "object" or "$defs" in t.input_schema
