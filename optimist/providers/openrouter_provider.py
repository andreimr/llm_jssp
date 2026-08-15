"""OpenRouter provider (OpenAI-compatible Chat Completions API).

Gives access to hundreds of third-party models. Tool calling is translated to
the function-calling wire format; images are sent as data-URI image_url parts;
PDFs fall back to pre-extracted text (OpenRouter models lack native PDF input,
so optimist extracts text at attach time and inlines it).
"""

from __future__ import annotations

import json
import os
from typing import Any

from optimist.messages import (
    CompletionResult,
    ImageBlock,
    Message,
    PdfBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from optimist.providers.base import Provider, ProviderError, StreamHandler

BASE_URL = "https://openrouter.ai/api/v1"

# A short curated list shown when the live model listing is unavailable.
FALLBACK_MODELS = [
    {"id": "openai/gpt-5.2", "name": "OpenAI GPT-5.2"},
    {"id": "google/gemini-3-pro", "name": "Google Gemini 3 Pro"},
    {"id": "deepseek/deepseek-v4", "name": "DeepSeek V4"},
    {"id": "qwen/qwen3-max", "name": "Qwen3 Max"},
    {"id": "meta-llama/llama-4-maverick", "name": "Llama 4 Maverick"},
]


def _user_parts(blocks: list[Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            parts.append({"type": "text", "text": b.text})
        elif isinstance(b, ImageBlock):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{b.media_type};base64,{b.data_b64}"},
                }
            )
        elif isinstance(b, PdfBlock):
            label = b.name or "attached document"
            body = b.extracted_text or "(no extractable text)"
            parts.append(
                {
                    "type": "text",
                    "text": f"<attached_pdf name={label!r}>\n{body}\n</attached_pdf>",
                }
            )
    return parts


def messages_to_openai(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Convert the neutral conversation to Chat Completions messages.

    Tool results (stored inside neutral user turns) become role="tool" messages;
    assistant tool uses become tool_calls entries.
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in messages:
        if msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            text = msg.text()
            entry["content"] = text or None
            tool_calls = [
                {
                    "id": tu.id,
                    "type": "function",
                    "function": {"name": tu.name, "arguments": json.dumps(tu.input)},
                }
                for tu in msg.tool_uses()
            ]
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            continue

        tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
        for tr in tool_results:
            content = tr.content
            if tr.is_error:
                content = f"ERROR: {content}"
            out.append({"role": "tool", "tool_call_id": tr.tool_use_id, "content": content})
        other = [b for b in msg.content if not isinstance(b, ToolResultBlock)]
        if other:
            out.append({"role": "user", "content": _user_parts(other)})
    return out


def tools_to_openai(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise ProviderError("The 'openai' package is not installed.") from exc
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ProviderError("OPENROUTER_API_KEY is not set.")
            self._client = OpenAI(
                base_url=BASE_URL,
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/andreimr/llm_jssp",
                    "X-Title": "optimist",
                },
            )
        return self._client

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
        stream: StreamHandler | None = None,
    ) -> CompletionResult:
        import openai

        handler = stream or StreamHandler()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages_to_openai(system, messages),
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools_to_openai(tools)

        text_parts: list[str] = []
        # index -> {"id":…, "name":…, "arguments": str-accumulator}
        calls: dict[int, dict[str, Any]] = {}
        finish_reason = ""
        usage = Usage()
        try:
            response = self.client.chat.completions.create(**kwargs)
            for chunk in response:
                if getattr(chunk, "usage", None):
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta is None:
                    continue
                if delta.content:
                    text_parts.append(delta.content)
                    handler.text(delta.content)
                reasoning = getattr(delta, "reasoning", None)
                if reasoning:
                    handler.thinking(reasoning)
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] = tc.function.name
                            handler.tool_use_start(tc.function.name)
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
        except openai.AuthenticationError as exc:
            raise ProviderError(f"OpenRouter authentication failed: {exc}") from exc
        except openai.NotFoundError as exc:
            raise ProviderError(f"Unknown OpenRouter model {model!r}: {exc}") from exc
        except openai.APIStatusError as exc:
            raise ProviderError(f"OpenRouter API error ({exc.status_code}): {exc}") from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(f"Could not reach OpenRouter: {exc}") from exc

        blocks: list[Any] = []
        text = "".join(text_parts)
        if text:
            blocks.append(TextBlock(text))
        for i in sorted(calls):
            slot = calls[i]
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_malformed_arguments": slot["arguments"]}
            blocks.append(
                ToolUseBlock(id=slot["id"] or f"call_{i}", name=slot["name"], input=args)
            )

        stop_reason = {
            "tool_calls": "tool_use",
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "refusal",
        }.get(finish_reason, finish_reason or "end_turn")
        if calls and stop_reason == "end_turn":
            # Some models report finish_reason="stop" even with tool calls.
            stop_reason = "tool_use"

        message = Message(role="assistant", content=blocks, provider="openrouter")
        return CompletionResult(message=message, stop_reason=stop_reason, usage=usage)

    def list_models(self) -> list[dict]:
        try:
            import httpx

            resp = httpx.get(f"{BASE_URL}/models", timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            models = [
                {"id": m["id"], "name": m.get("name", m["id"])}
                for m in data
                if "tools" in (m.get("supported_parameters") or [])
            ]
            models.sort(key=lambda m: m["id"])
            return models or list(FALLBACK_MODELS)
        except Exception:
            return list(FALLBACK_MODELS)
