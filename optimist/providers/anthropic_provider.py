"""Anthropic provider (official SDK, streaming, adaptive thinking, native PDF/vision).

Design notes:
- Assistant turns are replayed from ``Message.raw`` when they were produced by
  this provider, so thinking blocks round-trip unchanged (required by the API).
- The system prompt gets a cache_control breakpoint: agents re-send a stable
  system prompt plus growing history every turn, which is the textbook caching
  shape.
- Claude Opus 5 / Fable 5 requests opt into server-side refusal fallbacks
  (beta) so a safety-classifier decline is transparently retried on a fallback
  model instead of surfacing as a dead turn.
"""

from __future__ import annotations

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

# Models offered in /models. Any claude-* id is accepted; these are curated.
KNOWN_MODELS = [
    {"id": "claude-opus-5", "name": "Claude Opus 5 (default; best agentic/coding)"},
    {"id": "claude-fable-5", "name": "Claude Fable 5 (most capable; premium pricing)"},
    {"id": "claude-sonnet-5", "name": "Claude Sonnet 5 (fast, near-Opus quality)"},
    {"id": "claude-opus-4-8", "name": "Claude Opus 4.8"},
    {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5 (fastest, cheapest)"},
]

# Models supporting {"type": "adaptive"} thinking. Others get no thinking param.
_ADAPTIVE_THINKING_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)

# Models whose safety classifiers can refuse; opt into server-side fallbacks.
_FALLBACK_MODELS = ("claude-fable-5", "claude-mythos-5", "claude-opus-5")


def supports_adaptive_thinking(model: str) -> bool:
    return model.startswith(_ADAPTIVE_THINKING_PREFIXES)


def block_to_anthropic(block: Any) -> dict[str, Any]:
    """Convert one neutral content block to Anthropic wire format."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.media_type,
                "data": block.data_b64,
            },
        }
    if isinstance(block, PdfBlock):
        doc: dict[str, Any] = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": block.data_b64,
            },
        }
        if block.name:
            doc["title"] = block.name
        return doc
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    raise TypeError(f"Unsupported block type: {type(block)!r}")


def messages_to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert the neutral conversation to Anthropic messages.

    Assistant turns produced by this provider replay their raw content
    (preserving thinking blocks); anything else is rebuilt from neutral blocks.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "assistant" and msg.provider == "anthropic" and msg.raw:
            out.append({"role": "assistant", "content": msg.raw["content"]})
        else:
            out.append(
                {"role": msg.role, "content": [block_to_anthropic(b) for b in msg.content]}
            )
    return out


def tools_to_anthropic(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ProviderError("The 'anthropic' package is not installed.") from exc
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                raise ProviderError(
                    "No Anthropic credentials found. Set ANTHROPIC_API_KEY "
                    "(or ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile)."
                )
            self._client = anthropic.Anthropic()
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
        import anthropic

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages_to_anthropic(messages),
        }
        if tools:
            kwargs["tools"] = tools_to_anthropic(tools)
        if supports_adaptive_thinking(model):
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if model.startswith(_FALLBACK_MODELS):
            # Server-side refusal fallbacks: a classifier decline is retried on
            # Anthropic's recommended fallback model within the same call.
            kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kwargs["extra_body"] = {"fallbacks": "default"}

        handler = stream or StreamHandler()
        try:
            with self.client.messages.stream(**kwargs) as s:
                for event in s:
                    etype = getattr(event, "type", "")
                    if etype == "content_block_start":
                        cb = event.content_block
                        if cb.type == "tool_use":
                            handler.tool_use_start(cb.name)
                    elif etype == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            handler.text(delta.text)
                        elif delta.type == "thinking_delta" and delta.thinking:
                            handler.thinking(delta.thinking)
                final = s.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise ProviderError(f"Anthropic authentication failed: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise ProviderError(f"Unknown Anthropic model {model!r}: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"Could not reach the Anthropic API: {exc}") from exc

        blocks: list[Any] = []
        for cb in final.content:
            if cb.type == "text":
                blocks.append(TextBlock(cb.text))
            elif cb.type == "tool_use":
                blocks.append(ToolUseBlock(id=cb.id, name=cb.name, input=dict(cb.input)))
            # thinking / fallback blocks live only in raw

        usage = Usage(
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            cache_read_tokens=getattr(final.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(final.usage, "cache_creation_input_tokens", 0) or 0,
        )
        message = Message(
            role="assistant",
            content=blocks,
            raw={"content": [cb.model_dump() for cb in final.content]},
            provider="anthropic",
        )
        refusal_explanation = ""
        if final.stop_reason == "refusal" and getattr(final, "stop_details", None):
            refusal_explanation = getattr(final.stop_details, "explanation", "") or ""
        return CompletionResult(
            message=message,
            stop_reason=final.stop_reason or "end_turn",
            usage=usage,
            refusal_explanation=refusal_explanation,
        )

    def list_models(self) -> list[dict]:
        return list(KNOWN_MODELS)
