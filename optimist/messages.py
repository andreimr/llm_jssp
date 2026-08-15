"""Provider-neutral conversation model.

Every provider (Anthropic, OpenRouter) converts to and from this small set of
dataclasses, so agents and the CLI never touch provider wire formats. Assistant
messages keep the provider's raw content payload (``raw``) so that features like
Anthropic thinking blocks can be replayed verbatim on the next turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    text: str


@dataclass
class ImageBlock:
    media_type: str  # image/png, image/jpeg, image/gif, image/webp
    data_b64: str
    name: str = ""


@dataclass
class PdfBlock:
    data_b64: str
    name: str = ""
    extracted_text: str = ""  # fallback for providers without native PDF input


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ImageBlock | PdfBlock | ToolUseBlock | ToolResultBlock


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: list[ContentBlock] = field(default_factory=list)
    # Raw provider-specific content for assistant turns (e.g. Anthropic content
    # blocks including thinking). Keyed by provider name so a mid-conversation
    # provider switch degrades gracefully to the neutral blocks.
    raw: dict[str, Any] | None = None
    provider: str = ""

    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


def user_message(*blocks: ContentBlock) -> Message:
    return Message(role="user", content=list(blocks))


def user_text(text: str) -> Message:
    return user_message(TextBlock(text))


@dataclass
class ToolSpec:
    """A tool definition, provider-agnostic (JSON Schema input)."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


@dataclass
class CompletionResult:
    message: Message
    stop_reason: str  # end_turn | tool_use | max_tokens | refusal | ...
    usage: Usage = field(default_factory=Usage)
    refusal_explanation: str = ""
