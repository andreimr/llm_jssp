"""Provider interface.

A provider turns a neutral conversation (see optimist.messages) into one
assistant turn, streaming progress through a StreamHandler. Implementations
must be synchronous and blocking; the CLI drives them from the main thread.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from optimist.messages import CompletionResult, Message, ToolSpec


@dataclass
class StreamHandler:
    """Callbacks invoked during streaming. All optional."""

    on_text: Callable[[str], None] | None = None
    on_thinking: Callable[[str], None] | None = None
    on_tool_use_start: Callable[[str], None] | None = None  # tool name
    extra: dict = field(default_factory=dict)

    def text(self, delta: str) -> None:
        if self.on_text:
            self.on_text(delta)

    def thinking(self, delta: str) -> None:
        if self.on_thinking:
            self.on_thinking(delta)

    def tool_use_start(self, name: str) -> None:
        if self.on_tool_use_start:
            self.on_tool_use_start(name)


class ProviderError(RuntimeError):
    """A provider-level failure the agent loop should surface, not crash on."""


class Provider(ABC):
    name: str = "base"

    @abstractmethod
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
        """Run one assistant turn (blocking), streaming via ``stream``."""

    @abstractmethod
    def list_models(self) -> list[dict]:
        """Return [{'id': ..., 'name': ...}, ...] of selectable models."""
