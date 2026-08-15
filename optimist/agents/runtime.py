"""Generic agent loop.

An Agent is a system prompt plus a set of tools with Python handlers. The
AgentRunner drives the provider until the agent stops calling tools, reporting
progress to a UI object. The same runtime powers the user-facing orchestrator
and every sub-agent, so behavior (streaming, error handling, usage accounting,
turn caps) is uniform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from optimist.messages import Message, ToolResultBlock, ToolSpec, Usage
from optimist.providers.base import Provider, ProviderError, StreamHandler

ToolHandler = Callable[[dict[str, Any]], tuple[str, bool]]  # -> (content, is_error)


class UI(Protocol):
    """Progress reporting hooks. The CLI renders these; tests can ignore them."""

    def agent_started(self, agent: str) -> None: ...
    def agent_finished(self, agent: str, turns: int) -> None: ...
    def text_delta(self, agent: str, delta: str) -> None: ...
    def thinking_delta(self, agent: str, delta: str) -> None: ...
    def tool_called(self, agent: str, tool: str, tool_input: dict[str, Any]) -> None: ...
    def tool_finished(self, agent: str, tool: str, result: str, is_error: bool) -> None: ...
    def notice(self, text: str) -> None: ...


class NullUI:
    def agent_started(self, agent: str) -> None:
        pass

    def agent_finished(self, agent: str, turns: int) -> None:
        pass

    def text_delta(self, agent: str, delta: str) -> None:
        pass

    def thinking_delta(self, agent: str, delta: str) -> None:
        pass

    def tool_called(self, agent: str, tool: str, tool_input: dict[str, Any]) -> None:
        pass

    def tool_finished(self, agent: str, tool: str, result: str, is_error: bool) -> None:
        pass

    def notice(self, text: str) -> None:
        pass


@dataclass
class Agent:
    name: str
    system: str
    tools: list[ToolSpec] = field(default_factory=list)
    handlers: dict[str, ToolHandler] = field(default_factory=dict)


MAX_TOOL_RESULT_CHARS = 60_000


@dataclass
class AgentRunner:
    provider: Provider
    model: str
    ui: UI
    usage: Usage
    max_tokens: int = 32000

    def run(self, agent: Agent, messages: list[Message], max_turns: int = 24) -> str:
        """Drive ``agent`` on ``messages`` (mutated in place). Returns final text."""
        self.ui.agent_started(agent.name)
        stream = StreamHandler(
            on_text=lambda d: self.ui.text_delta(agent.name, d),
            on_thinking=lambda d: self.ui.thinking_delta(agent.name, d),
        )
        turns = 0
        final_text = ""
        while turns < max_turns:
            turns += 1
            try:
                result = self.provider.complete(
                    model=self.model,
                    system=agent.system,
                    messages=messages,
                    tools=agent.tools or None,
                    max_tokens=self.max_tokens,
                    stream=stream,
                )
            except ProviderError as exc:
                self.ui.notice(f"[{agent.name}] provider error: {exc}")
                final_text = f"(provider error: {exc})"
                break
            self.usage.add(result.usage)
            messages.append(result.message)
            final_text = result.message.text()

            if result.stop_reason == "refusal":
                note = result.refusal_explanation or "the request was declined by safety filters"
                final_text = final_text or f"(request declined: {note})"
                self.ui.notice(f"[{agent.name}] declined: {note}")
                break
            if result.stop_reason == "max_tokens":
                self.ui.notice(f"[{agent.name}] output truncated at the token limit")
                break
            tool_uses = result.message.tool_uses()
            if result.stop_reason != "tool_use" or not tool_uses:
                break

            result_blocks: list[ToolResultBlock] = []
            for tu in tool_uses:
                self.ui.tool_called(agent.name, tu.name, tu.input)
                handler = agent.handlers.get(tu.name)
                if handler is None:
                    content, is_error = f"Unknown tool: {tu.name}", True
                else:
                    try:
                        content, is_error = handler(tu.input)
                    except Exception as exc:  # tool bugs must not kill the loop
                        content, is_error = f"Tool crashed: {exc!r}", True
                if len(content) > MAX_TOOL_RESULT_CHARS:
                    content = content[:MAX_TOOL_RESULT_CHARS] + "\n... [truncated]"
                self.ui.tool_finished(agent.name, tu.name, content, is_error)
                result_blocks.append(
                    ToolResultBlock(tool_use_id=tu.id, content=content, is_error=is_error)
                )
            messages.append(Message(role="user", content=list(result_blocks)))
        else:
            self.ui.notice(f"[{agent.name}] stopped at the {max_turns}-turn limit")
        self.ui.agent_finished(agent.name, turns)
        return final_text
