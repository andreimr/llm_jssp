"""Rich-based rendering of agent activity.

The orchestrator's reply streams as live-rendered Markdown. Sub-agents
(Formulator / Solver / Verifier) stream dim text under an activity header, and
tool calls appear as compact one-liners, so the user can watch the pipeline
work without being buried in raw output.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

ORCHESTRATOR = "Optimist"

AGENT_COLORS = {
    "Optimist": "bold cyan",
    "Formulator": "bold magenta",
    "Solver": "bold green",
    "Verifier": "bold yellow",
}


def _preview(text: str, limit: int = 100) -> str:
    line = " ".join(text.strip().split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


class RichUI:
    """Implements the optimist.agents.runtime.UI protocol."""

    def __init__(self, console: Console | None = None, show_thinking: bool = False) -> None:
        self.console = console or Console()
        self.show_thinking = show_thinking
        self._live: Live | None = None
        self._buffer = ""
        self._sub_streaming = False  # a sub-agent is printing dim raw text
        self._thinking_shown = False

    # -- internal helpers ----------------------------------------------------

    def _start_live(self) -> None:
        if self._live is None:
            self._buffer = ""
            self._live = Live(
                Markdown(""),
                console=self.console,
                refresh_per_second=8,
                vertical_overflow="visible",
            )
            self._live.start()

    def _finalize_live(self) -> None:
        if self._live is not None:
            self._live.update(Markdown(self._buffer))
            self._live.stop()
            self._live = None
            self._buffer = ""

    def _end_sub_stream(self) -> None:
        if self._sub_streaming:
            self.console.print()
            self._sub_streaming = False

    # -- UI protocol -----------------------------------------------------------

    def agent_started(self, agent: str) -> None:
        self._thinking_shown = False
        if agent != ORCHESTRATOR:
            color = AGENT_COLORS.get(agent, "bold")
            self.console.print(Text(f"  ◆ {agent} working…", style=color))

    def agent_finished(self, agent: str, turns: int) -> None:
        if agent == ORCHESTRATOR:
            self._finalize_live()
        else:
            self._end_sub_stream()
            color = AGENT_COLORS.get(agent, "bold")
            self.console.print(Text(f"  ◆ {agent} done ({turns} turn{'s' if turns != 1 else ''})", style=color))

    def text_delta(self, agent: str, delta: str) -> None:
        if agent == ORCHESTRATOR:
            self._start_live()
            self._buffer += delta
            assert self._live is not None
            self._live.update(Markdown(self._buffer))
        else:
            self._sub_streaming = True
            self.console.print(Text(delta, style="dim"), end="")

    def thinking_delta(self, agent: str, delta: str) -> None:
        if self.show_thinking:
            self.console.print(Text(delta, style="grey50 italic"), end="")
            self._sub_streaming = True
        elif not self._thinking_shown:
            self._thinking_shown = True
            if agent == ORCHESTRATOR and self._live is None:
                self.console.print(Text("✻ thinking…", style="grey50 italic"))

    def tool_called(self, agent: str, tool: str, tool_input: dict[str, Any]) -> None:
        if agent == ORCHESTRATOR:
            self._finalize_live()
        self._end_sub_stream()
        if tool == "run_python":
            code = tool_input.get("code", "")
            detail = f"{len(code.splitlines())} lines of Python"
        elif tool == "solve_linear_model":
            n_vars = len(tool_input.get("variables", []) or [])
            n_cons = len(tool_input.get("constraints", []) or [])
            detail = f"{n_vars} variables, {n_cons} constraints"
        else:
            first = next(iter(tool_input.values()), "")
            detail = _preview(str(first), 80)
        self.console.print(Text(f"  ● {tool}  ", style="bold blue") + Text(detail, style="dim"))

    def tool_finished(self, agent: str, tool: str, result: str, is_error: bool) -> None:
        self._end_sub_stream()
        style = "red" if is_error else "dim"
        marker = "✗" if is_error else "✓"
        self.console.print(Text(f"    {marker} {_preview(result)}", style=style))

    def notice(self, text: str) -> None:
        if self._live is not None:
            self._finalize_live()
        self._end_sub_stream()
        self.console.print(Text(f"  ⚠ {text}", style="yellow"))
