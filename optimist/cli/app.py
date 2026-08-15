"""The optimist REPL: a Claude-Code-style terminal chat for optimization work."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from optimist import __version__
from optimist.agents.orchestrator import Session
from optimist.attachments import AttachmentError, load_attachment
from optimist.cli.render import RichUI
from optimist.config import Settings, provider_for
from optimist.messages import ImageBlock, PdfBlock, TextBlock
from optimist.providers import get_provider
from optimist.providers.base import ProviderError

COMMANDS = [
    "/help", "/model", "/models", "/subagent-model", "/attach", "/attachments",
    "/clear", "/save", "/usage", "/thinking", "/exit", "/quit",
]

BANNER = """\
[bold cyan]optimist[/bold cyan] v{version} — natural language in, solved models out.

Describe an optimization, scheduling, or constraint problem and the agent team
(Formulator → Solver → Verifier, on Google OR-Tools) takes it from there.
[dim]Model: {model}   ·   /help for commands   ·   /attach file.pdf to add data[/dim]"""

HELP = """\
[bold]/model \\[id][/bold]           show or set the model (claude-*, or vendor/model via OpenRouter)
[bold]/models \\[filter][/bold]      list available models (Anthropic + OpenRouter)
[bold]/subagent-model \\[id][/bold]  set a cheaper/faster model for the sub-agents ("" = same)
[bold]/attach <path>[/bold]        attach a PDF or image to your next message
[bold]/attachments[/bold]          list attachments in this session
[bold]/clear[/bold]                start a fresh conversation
[bold]/save \\[path][/bold]          save the transcript as markdown
[bold]/usage[/bold]                show token usage for this session
[bold]/thinking on|off[/bold]      stream model reasoning summaries
[bold]/exit[/bold]                 leave (also Ctrl-D)

Anything else you type is sent to the agent team. Examples:
  [dim]I run a bakery: flour, butter and sugar stocks are ..., croissants need ...
  which mix maximizes profit?[/dim]
  [dim]Schedule these 6 jobs on 3 machines to minimize makespan: ...[/dim]
  [dim]/attach tenders.pdf  then: assign contractors to lots at minimum cost[/dim]"""


def _check_credentials(console: Console, model: str) -> None:
    import os

    provider = provider_for(model)
    if provider == "anthropic" and not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        console.print(
            "[yellow]⚠ ANTHROPIC_API_KEY is not set — set it (or `ant auth login`) "
            "before sending a message.[/yellow]"
        )
    if provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        console.print("[yellow]⚠ OPENROUTER_API_KEY is not set.[/yellow]")


def save_transcript(session: Session, path: Path) -> None:
    lines = [f"# optimist transcript — {datetime.now(UTC).isoformat(timespec='seconds')}\n"]
    for msg in session.history:
        if msg.role == "user" and not msg.text().strip():
            continue  # tool-result plumbing
        lines.append(f"## {'You' if msg.role == 'user' else 'Optimist'}\n")
        for block in msg.content:
            if isinstance(block, TextBlock):
                lines.append(block.text + "\n")
            elif isinstance(block, (PdfBlock, ImageBlock)):
                lines.append(f"*(attachment: {block.name})*\n")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


class App:
    def __init__(self, settings: Settings, console: Console | None = None) -> None:
        self.settings = settings
        self.console = console or Console()
        self.ui = RichUI(self.console, show_thinking=settings.show_thinking)
        self.session = Session(settings=settings, ui=self.ui)

    # -- slash commands -------------------------------------------------------

    def handle_command(self, line: str) -> bool:
        """Returns False when the app should exit."""
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        c = self.console
        if cmd in ("/exit", "/quit"):
            return False
        elif cmd == "/help":
            c.print(Panel(HELP, title="optimist", border_style="cyan"))
        elif cmd == "/model":
            if arg:
                self.settings.model = arg
                _check_credentials(c, arg)
            c.print(f"model: [bold]{self.settings.model}[/bold] "
                    f"(provider: {provider_for(self.settings.model)})")
        elif cmd == "/subagent-model":
            self.settings.subagent_model = arg
            c.print(f"sub-agent model: [bold]{self.settings.effective_subagent_model()}[/bold]")
        elif cmd == "/models":
            self._list_models(arg)
        elif cmd == "/attach":
            if not arg:
                c.print("[red]usage: /attach <path>[/red]")
            else:
                for raw in arg.split():
                    try:
                        block = load_attachment(raw)
                        self.session.pending_attachments.append(block)
                        c.print(f"[green]✓ attached {block.name} "
                                f"(sent with your next message)[/green]")
                    except AttachmentError as exc:
                        c.print(f"[red]✗ {exc}[/red]")
        elif cmd == "/attachments":
            pending = [b.name for b in self.session.pending_attachments]
            sent = [b.name for b in self.session.all_attachments]
            c.print(f"pending: {pending or '—'}\nsent: {sent or '—'}")
        elif cmd == "/clear":
            self.session.reset()
            c.print("[green]conversation cleared[/green]")
        elif cmd == "/save":
            path = Path(arg) if arg else Path(
                f"optimist-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
            )
            save_transcript(self.session, path)
            c.print(f"[green]saved {path}[/green]")
        elif cmd == "/usage":
            u = self.session.usage
            c.print(
                f"input: {u.input_tokens:,}  output: {u.output_tokens:,}  "
                f"cache read: {u.cache_read_tokens:,}  cache write: {u.cache_write_tokens:,}"
            )
        elif cmd == "/thinking":
            self.ui.show_thinking = arg == "on"
            c.print(f"thinking display: {'on' if self.ui.show_thinking else 'off'}")
        else:
            c.print(f"[red]unknown command {cmd} — /help lists commands[/red]")
        return True

    def _list_models(self, query: str) -> None:
        table = Table(title="models", show_lines=False)
        table.add_column("id", style="bold")
        table.add_column("name", style="dim")
        for m in get_provider("anthropic").list_models():
            if query.lower() in m["id"].lower():
                table.add_row(m["id"], m["name"])
        try:
            openrouter = get_provider("openrouter").list_models()
        except ProviderError:
            openrouter = []
        shown = 0
        for m in openrouter:
            if query.lower() in m["id"].lower():
                table.add_row(m["id"], m["name"])
                shown += 1
                if shown >= 40 and not query:
                    table.add_row("…", "(pass a filter to see more, e.g. /models qwen)")
                    break
        self.console.print(table)

    # -- main loop -------------------------------------------------------------

    def ask(self, text: str) -> None:
        try:
            self.session.ask(text)
        except KeyboardInterrupt:
            self.ui._finalize_live()
            self.console.print("\n[yellow]— interrupted; partial context kept —[/yellow]")
        except ProviderError as exc:
            self.ui._finalize_live()
            self.console.print(f"[red]{exc}[/red]")

    def run(self) -> None:
        c = self.console
        c.print(Panel(BANNER.format(version=__version__, model=self.settings.model),
                      border_style="cyan"))
        _check_credentials(c, self.settings.model)
        self.settings.history_file.parent.mkdir(parents=True, exist_ok=True)
        prompt: PromptSession = PromptSession(
            history=FileHistory(str(self.settings.history_file)),
            completer=WordCompleter(COMMANDS, sentence=True),
            complete_while_typing=True,
        )
        while True:
            try:
                line = prompt.prompt([("bold fg:ansicyan", "optimist> ")]).strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            if not line:
                continue
            if line.startswith("/"):
                if not self.handle_command(line):
                    break
                continue
            c.print()
            self.ask(line)
            c.print()
        c.print(Text("goodbye ✦", style="cyan"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="optimist",
        description="Multi-agent optimization assistant: describe a problem, get a "
                    "solved, verified model with a plain-language explanation.",
    )
    parser.add_argument("-m", "--model", help="model id (claude-* or an OpenRouter id)")
    parser.add_argument("--subagent-model", help="model id for the sub-agents")
    parser.add_argument("--attach", action="append", default=[], metavar="FILE",
                        help="attach a PDF/image (repeatable)")
    parser.add_argument("prompt", nargs="*", help="one-shot prompt (omit for interactive mode)")
    args = parser.parse_args(argv)

    settings = Settings.load()
    if args.model:
        settings.model = args.model
    if args.subagent_model is not None:
        settings.subagent_model = args.subagent_model

    app = App(settings)
    for raw in args.attach:
        try:
            app.session.pending_attachments.append(load_attachment(raw))
        except AttachmentError as exc:
            app.console.print(f"[red]{exc}[/red]")
            return 2

    if args.prompt:
        _check_credentials(app.console, settings.model)
        app.ask(" ".join(args.prompt))
        return 0
    try:
        app.run()
    except EOFError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
