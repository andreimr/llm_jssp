"""The orchestrator agent and the session that hosts a conversation with it.

The orchestrator is the only agent the user talks to. Its tools either do
direct work (structured LP solve, Python sandbox) or spawn a specialist
sub-agent with a fresh conversation:

    user ──► Orchestrator ──► formulate_problem ──► Formulator (no tools)
                        ├──► solve_problem     ──► Solver    (run_python loop)
                        ├──► verify_solution   ──► Verifier  (run_python loop)
                        ├──► solve_linear_model ─► OR-Tools directly
                        └──► run_python         ─► sandbox directly

Attachments the user added (PDFs/images with problem data) are forwarded
automatically into every sub-agent's first message, so data never has to
survive a lossy retelling by the orchestrator.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from optimist import recipes
from optimist.agents import prompts
from optimist.agents.runtime import UI, Agent, AgentRunner, NullUI
from optimist.attachments import (
    AttachmentError,
    data_file_preview,
    is_data_file,
    load_attachment,
)
from optimist.config import Settings, provider_for, strip_model_prefix
from optimist.messages import (
    ContentBlock,
    Message,
    PdfBlock,
    TextBlock,
    ToolSpec,
    Usage,
)
from optimist.providers import get_provider
from optimist.tools import sandbox
from optimist.tools.lp import LinearProblem, solve_linear_problem

RUN_PYTHON_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python source to execute."},
        "timeout_s": {
            "type": "integer",
            "description": "Wall-clock timeout in seconds (default 120, max 600).",
        },
    },
    "required": ["code"],
}

RUN_PYTHON_TOOL = ToolSpec(
    name="run_python",
    description=(
        "Execute Python code in a subprocess and return stdout/stderr. Google "
        "OR-Tools (ortools) is installed. No network access should be assumed. "
        "The code runs in the session's persistent working directory: attached "
        "data files are there (read them by relative path), and files you write "
        "(intermediate data, solution.json) survive for later runs. Print "
        "everything you need to see."
    ),
    input_schema=RUN_PYTHON_SCHEMA,
)

READ_RECIPE_TOOL = ToolSpec(
    name="read_recipe",
    description=(
        "Read a verified OR-Tools cookbook recipe: current API idioms plus a "
        "complete tested example for a problem class. Consult the relevant "
        "recipe BEFORE writing solver code; training-data recollections of "
        "OR-Tools APIs are often outdated. Available recipes:\n"
        + recipes.recipe_catalog()
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": sorted(recipes.RECIPES),
                "description": "Recipe name.",
            }
        },
        "required": ["name"],
    },
)


def orchestrator_tools() -> list[ToolSpec]:
    lp_schema = LinearProblem.model_json_schema()
    return [
        ToolSpec(
            name="formulate_problem",
            description=(
                "Ask the Formulator agent to turn a natural-language problem into a "
                "formal optimization model (problem class, variables, objective, "
                "constraints, recommended solver). Include ALL data and context in "
                "problem_statement; the user's file attachments are forwarded "
                "automatically."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "problem_statement": {
                        "type": "string",
                        "description": "Complete, faithful problem statement with all data.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional guidance, e.g. clarifications the user gave.",
                    },
                },
                "required": ["problem_statement"],
            },
        ),
        ToolSpec(
            name="solve_problem",
            description=(
                "Ask the Solver agent to implement a formulation with Google OR-Tools, "
                "run it, debug it, and report the solution. Pass the full formulation "
                "text (typically the Formulator's output). Use previous_issues to relay "
                "Verifier findings when re-solving."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "formulation": {
                        "type": "string",
                        "description": "The complete formal model, including all data.",
                    },
                    "previous_issues": {
                        "type": "string",
                        "description": "Optional: problems found with an earlier attempt.",
                    },
                },
                "required": ["formulation"],
            },
        ),
        ToolSpec(
            name="verify_solution",
            description=(
                "Ask the Verifier agent to independently check a claimed solution "
                "against the original problem statement using its own code. Returns a "
                "verdict (VERIFIED / ISSUES FOUND / INCONCLUSIVE) with details."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "problem_statement": {"type": "string"},
                    "formulation": {"type": "string"},
                    "claimed_solution": {
                        "type": "string",
                        "description": "The solution to audit, with objective and variable values.",
                    },
                },
                "required": ["problem_statement", "claimed_solution"],
            },
        ),
        ToolSpec(
            name="solve_linear_model",
            description=(
                "Directly solve a small linear or mixed-integer linear program with "
                "OR-Tools from a structured spec — no code generation. Use for plainly "
                "linear problems with a modest number of variables/constraints."
            ),
            input_schema=lp_schema,
        ),
        RUN_PYTHON_TOOL,
    ]


@dataclass
class Session:
    """One conversation between the user and the orchestrator."""

    settings: Settings
    ui: UI = field(default_factory=NullUI)
    usage: Usage = field(default_factory=Usage)
    history: list[Message] = field(default_factory=list)
    pending_attachments: list[ContentBlock] = field(default_factory=list)
    all_attachments: list[ContentBlock] = field(default_factory=list)
    _workspace: Path | None = None

    # -- workspace ------------------------------------------------------------

    @property
    def workspace(self) -> Path:
        """The session's persistent working directory (created on first use).

        All run_python calls execute here; attachments are saved here; solver
        artifacts written here survive across runs within the session.
        """
        if self._workspace is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            root = self.settings.workspace_root
            root.mkdir(parents=True, exist_ok=True)
            self._workspace = Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=root))
        return self._workspace

    def workspace_files(self) -> list[Path]:
        if self._workspace is None or not self._workspace.is_dir():
            return []
        return sorted(
            p for p in self._workspace.iterdir()
            if p.is_file() and not p.name.startswith("_optimist_run_")
        )

    def _workspace_manifest(self) -> str:
        files = self.workspace_files()
        if not files:
            return ""
        rows = "\n".join(f"  {p.name}  ({p.stat().st_size:,} bytes)" for p in files)
        return (
            "Files currently in the working directory (run_python executes there; "
            f"read them by relative path):\n{rows}"
        )

    def add_file(self, path: str | Path) -> str:
        """Attach a file: documents/images become content blocks, data files
        become workspace files with a preview block. Returns a description
        for the UI. Raises AttachmentError on unsupported/missing files."""
        p = Path(path).expanduser()
        if is_data_file(p):
            if not p.is_file():
                raise AttachmentError(f"No such file: {p}")
            dest = self.workspace / p.name
            shutil.copyfile(p, dest)
            preview = data_file_preview(dest)
            self.pending_attachments.append(
                TextBlock(
                    f"<attached_data_file name={p.name!r} workspace_path='./{p.name}'>\n"
                    f"{preview}\n</attached_data_file>\n"
                    "(The full file is in the working directory; solver code should "
                    "read it from there rather than relying on this preview.)"
                )
            )
            return f"{p.name} → workspace (solver code reads it as ./{p.name})"
        block = load_attachment(p)
        # Keep a raw copy (and extracted text for PDFs) where solver code can reach it.
        dest = self.workspace / Path(p).name
        shutil.copyfile(p, dest)
        if isinstance(block, PdfBlock) and block.extracted_text:
            (self.workspace / f"{dest.stem}.extracted.txt").write_text(
                block.extracted_text, encoding="utf-8"
            )
        self.pending_attachments.append(block)
        return f"{p.name} (sent with your next message; raw copy in workspace)"

    # -- model plumbing -----------------------------------------------------

    def _runner(self, model: str) -> AgentRunner:
        model_id = strip_model_prefix(model)
        provider = get_provider(provider_for(model))
        return AgentRunner(
            provider=provider,
            model=model_id,
            ui=self.ui,
            usage=self.usage,
            max_tokens=self.settings.max_tokens,
        )

    def _run_subagent(
        self, name: str, system: str, first_message: str, *,
        tools: list[ToolSpec] | None = None, max_turns: int = 12,
        include_attachments: bool = True,
    ) -> str:
        manifest = self._workspace_manifest()
        if manifest:
            first_message = f"{first_message}\n\n{manifest}"
        blocks: list[ContentBlock] = [TextBlock(first_message)]
        if include_attachments and self.all_attachments:
            blocks.extend(self.all_attachments)
        available = {
            "run_python": self._handle_run_python,
            "read_recipe": self._handle_read_recipe,
        }
        handlers = {t.name: available[t.name] for t in (tools or []) if t.name in available}
        agent = Agent(name=name, system=system, tools=tools or [], handlers=handlers)
        runner = self._runner(self.settings.effective_subagent_model())
        messages = [Message(role="user", content=blocks)]
        text = runner.run(agent, messages, max_turns=max_turns)
        return text or "(the sub-agent returned no text)"

    # -- tool handlers ------------------------------------------------------

    def _handle_formulate(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        statement = tool_input.get("problem_statement", "").strip()
        if not statement:
            return "formulate_problem requires a problem_statement.", True
        notes = tool_input.get("notes", "")
        message = statement if not notes else f"{statement}\n\nAdditional notes:\n{notes}"
        return self._run_subagent(
            "Formulator", prompts.FORMULATOR, message,
            tools=[READ_RECIPE_TOOL], max_turns=4,
        ), False

    def _handle_solve(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        formulation = tool_input.get("formulation", "").strip()
        if not formulation:
            return "solve_problem requires a formulation.", True
        message = f"Implement and solve this formulation:\n\n{formulation}"
        issues = tool_input.get("previous_issues", "")
        if issues:
            message += (
                "\n\nA previous attempt had these issues (found by an independent "
                f"verifier) — address them:\n{issues}"
            )
        return self._run_subagent(
            "Solver", prompts.SOLVER, message,
            tools=[RUN_PYTHON_TOOL, READ_RECIPE_TOOL], max_turns=14,
        ), False

    def _handle_verify(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        statement = tool_input.get("problem_statement", "").strip()
        claimed = tool_input.get("claimed_solution", "").strip()
        if not statement or not claimed:
            return "verify_solution requires problem_statement and claimed_solution.", True
        formulation = tool_input.get("formulation", "")
        message = (
            f"Original problem statement:\n{statement}\n\n"
            + (f"Formulation used:\n{formulation}\n\n" if formulation else "")
            + f"Claimed solution to audit:\n{claimed}"
        )
        return self._run_subagent(
            "Verifier", prompts.VERIFIER, message, tools=[RUN_PYTHON_TOOL], max_turns=10
        ), False

    def _handle_solve_linear(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        try:
            problem = LinearProblem.model_validate(tool_input)
            solution = solve_linear_problem(problem)
        except ValidationError as exc:
            return f"Invalid linear model spec:\n{exc}", True
        except ValueError as exc:
            return f"Invalid linear model: {exc}", True
        except Exception as exc:
            return f"Solver failure: {exc!r}", True
        return json.dumps(solution.model_dump(), indent=2), solution.status in (
            "ERROR",
        )

    def _handle_read_recipe(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        name = str(tool_input.get("name", ""))
        try:
            return recipes.read_recipe(name), False
        except KeyError:
            return (
                f"Unknown recipe {name!r}. Available: {', '.join(sorted(recipes.RECIPES))}",
                True,
            )

    def _handle_run_python(self, tool_input: dict[str, Any]) -> tuple[str, bool]:
        code = tool_input.get("code", "")
        if not code.strip():
            return "run_python requires non-empty code.", True
        timeout = int(tool_input.get("timeout_s") or self.settings.sandbox_timeout_s)
        result = sandbox.run_python(code, timeout_s=timeout, workdir=self.workspace)
        return result.as_tool_result()

    # -- public API ----------------------------------------------------------

    def build_orchestrator(self) -> Agent:
        return Agent(
            name="Optimist",
            system=prompts.ORCHESTRATOR,
            tools=orchestrator_tools(),
            handlers={
                "formulate_problem": self._handle_formulate,
                "solve_problem": self._handle_solve,
                "verify_solution": self._handle_verify,
                "solve_linear_model": self._handle_solve_linear,
                "run_python": self._handle_run_python,
            },
        )

    def ask(self, user_input: str) -> str:
        """Send one user message through the orchestrator; returns final text."""
        manifest = self._workspace_manifest()
        if manifest:
            user_input = f"{user_input}\n\n<workspace_note>\n{manifest}\n</workspace_note>"
        blocks: list[ContentBlock] = [TextBlock(user_input)]
        if self.pending_attachments:
            blocks.extend(self.pending_attachments)
            self.all_attachments.extend(self.pending_attachments)
            self.pending_attachments = []
        self.history.append(Message(role="user", content=blocks))
        runner = self._runner(self.settings.model)
        return runner.run(self.build_orchestrator(), self.history, max_turns=24)

    def reset(self) -> None:
        self.history.clear()
        self.pending_attachments.clear()
        self.all_attachments.clear()
        self._workspace = None  # a fresh conversation gets a fresh workspace
