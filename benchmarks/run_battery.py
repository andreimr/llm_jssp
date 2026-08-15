"""Live evaluation battery: run optimist end-to-end on benchmark instances.

For each instance: a fresh Session, the instance file attached as a workspace
data file, one prompt, then scoring of the claimed objective against the known
(offline re-proven) optimum. Requires ANTHROPIC_API_KEY (or OpenRouter
credentials with --model).

Usage:
    python benchmarks/run_battery.py                 # whole battery
    python benchmarks/run_battery.py --only ft06     # one instance
    python benchmarks/run_battery.py --model claude-sonnet-5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from registry import INSTANCES, Instance  # noqa: E402

from optimist.agents.orchestrator import Session  # noqa: E402
from optimist.cli.render import RichUI  # noqa: E402
from optimist.config import Settings  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

# $/MTok for cost estimates, keyed by model prefix.
PRICES = {
    "claude-opus-5": (5.0, 25.0, 0.5, 6.25),
    "claude-sonnet-5": (3.0, 15.0, 0.3, 3.75),
    "claude-haiku-4-5": (1.0, 5.0, 0.1, 1.25),
    "claude-fable-5": (10.0, 50.0, 1.0, 12.5),
}

OBJECTIVE_RE = re.compile(r"OBJECTIVE:\s*\$?(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)


class CaptureUI(RichUI):
    """RichUI that also records verifier verdicts and tool errors for scoring."""

    def __init__(self, console: Console) -> None:
        super().__init__(console)
        self.verdicts: list[str] = []
        self.tool_errors: int = 0
        self.tool_calls: int = 0

    def tool_called(self, agent, tool, tool_input):
        self.tool_calls += 1
        super().tool_called(agent, tool, tool_input)

    def tool_finished(self, agent, tool, result, is_error):
        if is_error:
            self.tool_errors += 1
        if tool == "verify_solution":
            match = re.search(r"VERDICT:\s*([A-Z ]+)", result)
            self.verdicts.append(match.group(1).strip() if match else "MISSING")
        super().tool_finished(agent, tool, result, is_error)


@dataclass
class RunResult:
    instance: str
    family: str
    known_optimum: float
    claimed_objective: float | None
    objective_source: str  # "final_text" | "solution_json" | "none"
    outcome: str  # "optimal" | "suboptimal" | "superoptimal_anomaly" | "no_answer"
    gap_percent: float | None
    verdicts: list[str]
    tool_calls: int
    tool_errors: int
    wall_s: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    est_cost_usd: float
    final_text_tail: str = ""
    error: str = ""
    workspace: str = ""


def estimate_cost(model: str, usage) -> float:
    for prefix, (p_in, p_out, p_read, p_write) in PRICES.items():
        if model.startswith(prefix):
            return (
                usage.input_tokens * p_in
                + usage.output_tokens * p_out
                + usage.cache_read_tokens * p_read
                + usage.cache_write_tokens * p_write
            ) / 1e6
    return 0.0


def extract_objective(final_text: str, workspace: Path | None) -> tuple[float | None, str]:
    matches = OBJECTIVE_RE.findall(final_text)
    if matches:
        return float(matches[-1].replace(",", "")), "final_text"
    if workspace is not None:
        sol = workspace / "solution.json"
        if sol.is_file():
            try:
                data = json.loads(sol.read_text())
                value = data.get("objective")
                if isinstance(value, (int, float)):
                    return float(value), "solution_json"
            except (json.JSONDecodeError, OSError):
                pass
    return None, "none"


def classify(instance: Instance, claimed: float | None) -> tuple[str, float | None]:
    if claimed is None:
        return "no_answer", None
    diff = claimed - instance.known_optimum
    if abs(diff) < 1e-6:
        return "optimal", 0.0
    gap = 100.0 * abs(diff) / max(abs(instance.known_optimum), 1e-9)
    worse = diff > 0 if instance.sense == "min" else diff < 0
    return ("suboptimal" if worse else "superoptimal_anomaly"), round(gap, 3)


def run_instance(instance: Instance, model: str, console: Console) -> RunResult:
    settings = Settings.load()
    settings.model = model
    ui = CaptureUI(console)
    session = Session(settings=settings, ui=ui)
    session.add_file(instance.path)
    console.rule(f"[bold]{instance.name}[/bold] (known optimum {instance.known_optimum})")
    start = time.time()
    error = ""
    try:
        final = session.ask(instance.prompt())
    except Exception as exc:  # a crash scores as a failure, not a battery abort
        final, error = "", f"{type(exc).__name__}: {exc}"
    wall = time.time() - start
    claimed, source = extract_objective(final, session._workspace)
    outcome, gap = classify(instance, claimed)
    if error:
        outcome = "no_answer"
    usage = session.usage
    return RunResult(
        instance=instance.name,
        family=instance.family,
        known_optimum=instance.known_optimum,
        claimed_objective=claimed,
        objective_source=source,
        outcome=outcome,
        gap_percent=gap,
        verdicts=ui.verdicts,
        tool_calls=ui.tool_calls,
        tool_errors=ui.tool_errors,
        wall_s=round(wall, 1),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        est_cost_usd=round(estimate_cost(model, usage), 3),
        final_text_tail=final[-400:],
        error=error,
        workspace=str(session._workspace or ""),
    )


def write_reports(results: list[RunResult], model: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}.json").write_text(
        json.dumps({"model": model, "results": [asdict(r) for r in results]}, indent=2)
    )
    solved = sum(r.outcome == "optimal" for r in results)
    total_cost = sum(r.est_cost_usd for r in results)
    lines = [
        f"# Battery report — {stamp}",
        "",
        f"Model: `{model}`  ·  Solved to known optimum: **{solved}/{len(results)}**  ·  "
        f"Estimated cost: **${total_cost:.2f}**",
        "",
        "| Instance | Family | Known | Claimed | Outcome | Gap % | Verifier | Tools (err) | Wall s | Cost $ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.instance} | {r.family} | {r.known_optimum} | {r.claimed_objective} "
            f"| {r.outcome} | {r.gap_percent if r.gap_percent is not None else '—'} "
            f"| {', '.join(r.verdicts) or '—'} | {r.tool_calls} ({r.tool_errors}) "
            f"| {r.wall_s} | {r.est_cost_usd} |"
        )
    path = RESULTS_DIR / f"{stamp}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    console = Console(force_terminal=False, width=110)
    todo = [i for i in INSTANCES if not args.only or i.name in args.only]
    results = []
    for instance in todo:
        result = run_instance(instance, args.model, console)
        results.append(result)
        console.print(
            f"\n>>> {result.instance}: {result.outcome} "
            f"(claimed {result.claimed_objective}, known {result.known_optimum}, "
            f"{result.wall_s}s, ${result.est_cost_usd})\n"
        )
    report = write_reports(results, args.model)
    console.print(f"report: {report}")
    solved = sum(r.outcome == "optimal" for r in results)
    console.print(f"SOLVED {solved}/{len(results)}")
    return 0 if solved == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
