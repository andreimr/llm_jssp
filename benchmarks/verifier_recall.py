"""Verifier-recall test: can the Verifier catch sabotaged solutions?

For each tested instance we construct, with exact offline solvers:
  - control:      the true optimal solution, correctly reported  -> expect VERIFIED
  - wrong_obj:    a correct solution with a misreported objective -> expect ISSUES FOUND
  - infeasible:   a constraint-violating solution whose objective
                  is computed "correctly" from its own values     -> expect ISSUES FOUND
  - suboptimal:   a genuinely feasible but worse solution,
                  claimed to be optimal                            -> expect ISSUES FOUND

Each case is sent straight to the Verifier agent (Session._handle_verify) with
the instance file present in the session workspace, mirroring how the
orchestrator uses it. Scores: recall on sabotaged cases, false alarms on
controls. Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from registry import get  # noqa: E402
from validate import parse_jsp, parse_mknap  # noqa: E402

from optimist.agents.orchestrator import Session  # noqa: E402
from optimist.cli.render import RichUI  # noqa: E402
from optimist.config import Settings  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

JSP_STATEMENT = """\
Job-shop scheduling instance in the file {filename} (standard OR-Library /
JSPLIB format: '#' lines are comments; first data line "n m"; then one line
per job with m "machine duration" pairs in processing order, machines
0-indexed). Each machine handles one operation at a time; a job's operations
run in the given order; no preemption. Objective: minimize the makespan.
"""

MKNAP_STATEMENT = """\
Multidimensional 0/1 knapsack instance in the file {filename} (OR-Library
mknap1 format: first line "n m best_known"; then n profits; then the m x n
constraint matrix row by row; then m capacities). Choose items maximizing
total profit subject to every capacity constraint. Note: the third number on
the first line is a reported best-known value present in the file format;
judge the claim below on its own merits against the data.
"""


# -- exact solvers returning full solutions -----------------------------------


def solve_jsp_full(jobs, extra_makespan_eq=None):
    from ortools.sat.python import cp_model

    horizon = sum(d for job in jobs for _, d in job)
    model = cp_model.CpModel()
    starts, ends, per_machine = {}, {}, {}
    for j, job in enumerate(jobs):
        for k, (m, d) in enumerate(job):
            s = model.new_int_var(0, horizon, f"s{j}_{k}")
            e = model.new_int_var(0, horizon, f"e{j}_{k}")
            per_machine.setdefault(m, []).append(model.new_interval_var(s, d, e, f"i{j}_{k}"))
            starts[j, k], ends[j, k] = s, e
        for k in range(1, len(job)):
            model.add(starts[j, k] >= ends[j, k - 1])
    for ivs in per_machine.values():
        model.add_no_overlap(ivs)
    makespan = model.new_int_var(0, horizon, "mk")
    model.add_max_equality(makespan, [ends[j, len(job) - 1] for j, job in enumerate(jobs)])
    if extra_makespan_eq is not None:
        model.add(makespan == extra_makespan_eq)
    model.minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    status = solver.solve(model)
    assert solver.status_name(status) in ("OPTIMAL", "FEASIBLE"), solver.status_name(status)
    schedule = {
        (j, k): (m, solver.value(starts[j, k]), solver.value(ends[j, k]))
        for j, job in enumerate(jobs)
        for k, (m, _d) in enumerate(job)
    }
    return int(solver.value(makespan)), schedule


def jsp_schedule_text(schedule) -> str:
    lines = []
    for (j, k), (m, s, e) in sorted(schedule.items()):
        lines.append(f"job {j} op {k}: machine {m}, start {s}, end {e}")
    return "\n".join(lines)


def solve_mknap_full(profits, rows, caps, forbid=None):
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("SCIP")
    x = [solver.BoolVar(f"x{i}") for i in range(len(profits))]
    for row, cap in zip(rows, caps, strict=True):
        solver.Add(solver.Sum(c * xi for c, xi in zip(row, x, strict=True)) <= cap)
    if forbid is not None:
        solver.Add(x[forbid] == 0)
    solver.Maximize(solver.Sum(p * xi for p, xi in zip(profits, x, strict=True)))
    status = solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL
    chosen = [i for i in range(len(profits)) if x[i].solution_value() > 0.5]
    return solver.Objective().Value(), chosen


# -- sabotage construction -----------------------------------------------------


@dataclass
class Case:
    instance: str
    case: str
    expected: str  # "VERIFIED" | "ISSUES FOUND"
    claimed_solution: str


def jsp_cases(name: str) -> list[Case]:
    inst = get(name)
    jobs = parse_jsp(inst.path.read_text())
    opt_mk, sched = solve_jsp_full(jobs)
    assert opt_mk == inst.known_optimum

    def claim(mk, schedule, note="Status: OPTIMAL (proven)."):
        return f"Claimed makespan: {mk}. {note}\nSchedule:\n{jsp_schedule_text(schedule)}"

    cases = [Case(name, "control", "VERIFIED", claim(opt_mk, sched))]
    # wrong objective: true schedule, misreported (impossibly good) makespan
    cases.append(Case(name, "wrong_obj", "ISSUES FOUND", claim(opt_mk - 3, sched)))
    # infeasible: overlap two operations on the busiest machine, keep the claim
    bad = dict(sched)
    by_machine: dict[int, list] = {}
    for key, (m, s, e) in sched.items():
        by_machine.setdefault(m, []).append((s, e, key))
    ops = max(by_machine.values(), key=len)
    ops.sort()
    (s1, _e1, _k1), (_s2, e2, k2) = ops[0], ops[1]
    m2, s2, e2 = bad[k2]
    bad[k2] = (m2, s1, s1 + (e2 - s2))  # slam op2 onto op1's start
    cases.append(Case(name, "infeasible", "ISSUES FOUND", claim(opt_mk, bad)))
    # suboptimal claimed optimal: feasible schedule forced to a worse makespan
    worse_mk, worse = solve_jsp_full(jobs, extra_makespan_eq=opt_mk + 7)
    cases.append(Case(name, "suboptimal", "ISSUES FOUND", claim(worse_mk, worse)))
    return cases


def mknap_cases(name: str) -> list[Case]:
    inst = get(name)
    profits, rows, caps, _reported = parse_mknap(inst.path.read_text())
    opt_val, chosen = solve_mknap_full(profits, rows, caps)
    assert abs(opt_val - inst.known_optimum) < 1e-6

    def claim(value, items, note="Status: OPTIMAL (proven)."):
        return (
            f"Claimed total profit: {value}. {note}\n"
            f"Chosen items (0-indexed): {sorted(items)}"
        )

    cases = [Case(name, "control", "VERIFIED", claim(round(opt_val, 4), chosen))]
    cases.append(
        Case(name, "wrong_obj", "ISSUES FOUND", claim(round(opt_val * 1.06, 4), chosen))
    )
    # infeasible: add an unchosen item that violates >= 1 constraint; profit
    # honestly recomputed so only feasibility is wrong
    violator = None
    for i in range(len(profits)):
        if i in chosen:
            continue
        usage_ok = all(
            sum(row[j] for j in [*chosen, i]) <= cap for row, cap in zip(rows, caps, strict=True)
        )
        if not usage_ok:
            violator = i
            break
    assert violator is not None
    bad_items = [*chosen, violator]
    bad_val = round(sum(profits[j] for j in bad_items), 4)
    cases.append(Case(name, "infeasible", "ISSUES FOUND", claim(bad_val, bad_items)))
    # suboptimal claimed optimal: exact solve with the most profitable chosen item banned
    best_item = max(chosen, key=lambda j: profits[j])
    sub_val, sub_items = solve_mknap_full(profits, rows, caps, forbid=best_item)
    assert sub_val < opt_val - 1e-6
    cases.append(Case(name, "suboptimal", "ISSUES FOUND", claim(round(sub_val, 4), sub_items)))
    return cases


# -- runner --------------------------------------------------------------------


@dataclass
class CaseResult:
    instance: str
    case: str
    expected: str
    verdict: str
    correct: bool
    wall_s: float
    est_cost_usd: float
    detail_tail: str


def run_case(case: Case, console: Console) -> CaseResult:
    inst = get(case.instance)
    settings = Settings.load()
    settings.workspace_root = Path(tempfile.mkdtemp(prefix="verifier-recall-"))
    session = Session(settings=settings, ui=RichUI(console))
    session.add_file(inst.path)
    session.all_attachments.extend(session.pending_attachments)
    session.pending_attachments.clear()

    statement_tpl = JSP_STATEMENT if inst.family == "jsp" else MKNAP_STATEMENT
    console.rule(f"{case.instance} / {case.case} (expect {case.expected})")
    start = time.time()
    text, _ = session._handle_verify(
        {
            "problem_statement": statement_tpl.format(filename=inst.filename),
            "claimed_solution": case.claimed_solution,
        }
    )
    wall = time.time() - start
    match = re.search(r"VERDICT:\s*([A-Z ]+?)(?:\n|$)", text)
    verdict = match.group(1).strip() if match else "MISSING"
    correct = verdict == case.expected
    usage = session.usage
    cost = (usage.input_tokens * 5 + usage.output_tokens * 25
            + usage.cache_read_tokens * 0.5 + usage.cache_write_tokens * 6.25) / 1e6
    console.print(f">>> {case.instance}/{case.case}: verdict {verdict} "
                  f"(expected {case.expected}) {'OK' if correct else 'MISS'} "
                  f"{wall:.0f}s ${cost:.2f}")
    return CaseResult(
        instance=case.instance, case=case.case, expected=case.expected,
        verdict=verdict, correct=correct, wall_s=round(wall, 1),
        est_cost_usd=round(cost, 3), detail_tail=text[-300:],
    )


def main() -> int:
    console = Console(force_terminal=False, width=110)
    cases = jsp_cases("ft06") + mknap_cases("mknap01_2")
    results = [run_case(c, console) for c in cases]

    sabotaged = [r for r in results if r.expected != "VERIFIED"]
    controls = [r for r in results if r.expected == "VERIFIED"]
    recall = sum(r.correct for r in sabotaged)
    false_alarms = sum(not r.correct for r in controls)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"verifier-recall-{stamp}.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )
    lines = [
        f"# Verifier recall test — {stamp}",
        "",
        f"Recall on sabotaged solutions: **{recall}/{len(sabotaged)}**  ·  "
        f"False alarms on controls: **{false_alarms}/{len(controls)}**",
        "",
        "| Instance | Case | Expected | Verdict | Correct | Wall s | Cost $ |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.instance} | {r.case} | {r.expected} | {r.verdict} "
            f"| {'yes' if r.correct else 'NO'} | {r.wall_s} | {r.est_cost_usd} |"
        )
    (RESULTS_DIR / f"verifier-recall-{stamp}.md").write_text("\n".join(lines) + "\n")
    console.print(f"RECALL {recall}/{len(sabotaged)}  FALSE ALARMS {false_alarms}/{len(controls)}")
    return 0 if recall == len(sabotaged) and false_alarms == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
