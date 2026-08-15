"""Offline validation of the registry: re-prove every known optimum exactly.

Run this before trusting battery scores. Job-shop instances are solved to
proven optimality with CP-SAT; multidimensional knapsack instances with SCIP.
No LLM, no network. Exits non-zero on any mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from registry import INSTANCES, Instance  # noqa: E402


def parse_jsp(text: str) -> list[list[tuple[int, int]]]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    n_jobs, n_machines = map(int, lines[0].split())
    jobs = []
    for line in lines[1 : 1 + n_jobs]:
        nums = list(map(int, line.split()))
        assert len(nums) == 2 * n_machines, "malformed job line"
        jobs.append([(nums[2 * k], nums[2 * k + 1]) for k in range(n_machines)])
    return jobs


def solve_jsp(jobs: list[list[tuple[int, int]]], time_limit_s: float = 120.0) -> tuple[str, int]:
    from ortools.sat.python import cp_model

    horizon = sum(d for job in jobs for _, d in job)
    model = cp_model.CpModel()
    starts, ends, per_machine = {}, {}, {}
    for j, job in enumerate(jobs):
        for k, (machine, dur) in enumerate(job):
            s = model.new_int_var(0, horizon, f"s{j}_{k}")
            e = model.new_int_var(0, horizon, f"e{j}_{k}")
            per_machine.setdefault(machine, []).append(
                model.new_interval_var(s, dur, e, f"iv{j}_{k}")
            )
            starts[j, k], ends[j, k] = s, e
        for k in range(1, len(job)):
            model.add(starts[j, k] >= ends[j, k - 1])
    for ivs in per_machine.values():
        model.add_no_overlap(ivs)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [ends[j, len(job) - 1] for j, job in enumerate(jobs)])
    model.minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 8
    status = solver.solve(model)
    return solver.status_name(status), int(solver.value(makespan))


def parse_mknap(text: str) -> tuple[list[float], list[list[float]], list[float], float]:
    nums = text.split()
    n, m, reported_opt = int(nums[0]), int(nums[1]), float(nums[2])
    idx = 3
    profits = [float(x) for x in nums[idx : idx + n]]
    idx += n
    rows = []
    for _ in range(m):
        rows.append([float(x) for x in nums[idx : idx + n]])
        idx += n
    capacities = [float(x) for x in nums[idx : idx + m]]
    idx += m
    assert idx == len(nums), f"trailing tokens in mknap file ({len(nums) - idx})"
    return profits, rows, capacities, reported_opt


def solve_mknap(
    profits: list[float], rows: list[list[float]], capacities: list[float]
) -> tuple[str, float]:
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("SCIP")
    assert solver is not None
    x = [solver.BoolVar(f"x{i}") for i in range(len(profits))]
    for row, cap in zip(rows, capacities, strict=True):
        solver.Add(solver.Sum(c * xi for c, xi in zip(row, x, strict=True)) <= cap)
    solver.Maximize(solver.Sum(p * xi for p, xi in zip(profits, x, strict=True)))
    solver.SetTimeLimit(120_000)
    status = solver.Solve()
    name = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else str(status)
    return name, solver.Objective().Value()


def validate(instance: Instance) -> tuple[bool, str]:
    text = instance.path.read_text()
    if instance.family == "jsp":
        status, value = solve_jsp(parse_jsp(text))
    elif instance.family == "mknap":
        profits, rows, caps, reported = parse_mknap(text)
        if abs(reported - instance.known_optimum) > 1e-6:
            return False, f"file header optimum {reported} != registry {instance.known_optimum}"
        status, value = solve_mknap(profits, rows, caps)
    else:
        return False, f"unknown family {instance.family}"
    ok = status == "OPTIMAL" and abs(value - instance.known_optimum) < 1e-6
    return ok, f"{status} {value} (registry: {instance.known_optimum})"


def main() -> int:
    failures = 0
    for instance in INSTANCES:
        ok, detail = validate(instance)
        print(f"{'OK  ' if ok else 'FAIL'} {instance.name:12s} {detail}")
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
