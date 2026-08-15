# Scheduling with CP-SAT

Use CP-SAT interval variables for job shop, flow shop, open shop, single- and
parallel-machine scheduling, timetabling, and resource-constrained project
scheduling (RCPSP). Do not attempt big-M MILP formulations of disjunctive
scheduling when CP-SAT's `add_no_overlap` does it natively and faster.

Key devices:
- `model.new_interval_var(start, size, end, name)` ties start + size == end.
- One machine at a time: `model.add_no_overlap(intervals_on_that_machine)`.
- Precedence within a job: `start[j,k] >= end[j,k-1]`.
- Makespan: `add_max_equality(makespan, last_ends)`, then minimize.
- Renewable resources: `model.add_cumulative(intervals, demands, capacity)`.
- Optional activities: `model.new_optional_interval_var(..., presence_bool)`.

## Job shop (minimize makespan)

```python
from ortools.sat.python import cp_model

# Each job is an ordered list of (machine, duration).
jobs = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3), (0, 4)],
]
horizon = sum(d for job in jobs for _, d in job)

model = cp_model.CpModel()
starts, ends = {}, {}
machine_intervals = {}
for j, job in enumerate(jobs):
    for k, (machine, duration) in enumerate(job):
        start = model.new_int_var(0, horizon, f"start_{j}_{k}")
        end = model.new_int_var(0, horizon, f"end_{j}_{k}")
        interval = model.new_interval_var(start, duration, end, f"iv_{j}_{k}")
        starts[j, k], ends[j, k] = start, end
        machine_intervals.setdefault(machine, []).append(interval)
    for k in range(1, len(job)):
        model.add(starts[j, k] >= ends[j, k - 1])

for intervals in machine_intervals.values():
    model.add_no_overlap(intervals)

makespan = model.new_int_var(0, horizon, "makespan")
model.add_max_equality(makespan, [ends[j, len(job) - 1] for j, job in enumerate(jobs)])
model.minimize(makespan)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0
solver.parameters.num_workers = 8
status = solver.solve(model)
print(solver.status_name(status), "makespan =", solver.value(makespan))
for j, job in enumerate(jobs):
    for k, (machine, duration) in enumerate(job):
        s = solver.value(starts[j, k])
        print(f"  job {j} op {k}: machine {machine}, {s} .. {s + duration}")
assert solver.status_name(status) == "OPTIMAL" and solver.value(makespan) == 11
```

## Cumulative resource (RCPSP-style)

```python
from ortools.sat.python import cp_model

durations = [3, 2, 4]
demands = [2, 2, 1]
CAPACITY = 3

model = cp_model.CpModel()
intervals = []
last_ends = []
for i, (dur, _dem) in enumerate(zip(durations, demands)):
    start = model.new_int_var(0, 20, f"s{i}")
    interval = model.new_interval_var(start, dur, start + dur, f"iv{i}")
    intervals.append(interval)
    last_ends.append(start + dur)
model.add_cumulative(intervals, demands, CAPACITY)

makespan = model.new_int_var(0, 20, "makespan")
model.add_max_equality(makespan, last_ends)
model.minimize(makespan)

solver = cp_model.CpSolver()
status = solver.solve(model)
print(solver.status_name(status), "makespan =", solver.value(makespan))
assert status == cp_model.OPTIMAL
```
