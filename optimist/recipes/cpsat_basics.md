# CP-SAT fundamentals

Use CP-SAT (`ortools.sat.python.cp_model`) for anything combinatorial:
integer/boolean models, logical constraints, scheduling, sequencing, and
feasibility problems. It is the strongest general-purpose solver in OR-Tools.

Current-API notes (OR-Tools 9.x):
- Prefer the snake_case API: `model.new_int_var`, `model.add`, `solver.solve`,
  `solver.value(x)`. The old CamelCase (`NewIntVar`) still works but is legacy.
- CP-SAT is integer-only. Scale rational data (e.g. dollars-and-cents × 100)
  and report the scale factor in your output.
- `x < 5` is allowed directly; negate a bool var with `~b`.
- Always set `solver.parameters.max_time_in_seconds`; add
  `solver.parameters.num_workers = 8` for parallel search on hard models.
- Check status against `cp_model.OPTIMAL` / `cp_model.FEASIBLE`; a FEASIBLE
  result is not proven optimal, say so.

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
x = model.new_int_var(0, 100, "x")
y = model.new_int_var(0, 100, "y")
b = model.new_bool_var("b")

model.add(x + 2 * y <= 14)
model.add(3 * x - y >= 0)
model.add(x - y <= 2)
# Reified (conditional) constraint: b is true iff x >= 5.
model.add(x >= 5).only_enforce_if(b)
model.add(x < 5).only_enforce_if(~b)

model.maximize(3 * x + 4 * y)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0
solver.parameters.num_workers = 8
status = solver.solve(model)

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(solver.status_name(status), solver.objective_value)
    print("x =", solver.value(x), "y =", solver.value(y), "b =", solver.value(b))
assert status == cp_model.OPTIMAL and solver.objective_value == 34
```

## Diagnosing infeasibility with assumptions

Guard each suspect constraint group with a bool literal and add the literals
as assumptions; on INFEASIBLE, the solver names a sufficient conflicting subset.

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
z = model.new_int_var(0, 10, "z")
a_low = model.new_bool_var("needs_z_at_least_8")
a_high = model.new_bool_var("needs_z_at_most_3")
model.add(z >= 8).only_enforce_if(a_low)
model.add(z <= 3).only_enforce_if(a_high)
model.add_assumptions([a_low, a_high])

solver = cp_model.CpSolver()
status = solver.solve(model)
if status == cp_model.INFEASIBLE:
    conflict = [
        str(model.get_bool_var_from_proto_index(i))
        for i in solver.sufficient_assumptions_for_infeasibility()
    ]
    print("conflicting constraint groups:", conflict)
assert status == cp_model.INFEASIBLE
```

## Enumerating all solutions

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
v = model.new_int_var(1, 3, "v")
w = model.new_int_var(1, 3, "w")
model.add(v != w)


class Collector(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        super().__init__()
        self.solutions = []

    def on_solution_callback(self):
        self.solutions.append((self.value(v), self.value(w)))


collector = Collector()
solver = cp_model.CpSolver()
solver.parameters.enumerate_all_solutions = True
solver.solve(model, collector)
print(len(collector.solutions), "solutions")
assert len(collector.solutions) == 6
```
