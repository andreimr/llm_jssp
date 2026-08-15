# Knapsack and bin packing

## Dedicated knapsack solver

For pure knapsack (pick items maximizing value under one or more capacity
dimensions), `ortools.algorithms.python.knapsack_solver` beats a generic MILP.
Values and weights must be integers.

```python
from ortools.algorithms.python import knapsack_solver

values = [60, 100, 120]
weights = [[10, 20, 30]]  # one row per capacity dimension
capacities = [50]

solver = knapsack_solver.KnapsackSolver(
    knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
    "knapsack",
)
solver.init(values, weights, capacities)
best_value = solver.solve()
chosen = [i for i in range(len(values)) if solver.best_solution_contains(i)]
print("value:", best_value, "items:", chosen)
assert best_value == 220 and chosen == [1, 2]
```

Side constraints (conflicts between items, "at most one of these",
dependencies) are not supported here; use CP-SAT with booleans instead.

## Bin packing (minimize bins used) with CP-SAT

```python
from ortools.sat.python import cp_model

weights = [7, 5, 3, 4, 6, 2]
BIN_CAP = 10
n_items = len(weights)
n_bins = n_items  # worst case: one item per bin

model = cp_model.CpModel()
# x[i, b] = item i placed in bin b; used[b] = bin b is open.
x = {(i, b): model.new_bool_var(f"x_{i}_{b}") for i in range(n_items) for b in range(n_bins)}
used = [model.new_bool_var(f"used_{b}") for b in range(n_bins)]

for i in range(n_items):
    model.add_exactly_one(x[i, b] for b in range(n_bins))
for b in range(n_bins):
    model.add(
        sum(weights[i] * x[i, b] for i in range(n_items)) <= BIN_CAP * used[b]
    )
# Symmetry breaking: use lower-numbered bins first.
for b in range(1, n_bins):
    model.add(used[b] <= used[b - 1])

model.minimize(sum(used))
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0
status = solver.solve(model)
print(solver.status_name(status), "bins used:", int(solver.objective_value))
for b in range(n_bins):
    if solver.value(used[b]):
        items = [i for i in range(n_items) if solver.value(x[i, b])]
        print(f"  bin {b}: items {items} (load {sum(weights[i] for i in items)})")
assert status == cp_model.OPTIMAL and solver.objective_value == 3
```
