# LP and MILP with pywraplp

Use `ortools.linear_solver.pywraplp` for continuous linear programs and mixed-
integer linear programs. Backend choice:
- `CreateSolver("GLOP")`: pure LP (fast, exact simplex, exposes duals).
- `CreateSolver("SCIP")` (or `"CBC"`): MILP. Integer variables require these.
- Very large LPs (millions of nonzeros): consider `CreateSolver("PDLP")`,
  a first-order method; expect approximate optima.

Notes:
- `CreateSolver` returns `None` when a backend is unavailable; check it.
- Use `solver.infinity()` for unbounded sides.
- `solver.SetTimeLimit(milliseconds)`; after solving check
  `pywraplp.Solver.OPTIMAL` vs `FEASIBLE` vs `INFEASIBLE` vs `UNBOUNDED`.
- MILPs give no dual values; sensitivity questions need the LP relaxation.

## MILP

```python
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver("SCIP")
assert solver is not None
inf = solver.infinity()

x = solver.IntVar(0, inf, "x")
y = solver.IntVar(0, inf, "y")
solver.Add(x + 7 * y <= 17.5)
solver.Add(x <= 3.5)
solver.Maximize(x + 10 * y)
solver.SetTimeLimit(30_000)

status = solver.Solve()
assert status == pywraplp.Solver.OPTIMAL
print("objective:", solver.Objective().Value())
print("x =", x.solution_value(), " y =", y.solution_value())
assert solver.Objective().Value() == 23
```

## LP with sensitivity analysis (shadow prices, reduced costs)

```python
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver("GLOP")
assert solver is not None
inf = solver.infinity()

x = solver.NumVar(0, inf, "x")
y = solver.NumVar(0, inf, "y")
c_x_cap = solver.Add(x <= 4)
c_y_cap = solver.Add(2 * y <= 12)
c_joint = solver.Add(3 * x + 2 * y <= 18)
solver.Maximize(3 * x + 5 * y)

status = solver.Solve()
assert status == pywraplp.Solver.OPTIMAL
print("objective:", solver.Objective().Value())

# Shadow price: objective gain per unit of extra right-hand side.
for name, con in [("x_cap", c_x_cap), ("y_cap", c_y_cap), ("joint", c_joint)]:
    print(f"  dual[{name}] = {con.dual_value():.3f}")
# Reduced cost: objective change per unit of forcing a nonbasic variable up.
for var in (x, y):
    print(f"  reduced_cost[{var.name()}] = {var.reduced_cost():.3f}")
print("  constraint activities:", solver.ComputeConstraintActivities())
assert abs(c_y_cap.dual_value() - 1.5) < 1e-6
assert abs(c_joint.dual_value() - 1.0) < 1e-6
```

Interpretation for users: a dual value of 1.5 on a capacity constraint means
one more unit of that capacity is worth 1.5 objective units (locally, while
the basis stays optimal); a zero dual means the constraint is slack.
