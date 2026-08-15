# Network flows and assignment

For problems with pure network structure, the dedicated solvers in
`ortools.graph.python` are orders of magnitude faster than a MILP formulation
and exact. Reach for them when the problem is literally a flow: transportation
and transshipment, bipartite assignment, max-throughput, project selection via
min-cut.

Notes:
- All capacities, costs, and supplies must be integers.
- Node ids are dense integers you assign; keep a mapping to real names.
- Total supply must equal total demand for `SimpleMinCostFlow` (add a dummy
  node to absorb imbalance if needed).

## Max flow

```python
from ortools.graph.python import max_flow

smf = max_flow.SimpleMaxFlow()
arcs = [(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 3), (1, 2, 1)]  # tail, head, cap
for tail, head, capacity in arcs:
    smf.add_arc_with_capacity(tail, head, capacity)

status = smf.solve(0, 3)  # source, sink
assert status == smf.OPTIMAL
print("max flow:", smf.optimal_flow())
for i in range(smf.num_arcs()):
    print(f"  {smf.tail(i)} -> {smf.head(i)}: {smf.flow(i)}/{smf.capacity(i)}")
assert smf.optimal_flow() == 5
```

## Min-cost flow (transportation problems)

```python
from ortools.graph.python import min_cost_flow

mcf = min_cost_flow.SimpleMinCostFlow()
# tail, head, capacity, unit cost
for tail, head, cap, cost in [(0, 1, 4, 4), (0, 2, 4, 2), (1, 3, 4, 1), (2, 3, 4, 3)]:
    mcf.add_arc_with_capacity_and_unit_cost(tail, head, cap, cost)
mcf.set_node_supply(0, 4)    # positive = supply
mcf.set_node_supply(3, -4)   # negative = demand

status = mcf.solve()
assert status == mcf.OPTIMAL
print("min cost:", mcf.optimal_cost())
for i in range(mcf.num_arcs()):
    if mcf.flow(i) > 0:
        print(f"  {mcf.tail(i)} -> {mcf.head(i)}: {mcf.flow(i)} units")
assert mcf.optimal_cost() == 20
```

## Linear sum assignment (one worker per task)

```python
from ortools.graph.python import linear_sum_assignment

costs = [[90, 76, 75], [35, 85, 55], [125, 95, 90]]
assignment = linear_sum_assignment.SimpleLinearSumAssignment()
for worker in range(3):
    for task in range(3):
        assignment.add_arc_with_cost(worker, task, costs[worker][task])

status = assignment.solve()
assert status == assignment.OPTIMAL
print("total cost:", assignment.optimal_cost())
for worker in range(3):
    print(f"  worker {worker} -> task {assignment.right_mate(worker)}")
assert assignment.optimal_cost() == 201
```

When the assignment has side constraints (capacities above 1, forbidden pairs
with logic, fairness terms), fall back to CP-SAT or MILP with binary variables;
the dedicated solver only handles the pure one-to-one min-cost case.
