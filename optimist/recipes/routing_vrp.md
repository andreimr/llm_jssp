# TSP and VRP with the routing library

Use `ortools.constraint_solver.pywrapcp`'s `RoutingModel` for travelling
salesman and vehicle routing problems: capacities, time windows, pickup and
delivery, multiple depots. For small instances (< ~15 nodes) a CP-SAT
`add_circuit` model also works, but the routing library scales far better and
ships strong metaheuristics.

Current-API notes (OR-Tools 9.x):
- This library keeps the CamelCase API (`RegisterTransitCallback`,
  `SolveWithParameters`); there is no snake_case variant here.
- The model works on internal *indices*, not your node ids; always translate
  with `manager.IndexToNode(index)` inside callbacks and when reading routes.
- Costs and callback returns must be integers; scale distances if fractional.
- Always set a time limit and a metaheuristic
  (`GUIDED_LOCAL_SEARCH` is the usual choice); the first solution strategy
  alone is just a construction heuristic.
- `SolveWithParameters` returns `None` when no solution was found; check it.

## Capacitated VRP

```python
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

distance = [
    [0, 2, 4, 6, 8],
    [2, 0, 3, 5, 7],
    [4, 3, 0, 2, 4],
    [6, 5, 2, 0, 3],
    [8, 7, 4, 3, 0],
]
demands = [0, 2, 3, 4, 3]  # node 0 is the depot
NUM_VEHICLES, CAPACITY, DEPOT = 2, 6, 0

manager = pywrapcp.RoutingIndexManager(len(distance), NUM_VEHICLES, DEPOT)
routing = pywrapcp.RoutingModel(manager)


def distance_callback(from_index, to_index):
    return distance[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]


transit_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_index)


def demand_callback(from_index):
    return demands[manager.IndexToNode(from_index)]


demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_index,
    0,                            # no slack
    [CAPACITY] * NUM_VEHICLES,    # per-vehicle capacities
    True,                         # start cumul at zero
    "Capacity",
)

params = pywrapcp.DefaultRoutingSearchParameters()
params.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
params.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
params.time_limit.FromSeconds(5)

solution = routing.SolveWithParameters(params)
assert solution is not None, "no solution found - relax constraints or raise the time limit"

total = 0
for vehicle in range(NUM_VEHICLES):
    index = routing.Start(vehicle)
    route = []
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        next_index = solution.Value(routing.NextVar(index))
        total += routing.GetArcCostForVehicle(index, next_index, vehicle)
        index = next_index
    route.append(manager.IndexToNode(index))
    print(f"vehicle {vehicle}: {route}")
print("total distance:", total)
assert total == 29
```

## Common extensions

- **Time windows**: add a "Time" dimension from a travel-time callback
  (`routing.AddDimension(transit, slack_max, horizon, False, "Time")`), then
  constrain each node with
  `time_dimension.CumulVar(manager.NodeToIndex(node)).SetRange(a, b)`.
- **Optional visits with penalties**:
  `routing.AddDisjunction([manager.NodeToIndex(node)], penalty)` lets the
  solver drop a visit at a price (also the standard way to make infeasible
  instances solvable and see which visits do not fit).
- **Pickup and delivery**: `routing.AddPickupAndDelivery(p, d)` plus same-
  vehicle and ordering constraints via the routing solver's `solver()`.
