"""Verified OR-Tools cookbook, served to agents via the read_recipe tool.

Every code block in these recipes is executed by the test suite against the
installed OR-Tools version, so the idioms agents copy are current API fact,
not training-data recollection. Add a recipe: drop a .md file here, register
it in RECIPES, and the tests pick it up automatically.
"""

from __future__ import annotations

from importlib import resources

# name -> one-line summary (shown in the read_recipe tool description)
RECIPES: dict[str, str] = {
    "cpsat_basics": (
        "CP-SAT fundamentals: current snake_case API, solver parameters (time "
        "limit, workers), status handling, reified constraints, enumerating all "
        "solutions, and diagnosing INFEASIBLE with assumptions."
    ),
    "scheduling": (
        "Scheduling with CP-SAT: interval variables, no-overlap (job shop / "
        "disjunctive machines), precedence, makespan via add_max_equality, and "
        "cumulative resources (RCPSP)."
    ),
    "routing_vrp": (
        "TSP/VRP with the routing library: RoutingIndexManager/RoutingModel, "
        "transit callbacks, capacity dimensions, time windows, search "
        "parameters, and extracting routes."
    ),
    "network_flows": (
        "Graph optimization: max flow, min-cost flow, and the dedicated "
        "linear sum assignment solver."
    ),
    "linear_milp": (
        "LP and MILP with pywraplp: solver backend choice (GLOP vs SCIP/CBC), "
        "and LP sensitivity analysis (dual values / shadow prices, reduced "
        "costs)."
    ),
    "knapsack_binpacking": (
        "The dedicated multidimensional knapsack solver, and bin packing as a "
        "CP-SAT model."
    ),
}


def recipe_catalog() -> str:
    return "\n".join(f"- {name}: {summary}" for name, summary in RECIPES.items())


def read_recipe(name: str) -> str:
    """Return the full text of a recipe; raises KeyError for unknown names."""
    if name not in RECIPES:
        raise KeyError(name)
    return (resources.files(__name__) / f"{name}.md").read_text(encoding="utf-8")
