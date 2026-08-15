"""Structured LP/MILP solving.

The fast path for problems the orchestrator can express directly as a linear
(or mixed-integer linear) program: a validated pydantic spec goes straight
into OR-Tools, no code generation involved. Anything richer (disjunctions,
intervals, nonlinear terms, large data tables) goes through the Solver agent's
code sandbox instead.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Variable(BaseModel):
    name: str
    type: Literal["continuous", "integer", "binary"] = "continuous"
    lower_bound: float | None = None
    upper_bound: float | None = None

    @field_validator("name")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("variable name must be non-empty")
        return v


class Constraint(BaseModel):
    name: str = ""
    # linear expression: variable name -> coefficient
    coefficients: dict[str, float]
    sense: Literal["<=", ">=", "=="]
    rhs: float


class Objective(BaseModel):
    sense: Literal["minimize", "maximize"]
    coefficients: dict[str, float]
    constant: float = 0.0


class LinearProblem(BaseModel):
    """A linear or mixed-integer linear program."""

    variables: list[Variable] = Field(min_length=1)
    constraints: list[Constraint] = Field(default_factory=list)
    objective: Objective

    def variable_names(self) -> set[str]:
        return {v.name for v in self.variables}

    def validate_references(self) -> None:
        names = self.variable_names()
        if len(names) != len(self.variables):
            raise ValueError("duplicate variable names in problem spec")
        for i, c in enumerate(self.constraints):
            unknown = set(c.coefficients) - names
            if unknown:
                label = c.name or f"constraint[{i}]"
                raise ValueError(f"{label} references unknown variables: {sorted(unknown)}")
        unknown = set(self.objective.coefficients) - names
        if unknown:
            raise ValueError(f"objective references unknown variables: {sorted(unknown)}")


class LinearSolution(BaseModel):
    status: str  # OPTIMAL | FEASIBLE | INFEASIBLE | UNBOUNDED | ERROR
    objective_value: float | None = None
    variables: dict[str, float] = Field(default_factory=dict)
    solver: str = ""
    message: str = ""


def _make_solver(is_mip: bool):
    from ortools.linear_solver import pywraplp

    candidates = ["SCIP", "CBC", "SAT"] if is_mip else ["GLOP", "SCIP", "CBC"]
    for backend in candidates:
        solver = pywraplp.Solver.CreateSolver(backend)
        if solver is not None:
            return solver, backend
    raise RuntimeError("No OR-Tools linear solver backend available")


def solve_linear_problem(problem: LinearProblem, time_limit_s: float = 30.0) -> LinearSolution:
    """Solve a LinearProblem with OR-Tools and return a structured solution."""
    from ortools.linear_solver import pywraplp

    problem.validate_references()
    is_mip = any(v.type != "continuous" for v in problem.variables)
    solver, backend = _make_solver(is_mip)
    solver.SetTimeLimit(int(time_limit_s * 1000))
    inf = solver.infinity()

    var_map: dict[str, Any] = {}
    for v in problem.variables:
        if v.type == "binary":
            lo, hi = 0.0, 1.0
        else:
            lo = v.lower_bound if v.lower_bound is not None else -inf
            hi = v.upper_bound if v.upper_bound is not None else inf
        if v.type == "continuous":
            var_map[v.name] = solver.NumVar(lo, hi, v.name)
        else:
            var_map[v.name] = solver.IntVar(lo, hi, v.name)

    for c in problem.constraints:
        expr = solver.Sum(coef * var_map[name] for name, coef in c.coefficients.items())
        if c.sense == "<=":
            solver.Add(expr <= c.rhs)
        elif c.sense == ">=":
            solver.Add(expr >= c.rhs)
        else:
            solver.Add(expr == c.rhs)

    obj_expr = solver.Sum(
        coef * var_map[name] for name, coef in problem.objective.coefficients.items()
    )
    if problem.objective.sense == "minimize":
        solver.Minimize(obj_expr)
    else:
        solver.Maximize(obj_expr)

    status = solver.Solve()
    status_names = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ERROR",
        pywraplp.Solver.NOT_SOLVED: "ERROR",
    }
    name = status_names.get(status, "ERROR")
    if name in ("OPTIMAL", "FEASIBLE"):
        return LinearSolution(
            status=name,
            objective_value=solver.Objective().Value() + problem.objective.constant,
            variables={n: var.solution_value() for n, var in var_map.items()},
            solver=backend,
        )
    return LinearSolution(status=name, solver=backend, message=f"solver status: {name}")
