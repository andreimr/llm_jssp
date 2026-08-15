import pytest

from optimist.tools.lp import LinearProblem, solve_linear_problem


def make(spec: dict) -> LinearProblem:
    return LinearProblem.model_validate(spec)


def test_continuous_lp_production_mix():
    # Classic two-product mix: maximize 3x + 5y s.t. x<=4, 2y<=12, 3x+2y<=18.
    problem = make(
        {
            "variables": [
                {"name": "x", "lower_bound": 0},
                {"name": "y", "lower_bound": 0},
            ],
            "constraints": [
                {"coefficients": {"x": 1}, "sense": "<=", "rhs": 4},
                {"coefficients": {"y": 2}, "sense": "<=", "rhs": 12},
                {"coefficients": {"x": 3, "y": 2}, "sense": "<=", "rhs": 18},
            ],
            "objective": {"sense": "maximize", "coefficients": {"x": 3, "y": 5}},
        }
    )
    solution = solve_linear_problem(problem)
    assert solution.status == "OPTIMAL"
    assert solution.objective_value == pytest.approx(36.0)
    assert solution.variables["x"] == pytest.approx(2.0)
    assert solution.variables["y"] == pytest.approx(6.0)


def test_binary_knapsack():
    values = {"a": 60, "b": 100, "c": 120}
    weights = {"a": 10, "b": 20, "c": 30}
    problem = make(
        {
            "variables": [{"name": n, "type": "binary"} for n in values],
            "constraints": [{"coefficients": weights, "sense": "<=", "rhs": 50}],
            "objective": {"sense": "maximize", "coefficients": values},
        }
    )
    solution = solve_linear_problem(problem)
    assert solution.status == "OPTIMAL"
    assert solution.objective_value == pytest.approx(220.0)
    assert solution.variables["a"] == pytest.approx(0.0)
    assert solution.variables["b"] == pytest.approx(1.0)
    assert solution.variables["c"] == pytest.approx(1.0)


def test_integer_variables_are_integral():
    problem = make(
        {
            "variables": [{"name": "x", "type": "integer", "lower_bound": 0}],
            "constraints": [{"coefficients": {"x": 2}, "sense": "<=", "rhs": 7}],
            "objective": {"sense": "maximize", "coefficients": {"x": 1}},
        }
    )
    solution = solve_linear_problem(problem)
    assert solution.status == "OPTIMAL"
    assert solution.variables["x"] == pytest.approx(3.0)


def test_infeasible_is_reported_not_raised():
    problem = make(
        {
            "variables": [{"name": "x", "lower_bound": 0, "upper_bound": 1}],
            "constraints": [{"coefficients": {"x": 1}, "sense": ">=", "rhs": 2}],
            "objective": {"sense": "minimize", "coefficients": {"x": 1}},
        }
    )
    assert solve_linear_problem(problem).status == "INFEASIBLE"


def test_objective_constant_offset():
    problem = make(
        {
            "variables": [{"name": "x", "lower_bound": 0, "upper_bound": 5}],
            "objective": {
                "sense": "maximize",
                "coefficients": {"x": 2},
                "constant": 100,
            },
        }
    )
    assert solve_linear_problem(problem).objective_value == pytest.approx(110.0)


def test_unknown_variable_reference_rejected():
    problem = make(
        {
            "variables": [{"name": "x"}],
            "constraints": [{"coefficients": {"ghost": 1}, "sense": "<=", "rhs": 1}],
            "objective": {"sense": "minimize", "coefficients": {"x": 1}},
        }
    )
    with pytest.raises(ValueError, match="ghost"):
        solve_linear_problem(problem)


def test_duplicate_variable_names_rejected():
    problem = make(
        {
            "variables": [{"name": "x"}, {"name": "x"}],
            "objective": {"sense": "minimize", "coefficients": {"x": 1}},
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        solve_linear_problem(problem)
