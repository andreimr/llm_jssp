import pytest
from ortools.constraint_solver import pywrapcp


@pytest.fixture
def ortools_fixture_1():
    # Code to set up the test fixture
    # This code will run before each test function

    # Example: Creating objects, initializing variables, etc.
    solver = pywrapcp.Solver("jobshop")

    yield solver # Optional: If you need to perform any cleanup after each test, you can place it here

    # Code to clean up the test fixture
    # This code will run after each test function

    # Example: Closing connections, deleting temporary files, etc.

def test_ortools_on_jssp_1(ortools_fixture_1: pywrapcp.Solver):
    # Get the solver.
    solver = ortools_fixture_1

    machines_count = 3  # number of machines
    jobs_count = 3  # number of jobs
    all_machines = range(0, machines_count)
    all_jobs = range(0, jobs_count)

    # Define data.
    machines = [[0, 1, 2],
                [0, 2, 1],
                [1, 2, 0]]
    processing_times = [[3, 2, 2],
                        [2, 1, 4],
                        [4, 3, 4]]

    # Computes horizon.
    horizon = sum([sum(processing_times[i]) for i in all_jobs])

    # Creates jobs.
    all_tasks = {}
    for i in all_jobs:
        for j in all_machines:
            all_tasks[(i, j)] = solver.FixedDurationIntervalVar(0,
                                                                horizon,
                                                                processing_times[i][j],
                                                                False,
                                                                "Task_%i_%i" % (i, j))

    # Creates sequence variables and add disjunctive constraints.
    all_sequences = []
    all_machines_jobs = []
    for i in all_machines:

        machines_jobs = [all_tasks[(j, i)] for j in all_jobs]
        all_machines_jobs.append(machines_jobs)

        disj = solver.DisjunctiveConstraint(machines_jobs, "machine %i" % i)
        solver.Add(disj)
        all_sequences.append(disj.SequenceVar())

    # Add conjunctive contraints.
    for i in all_jobs:
        for j in range(0, machines_count - 1):
            solver.Add(all_tasks[(i, j)].EndExpr() <= all_tasks[(i, j + 1)].StartExpr())

    # Set the objective.
    obj_var = solver.Max([all_tasks[(i, machines_count - 1)].EndExpr() for i in all_jobs])
    objective_monitor = solver.Minimize(obj_var, 1)
    # Create search phases.
    sequence_phase = solver.Phase([all_sequences[i] for i in all_machines], solver.SEQUENCE_DEFAULT)
    vars_phase = solver.Phase([obj_var], solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
    main_phase = solver.Compose([sequence_phase, vars_phase])
    # Create the solution collector.
    collector = solver.LastSolutionCollector()

    # Add the interesting variables to the SolutionCollector.
    collector.Add(all_sequences)
    collector.AddObjective(obj_var)

    for i in all_machines:
        sequence = all_sequences[i]
        sequence_count = sequence.Size()
        for j in range(0, sequence_count):
            t = sequence.Interval(j)
            collector.Add(t.StartExpr().Var())
            collector.Add(t.EndExpr().Var())
    # Solve the problem.
    disp_col_width = 10
    if solver.Solve(main_phase, [objective_monitor, collector]):
        print("\nOptimal Schedule Length:", collector.ObjectiveValue(0))
        sol_line = ""
        sol_line_tasks = ""
        print("Optimal Schedule", "\n")

        for i in all_machines:
            seq = all_sequences[i]
            sol_line += "Machine " + str(i) + ": "
            sol_line_tasks += "Machine " + str(i) + ": "
            sequence = collector.ForwardSequence(0, seq)
            seq_size = len(sequence)

            for j in range(0, seq_size):
                t = seq.Interval(sequence[j])
                # Add spaces to output to align columns.
                sol_line_tasks +=  t.Name() + " " * (disp_col_width - len(t.Name()))

            for j in range(0, seq_size):
                t = seq.Interval(sequence[j])
                sol_tmp = "[" + str(collector.Value(0, t.StartExpr().Var())) + ","
                sol_tmp += str(collector.Value(0, t.EndExpr().Var())) + "] "
                # Add spaces to output to align columns.
                sol_line += sol_tmp + " " * (disp_col_width - len(sol_tmp))

            sol_line += "\n"
            sol_line_tasks += "\n"

        print(sol_line_tasks)
        print("Time Intervals for Tasks\n")
        print(sol_line)
    else:
        assert False, "No solution found!"


if __name__ == "__main__":
    pytest.main()
