from optimist.tools.sandbox import run_python


def test_stdout_capture():
    result = run_python("print('hello', 6 * 7)")
    assert result.returncode == 0
    assert "hello 42" in result.stdout
    assert not result.timed_out


def test_error_capture():
    result = run_python("raise ValueError('boom')")
    assert result.returncode != 0
    assert "ValueError: boom" in result.stderr
    text, is_error = result.as_tool_result()
    assert is_error
    assert "boom" in text


def test_timeout():
    result = run_python("import time; time.sleep(30)", timeout_s=2)
    assert result.timed_out
    text, is_error = result.as_tool_result()
    assert is_error
    assert "TIMED OUT" in text


def test_ortools_importable_in_sandbox():
    result = run_python(
        "from ortools.sat.python import cp_model\n"
        "m = cp_model.CpModel()\n"
        "x = m.new_int_var(0, 10, 'x')\n"
        "m.add(x >= 7)\n"
        "m.minimize(x)\n"
        "s = cp_model.CpSolver()\n"
        "print(s.solve(m) == cp_model.OPTIMAL, s.value(x))"
    )
    assert result.returncode == 0, result.stderr
    assert "True 7" in result.stdout


def test_output_truncation():
    result = run_python("print('x' * 100_000)")
    assert len(result.stdout) < 50_000
    assert "truncated" in result.stdout
