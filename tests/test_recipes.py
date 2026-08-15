"""The recipe library must be executable fact, not documentation drift:
every Python code block in every recipe runs against the installed OR-Tools.
"""

import re

import pytest

from optimist.recipes import RECIPES, read_recipe, recipe_catalog
from optimist.tools.sandbox import run_python

CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def code_blocks() -> list[tuple[str, int, str]]:
    blocks = []
    for name in RECIPES:
        for i, match in enumerate(CODE_BLOCK.finditer(read_recipe(name))):
            blocks.append((name, i, match.group(1)))
    return blocks


def test_registry_and_files_agree():
    assert RECIPES, "recipe registry is empty"
    for name in RECIPES:
        text = read_recipe(name)
        assert text.strip(), f"recipe {name} is empty"
        assert "```python" in text, f"recipe {name} has no code example"
    assert all(name in recipe_catalog() for name in RECIPES)


def test_unknown_recipe_raises():
    with pytest.raises(KeyError):
        read_recipe("does_not_exist")


@pytest.mark.parametrize(
    "name,index,code",
    code_blocks(),
    ids=[f"{n}[{i}]" for n, i, _ in code_blocks()],
)
def test_recipe_code_executes(name, index, code):
    result = run_python(code, timeout_s=90)
    assert result.returncode == 0, (
        f"recipe {name} block {index} failed:\n{result.stderr}"
    )
