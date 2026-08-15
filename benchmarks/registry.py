"""The benchmark instance registry.

Instances are classic OR-Library problems (J.E. Beasley's OR-Library,
https://people.brunel.ac.uk/~mastjjb/jeb/info.html):

- Job shop: ft06 (Fisher & Thompson 1963), la01 (Lawrence 1984), orb01
  (Applegate & Cook 1991), in the standard JSP format, mirrored from JSPLIB
  (github.com/tamy0612/JSPLIB) together with the curated known optima.
- Multidimensional knapsack: Petersen (1967) instances from OR-Library's
  mknap1 set; the known optimum is embedded in each file's header.

Every known optimum in this registry is additionally re-proven offline by
``validate.py`` with exact OR-Tools solves before the live battery runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

INSTANCE_DIR = Path(__file__).parent / "instances"

JSP_PROMPT = """\
The file {filename} in the working directory is a job-shop scheduling instance
in the standard OR-Library / JSPLIB format: lines starting with '#' are
comments; the first data line has two integers, the number of jobs n and the
number of machines m; then n lines follow, one per job, each with m pairs
"machine duration" giving that job's operations in processing order (machines
are 0-indexed). Each machine processes one operation at a time; operations of
a job run in the given order; no preemption.

Find a schedule minimizing the makespan, and prove optimality if you can.
Use your formulate/solve/verify pipeline. End your final answer with a line
of exactly this form: OBJECTIVE: <number>
"""

MKNAP_PROMPT = """\
The file {filename} in the working directory is a multidimensional 0/1
knapsack instance in the OR-Library mknap1 format: the first line has three
numbers "n m opt" (n items, m constraints, and a reported best-known value
which you must IGNORE and not consult while solving); the next numbers (across
one or more lines) are the n profits; then the m x n constraint coefficient
matrix row by row (row i = coefficients of all n items in constraint i); then
the m right-hand-side capacities.

Choose a subset of items maximizing total profit subject to all m capacity
constraints (each item chosen at most once), and prove optimality if you can.
Use your formulate/solve/verify pipeline. End your final answer with a line
of exactly this form: OBJECTIVE: <number>
"""


@dataclass(frozen=True)
class Instance:
    name: str
    family: str  # "jsp" | "mknap"
    filename: str
    known_optimum: float
    sense: str  # "min" | "max"
    prompt_template: str

    @property
    def path(self) -> Path:
        return INSTANCE_DIR / self.filename

    def prompt(self) -> str:
        return self.prompt_template.format(filename=self.filename)


INSTANCES: list[Instance] = [
    Instance("ft06", "jsp", "ft06.txt", 55, "min", JSP_PROMPT),
    Instance("la01", "jsp", "la01.txt", 666, "min", JSP_PROMPT),
    Instance("orb01", "jsp", "orb01.txt", 1059, "min", JSP_PROMPT),
    Instance("mknap01_2", "mknap", "mknap01_2.txt", 8706.1, "max", MKNAP_PROMPT),
    Instance("mknap01_5", "mknap", "mknap01_5.txt", 12400, "max", MKNAP_PROMPT),
    Instance("mknap01_7", "mknap", "mknap01_7.txt", 16537, "max", MKNAP_PROMPT),
]


def get(name: str) -> Instance:
    for inst in INSTANCES:
        if inst.name == name:
            return inst
    raise KeyError(name)
