"""Put ``examples/`` on the import path for the tests that check them.

The examples are deliverables rather than decoration — mypy already type-checks
them, the studio ships a project exported by one, and the reference designs are
among the things this project is *for*. A test that rebuilt one of them instead
of importing it would be testing a copy, and a copy is exactly the thing that
goes stale.

**And refuse to let the specification's checks skip when they were asked for.**
The W-Port vectors and shaping tables carry no licence to redistribute, so the
tests that read them skip wherever they are absent -- which is every CI runner.
A skip reads as a pass in a summary line. With ``MAIMAN_REQUIRE_SPEC=1`` the run
stops before collecting anything unless both are present and hold what the
tests will open, so a release check that was meant to cover the DPO path cannot
quietly not.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "python"))

#: What each variable must point at, as the tests and :mod:`maiman.pcs` open it.
SPECIFICATION = {
    "MAIMAN_OFEC_VECTORS": ("qpsk", "16qam", "b72", "b106", "b116"),
    "MAIMAN_WPORT_TABLES": ("SCS_LUT10.txt", "SCS_LUT11.txt", "rewire.json"),
}


def missing_specification() -> list[str]:
    """Each variable that is unset, or names a directory without what it should hold."""
    problems = []
    for variable, entries in SPECIFICATION.items():
        where = os.environ.get(variable)
        if not where:
            problems.append(f"{variable} is not set")
            continue
        absent = [entry for entry in entries if not (Path(where) / entry).exists()]
        if absent:
            problems.append(f"{variable}={where} has no {', '.join(absent)}")
    return problems


def pytest_sessionstart(session: pytest.Session) -> None:
    if os.environ.get("MAIMAN_REQUIRE_SPEC") != "1":
        return
    problems = missing_specification()
    if problems:
        pytest.exit(
            "MAIMAN_REQUIRE_SPEC=1, but the specification's data is not all here:\n  "
            + "\n  ".join(problems),
            returncode=pytest.ExitCode.USAGE_ERROR,
        )
