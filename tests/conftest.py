"""Put ``examples/`` on the import path for the tests that check them.

The examples are deliverables rather than decoration — mypy already type-checks
them, the studio ships a project exported by one, and the reference designs are
among the things this project is *for*. A test that rebuilt one of them instead
of importing it would be testing a copy, and a copy is exactly the thing that
goes stale.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
