"""``python -m maiman``, which is the same command as ``maiman``.

An installed script can be missing from ``PATH`` for reasons that have nothing
to do with this project — a user-site install, a virtual environment that was
never activated. ``python -m`` needs only the interpreter that imported the
package, so it is the fallback that always exists.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
