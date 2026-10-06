"""The session server as a standalone program, for the desktop build to freeze.

PyInstaller needs a script, not a console entry point, and this is that script:
``maiman serve`` with whatever arguments the desktop shell passes. The shell
asks for ``--port 0`` and reads the URL the server prints, so two copies of the
application, or an application next to a ``maiman serve`` already running in a
terminal, never fight over 8765.
"""

import sys

from maiman.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["serve", *sys.argv[1:]]))
