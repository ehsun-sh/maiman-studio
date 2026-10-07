#!/usr/bin/env sh
# Run once: puts a "Maiman Studio" shortcut with the Maiman icon in this folder
# and opens the studio. Use that shortcut from then on. On macOS,
# make-shortcut.command does the same from Finder.
set -e
cd "$(dirname "$0")"
python="${MAIMAN_PYTHON:-python3}"
if ! "$python" -c "import numpy" >/dev/null 2>&1; then
  echo "Maiman needs Python 3.11+ with NumPy. Install it with:" >&2
  echo "  $python -m pip install -e ." >&2
  exit 1
fi
"$python" tools/make_shortcut.py
exec "$python" tools/maiman_app.py
