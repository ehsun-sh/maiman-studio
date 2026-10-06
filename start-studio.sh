#!/usr/bin/env sh
# Open Maiman Studio as a desktop application, straight from this checkout.
#
# Double-click it (or run ./start-studio.sh). The first run installs Electron
# into desktop/node_modules, which takes a minute; every run after that opens
# the window directly. Needs Node.js and a Python that has NumPy — the same
# Python `maiman serve` would use. On macOS, start-studio.command does the same
# thing from Finder.

set -e
cd "$(dirname "$0")"

say() { printf '%s\n' "$*" >&2; }

if ! command -v npm >/dev/null 2>&1; then
  say "Maiman Studio needs Node.js to open as a desktop app."
  say "Install it from https://nodejs.org (the LTS version), then run this again."
  exit 1
fi

python="${MAIMAN_PYTHON:-python3}"
if ! command -v "$python" >/dev/null 2>&1; then
  say "Maiman Studio needs Python 3.11 or newer: https://www.python.org/downloads/"
  exit 1
fi
if ! "$python" -c "import numpy" >/dev/null 2>&1; then
  say "The engine needs NumPy, and $python does not have it. Install it with:"
  say "  $python -m pip install -e ."
  exit 1
fi

cd desktop
if [ ! -d node_modules/electron ]; then
  say "First run: installing Electron (once)..."
  npm install --no-audit --no-fund
fi
export MAIMAN_PYTHON="$python"
exec npm start --silent
