#!/usr/bin/env sh
# macOS: double-click in Finder to open Maiman Studio. Finder runs .command
# files in Terminal; this hands over to start-studio.sh, which does the work.
exec "$(dirname "$0")/start-studio.sh"
