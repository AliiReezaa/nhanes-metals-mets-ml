#!/bin/bash
# Double-click this file to run the whole pipeline on macOS.
# First time only, if macOS refuses to open it: right-click > Open, or run
#   chmod +x RUN_ME_macOS.command   in Terminal from this folder.
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
  python3 RUN_ME.py "$@"
else
  echo "Python 3 was not found. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ and double-click this file again."
  read -r -p "Press Enter to close..."
fi
