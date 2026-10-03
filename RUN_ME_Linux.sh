#!/bin/bash
# Run the whole pipeline on Linux:  ./RUN_ME_Linux.sh   (or: bash RUN_ME_Linux.sh)
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
  python3 RUN_ME.py "$@"
else
  echo "Python 3 was not found. Install it with your package manager (for example: sudo apt install python3 python3-venv) and run again."
fi
