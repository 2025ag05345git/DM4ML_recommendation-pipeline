#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
if [ ! -x "$PYTHON" ]; then
    printf 'Python executable not found: %s\n' "$PYTHON" >&2
    exit 1
fi
cd "$ROOT"
exec "$PYTHON" dags/workflow.py
