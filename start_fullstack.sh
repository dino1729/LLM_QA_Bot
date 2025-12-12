#!/usr/bin/env bash
set -euo pipefail

# Bootstrap frontend build and start the Python backend that serves /api and frontend/dist.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# --- Python env and deps ---
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# --- Frontend install + build ---
pushd frontend >/dev/null
npm install
npm run build
popd >/dev/null

echo "Frontend built at frontend/dist. Starting backend server on http://0.0.0.0:7860 ..."
exec python gradio_ui_full.py
