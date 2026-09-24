#!/usr/bin/env bash
# Run the Cocoon backend copy (cocoon/langgraph-agent) on :8000 in mock LLM mode.
#   scripts/run_cocoon_backend.sh
# First run creates its own Python 3.11 venv (the backend never shares ours). Needs the team dataset
# linked at cocoon/Cocoon_Dataset_v1. Its SQLite files live in cocoon/langgraph-agent/data/.
set -euo pipefail
cd "$(dirname "$0")/../cocoon/langgraph-agent"
if [ ! -x .venv/bin/python ]; then
  echo "Creating the backend venv (Python 3.11, first run only)..."
  uv venv --python 3.11 .venv -q
  uv pip install --python .venv/bin/python -q -r requirements-dev.txt
  uv pip install --python .venv/bin/python -q -e . --no-deps
fi
[ -e ../Cocoon_Dataset_v1/data/generated/manifest.json ] || {
  echo "Link the team dataset first: ln -s /path/to/Cocoon_Dataset_v1 cocoon/Cocoon_Dataset_v1"; exit 1; }
export COCOON_SERVICE_TOKEN="${COCOON_SERVICE_TOKEN:-dev-local-change-me}"
export COCOON_LLM_MODE="${COCOON_LLM_MODE:-mock}"
exec .venv/bin/python -m cocoon_agent
