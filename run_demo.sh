#!/usr/bin/env bash
# One-command demo: API + (optional) mock agent + dashboard.
#   ./run_demo.sh            API + dashboard + mock agent
#   ./run_demo.sh --no-agent API + dashboard only (use when the LangGraph agent is connected)
#   ./run_demo.sh --cocoon   also stream EXC001 into the Cocoon backend. Uses the backend at COCOON_BACKEND_URL
#                            (default http://127.0.0.1:8000); if none is running, starts the local copy in cocoon/.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "Create the venv first: uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt"; exit 1; }
export OMP_NUM_THREADS=1

AGENT=1; COCOON=0
for arg in "$@"; do
  case "$arg" in
    --no-agent) AGENT=0 ;;
    --cocoon) COCOON=1 ;;
    *) echo "unknown option $arg"; exit 1 ;;
  esac
done

if [ ! -f models/health_models.joblib ]; then
  echo "Training models (first run only, ~3 min)..."
  "$PY" -m ml.train
fi

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

.venv/bin/uvicorn server.api:app --port 8100 --log-level warning &
for _ in $(seq 1 60); do curl -s localhost:8100/api/status >/dev/null && break; sleep 0.5; done
echo "API      http://localhost:8100/docs"

if [ "$COCOON" = 1 ]; then
  export COCOON_SERVICE_TOKEN="${COCOON_SERVICE_TOKEN:-dev-local-change-me}"
  BACKEND="${COCOON_BACKEND_URL:-http://127.0.0.1:8000}"
  if ! curl -s "$BACKEND/readyz" >/dev/null; then
    scripts/run_cocoon_backend.sh &
    echo "Cocoon   starting the local backend copy (cocoon/langgraph-agent)..."
    for _ in $(seq 1 360); do curl -s "$BACKEND/readyz" >/dev/null && break; sleep 0.5; done
  fi
  echo "Cocoon   backend $BACKEND (docs at $BACKEND/docs)"
  "$PY" -u scripts/cocoon_bridge.py &
  echo "Cocoon   bridge streaming EXC001 -> EXC_DEMO_001: telemetry + detected alerts"
fi

if [ "$AGENT" = 1 ]; then
  "$PY" scripts/mock_agent.py &
  echo "Agent    mock agent attached (replace with LangGraph; see docs/INTEGRATION.md)"
fi

echo "Demo UI  http://localhost:8501"
.venv/bin/streamlit run app/dashboard.py --server.port 8501
