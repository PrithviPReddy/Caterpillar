#!/usr/bin/env bash
# One-command demo: API + (optional) mock agent + dashboard.
#   ./run_demo.sh            API + dashboard + mock agent
#   ./run_demo.sh --no-agent API + dashboard only (use when the LangGraph agent is connected)
#   ./run_demo.sh --cocoon   also stream EXC001 into the Cocoon backend (needs COCOON_SERVICE_TOKEN;
#                            backend URL from COCOON_BACKEND_URL, default http://127.0.0.1:8000)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "Create the venv first: uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt"; exit 1; }
export OMP_NUM_THREADS=1

if [ ! -f models/health_models.joblib ]; then
  echo "Training models (first run only, ~3 min)..."
  "$PY" -m ml.train
fi

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

.venv/bin/uvicorn server.api:app --port 8100 --log-level warning &
for _ in $(seq 1 60); do curl -s localhost:8100/api/status >/dev/null && break; sleep 0.5; done
echo "API      http://localhost:8100/docs"

AGENT=1; COCOON=0
for arg in "$@"; do
  case "$arg" in
    --no-agent) AGENT=0 ;;
    --cocoon) COCOON=1 ;;
    *) echo "unknown option $arg"; exit 1 ;;
  esac
done

if [ "$COCOON" = 1 ]; then
  "$PY" -u scripts/cocoon_bridge.py &
  echo "Cocoon   bridge streaming EXC001 -> EXC_DEMO_001 (see scripts/cocoon_bridge.py)"
fi

if [ "$AGENT" = 1 ]; then
  "$PY" scripts/mock_agent.py &
  echo "Agent    mock agent attached (replace with LangGraph; see docs/INTEGRATION.md)"
fi

echo "Demo UI  http://localhost:8501"
.venv/bin/streamlit run app/dashboard.py --server.port 8501
