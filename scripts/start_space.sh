#!/usr/bin/env bash
set -euo pipefail

uvicorn api.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!
ui_pid=""

cleanup() {
  if [[ -n "$ui_pid" ]]; then kill "$ui_pid" 2>/dev/null || true; fi
  kill "$api_pid" 2>/dev/null || true
  wait "$api_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _attempt in $(seq 1 90); do
  health="$(curl --silent --fail http://127.0.0.1:8000/health || true)"
  if [[ "$health" == *'"model_ready":true'* ]]; then
    break
  fi
  if ! kill -0 "$api_pid" 2>/dev/null; then
    wait "$api_pid"
    echo "FastAPI exited before becoming healthy." >&2
    exit 1
  fi
  sleep 1
done

if [[ "$health" != *'"model_ready":true'* ]]; then
  echo "FastAPI model service did not become ready within 90 seconds." >&2
  exit 1
fi

streamlit run streamlit/app.py --server.address 0.0.0.0 --server.port 7860 &
ui_pid=$!
set +e
wait -n "$api_pid" "$ui_pid"
status=$?
set -e
exit "$status"
