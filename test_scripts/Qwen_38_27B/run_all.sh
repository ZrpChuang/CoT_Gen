#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
mkdir -p runtime
exec 9>runtime/suite.lock
flock -n 9 || { echo 'Suite already running'; exit 1; }
PY=/home/tiger/.venvs/cot-analyse-ssr/bin/python
trap 'bash serve.sh stop' EXIT
ready=0
for attempt in $(seq 1 180); do
  if curl -fsS --max-time 2 http://127.0.0.1:8101/v1/models >/dev/null 2>&1 && curl -fsS --max-time 2 http://127.0.0.1:8102/v1/models >/dev/null 2>&1; then ready=1; break; fi
  sleep 10
done
[[ "$ready" == 1 ]] || { echo 'Servers did not become ready'; exit 1; }
"$PY" evaluate.py --bench ifeval --stage generate
"$PY" evaluate.py --bench ifeval --stage score
pids=()
for bench in multi_challenge eq_bench3 arena_hard_v2; do
  "$PY" evaluate.py --bench "$bench" --stage generate >"runtime/${bench}_generation.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]] || { echo 'Some response requests remain unresolved; rerun to resume.'; exit 1; }
echo 'All local model responses complete. GPU service will stop and keepalive will resume.'
