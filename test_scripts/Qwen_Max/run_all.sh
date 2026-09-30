#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
mkdir -p runtime
exec 9>runtime/suite.lock
flock -n 9 || { echo 'Suite already running'; exit 1; }
PY=/home/tiger/.venvs/cot-analyse-ssr/bin/python
for bench in ifeval multi_challenge eq_bench3 arena_hard_v2; do
  "$PY" evaluate.py --bench "$bench" --stage generate
  if [[ "$bench" == ifeval ]]; then "$PY" evaluate.py --bench "$bench" --stage score; fi
done
for bench in multi_challenge eq_bench3 arena_hard_v2; do
  "$PY" evaluate.py --bench "$bench" --stage score
done
