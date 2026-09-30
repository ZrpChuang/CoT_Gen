#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
exec /home/tiger/.venvs/cot-analyse-ssr/bin/python ../evaluate.py --bench arena_hard_v2 "$@"
