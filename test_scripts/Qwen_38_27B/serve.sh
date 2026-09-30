#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
mkdir -p runtime
action=${1:-status}
PY=/mnt/bn/vai-llm-hl/ruipeng/.venvs/vllm-0.26.0/bin/python
MODEL=/mnt/bn/vai-llm-hl/ruipeng/text_model_qwen3_8_27b
if [[ "$action" == start ]]; then
  "$PY" - <<'PY'
import json, pathlib, os, signal
r=pathlib.Path('runtime'); saved=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        args=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if '/gpu/gpu_keepalive.py --gpu-index ' in args:
            gpu=int(args.split('--gpu-index ')[1].split()[0])
            if gpu in range(4):
                saved.append({'pid':int(p.name),'args':args})
                os.kill(int(p.name),signal.SIGSTOP)
    except (OSError,ValueError): pass
(r/'paused_keepalive.json').write_text(json.dumps(saved))
print('Paused keepalive processes:',len(saved))
PY
  for replica in 0 1; do
    port=$((8101+replica)); g0=$((replica*2)); g1=$((g0+1))
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null; then continue; fi
    [[ ! -f runtime/server-$replica.pid ]] || ! kill -0 "$(cat runtime/server-$replica.pid)" 2>/dev/null || continue
    nohup setsid env CUDA_VISIBLE_DEVICES="$g0,$g1" \
      LD_LIBRARY_PATH=/mnt/bn/vai-llm-hl2/ruipeng/runtime/cuda-compat-13-0/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
      NCCL_DEBUG=WARN VLLM_WORKER_MULTIPROC_METHOD=spawn \
      TRITON_CACHE_DIR="/home/tiger/.cache/cot-baseline-$replica/triton" \
      TORCHINDUCTOR_CACHE_DIR="/home/tiger/.cache/cot-baseline-$replica/torchinductor" \
      "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
      --host 127.0.0.1 --port "$port" --served-model-name qwen38-27b-base \
      --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 65536 \
      --max-num-seqs 32 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.88 \
      --enable-prefix-caching --enable-chunked-prefill --gdn-prefill-backend triton \
      --language-model-only --reasoning-parser qwen3 \
      --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"medium"}' \
      --compilation-config '{"mode":3}' >"runtime/server-$replica.log" 2>&1 </dev/null &
    echo "$!" >"runtime/server-$replica.pid"
    echo "Starting replica $replica on GPUs $g0,$g1, port $port, pid $!"
  done
elif [[ "$action" == stop ]]; then
  "$PY" - <<'PY'
import pathlib,os,signal,json
r=pathlib.Path('runtime')
for p in r.glob('server-*.pid'):
    pid=int(p.read_text())
    try:
        args=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
        if b'qwen38-27b-base' in args: os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError: pass
    except FileNotFoundError: pass
    p.unlink(missing_ok=True)
p=r/'paused_keepalive.json'
if p.exists():
    for v in json.loads(p.read_text()):
        try:
            args=pathlib.Path(f'/proc/{v["pid"]}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            if args==v['args']: os.kill(v['pid'],signal.SIGCONT)
        except (OSError,ProcessLookupError): pass
    p.unlink()
PY
else
  for port in 8101 8102; do curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" || true; done
fi
