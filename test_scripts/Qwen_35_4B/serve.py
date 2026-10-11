#!/usr/bin/env python3
"""Own two single-GPU vLLM replicas."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / 'deployment.json').read_text())
NAME = json.loads((ROOT / 'config.json').read_text())['target']['model']
RUN = ROOT / 'runtime'
RUN.mkdir(exist_ok=True)
STATE = RUN / 'service_state.json'


def cmdline(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except OSError:
        return ''


def stop():
    if not STATE.exists():
        return
    state = json.loads(STATE.read_text())
    for p in state['servers']:
        if NAME in cmdline(p['pid']):
            os.killpg(p['pid'], signal.SIGTERM)
    for _ in range(30):
        if not any(NAME in cmdline(p['pid']) for p in state['servers']):
            break
        time.sleep(1)
    for p in state['servers']:
        if NAME in cmdline(p['pid']):
            os.killpg(p['pid'], signal.SIGKILL)
    state['stopped_at'] = time.time()
    (RUN / 'service_history.json').write_text(json.dumps(state, indent=2) + '\n')
    STATE.unlink()
    print('Services stopped', flush=True)


def start():
    if STATE.exists():
        raise RuntimeError('Service state exists; inspect or stop before restart')
    assert (RUN / 'download_complete.json').exists(), 'Model download is not verified'
    state = {'started_at': time.time(), 'servers': []}
    STATE.write_text(json.dumps(state))
    try:
        for gpu, port in zip(C['gpus'], C['ports']):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', NCCL_DEBUG='WARN', VLLM_WORKER_MULTIPROC_METHOD='spawn')
            env['LD_LIBRARY_PATH'] = '/mnt/bn/vai-llm-hl2/ruipeng/runtime/cuda-compat-13-0/usr/local/cuda-13.0/compat' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
            env['TRITON_CACHE_DIR'] = f'/home/tiger/.cache/{NAME}-{gpu}/triton'
            env['TORCHINDUCTOR_CACHE_DIR'] = f'/home/tiger/.cache/{NAME}-{gpu}/torchinductor'
            args = [C['python'], '-m', 'vllm.entrypoints.cli.main', 'serve', C['model_dir'], '--host', '127.0.0.1', '--port', str(port), '--served-model-name', NAME,
                    '--dtype', 'bfloat16', '--tensor-parallel-size', '1', '--max-model-len', '65536', '--max-num-seqs', '32', '--max-num-batched-tokens', '16384', '--gpu-memory-utilization', '0.88',
                    '--enable-prefix-caching', '--enable-chunked-prefill', '--gdn-prefill-backend', 'triton', '--language-model-only', '--reasoning-parser', 'qwen3',
                    '--default-chat-template-kwargs', '{"enable_thinking":true}', '--compilation-config', '{"mode":3}']
            with (RUN / f'server-{gpu}.log').open('a') as log:
                proc = subprocess.Popen(args, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            state['servers'].append({'pid': proc.pid, 'gpu': gpu, 'port': port, 'args': args})
            STATE.write_text(json.dumps(state, indent=2))
            print('Started GPU', gpu, 'port', port, 'pid', proc.pid, flush=True)
    except BaseException:
        stop()
        raise


if __name__ == '__main__':
    {'start': start, 'stop': stop}[sys.argv[1]]()
