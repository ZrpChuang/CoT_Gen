#!/usr/bin/env python3
"""Own two single-GPU vLLM replicas and restore temporarily paused keepalives."""
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
    for p in state['keepalives']:
        if cmdline(p['pid']) == p['args']:
            os.kill(p['pid'], signal.SIGCONT)
    state['stopped_at'] = time.time()
    (RUN / 'service_history.json').write_text(json.dumps(state, indent=2) + '\n')
    STATE.unlink()
    print('Services stopped; keepalives restored', flush=True)


def start():
    if STATE.exists():
        raise RuntimeError('Service state exists; inspect or stop before restart')
    assert (RUN / 'download_complete.json').exists(), 'Model download is not verified'
    # Check that the selected GPUs contain no other workloads.
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {r.split(',')[1].strip(): int(r.split(',')[0]) for r in gpu_rows.splitlines() if int(r.split(',')[0]) in C['gpus']}
    keepalive_gpus = set()
    for p in Path('/proc').glob('[0-9]*'):
        args = cmdline(int(p.name))
        if args.startswith('/') and '/gpu/gpu_keepalive.py --gpu-index ' in args:
            keepalive_gpus.add(int(args.split('--gpu-index ')[1].split()[0]))
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True)
    for row in processes.splitlines():
        uuid, pid, memory = row.split(',')
        args = cmdline(int(pid))
        # NVML exposes host PIDs while /proc is in the container PID namespace.
        # The known keepalive occupies <1 GiB; a larger/identified foreign task blocks startup.
        known_namespace_keepalive = not args and uuids.get(uuid.strip()) in keepalive_gpus and float(memory) < 1024
        if uuid.strip() in uuids and 'gpu_keepalive.py' not in args and not known_namespace_keepalive:
            raise RuntimeError('Selected GPU has another workload; refusing to disturb it')
    state = {'started_at': time.time(), 'servers': [], 'keepalives': []}
    STATE.write_text(json.dumps(state))
    try:
        for p in Path('/proc').glob('[0-9]*'):
            args = cmdline(int(p.name))
            if '/gpu/gpu_keepalive.py --gpu-index ' in args:
                gpu = int(args.split('--gpu-index ')[1].split()[0])
                if gpu in C['gpus']:
                    state['keepalives'].append({'pid': int(p.name), 'args': args})
                    STATE.write_text(json.dumps(state))
                    os.kill(int(p.name), signal.SIGSTOP)
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
