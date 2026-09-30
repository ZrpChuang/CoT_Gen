#!/usr/bin/env python3
"""Generate all four full benchmarks, overlap judging, audit, then release GPUs."""
import concurrent.futures as cf
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runtime'
RUN.mkdir(exist_ok=True)
BENCHES = ['ifeval', 'multi_challenge', 'eq_bench3', 'arena_hard_v2']
C = json.loads((ROOT / 'config.json').read_text())
lock_dir = Path('/home/tiger/.cache/cot-baseline/locks')
lock_dir.mkdir(parents=True, exist_ok=True)
lock = (lock_dir / (hashlib.sha256(str(ROOT).encode()).hexdigest() + '-suite.lock')).open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def event(stage, **kw):
    with (RUN / 'execution_events.jsonl').open('a') as f:
        f.write(json.dumps({'time': time.time(), 'stage': stage, **kw}) + '\n')
    print(stage, kw, flush=True)


def stage(bench, action):
    event(action + '_start', benchmark=bench)
    with (RUN / f'{bench}_{action}.log').open('a') as log:
        for attempt in range(3):
            r = subprocess.run([sys.executable, 'evaluate.py', '--bench', bench, '--stage', action], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode == 0:
                event(action + '_complete', benchmark=bench)
                return
            event(action + '_retry', benchmark=bench, attempt=attempt + 1)
            time.sleep(10)
    raise RuntimeError(f'Unresolved {bench} {action} errors')


event('waiting_for_download')
for _ in range(360):
    if (RUN / 'download_complete.json').exists():
        break
    pid = int((RUN / 'download.pid').read_text())
    if not Path(f'/proc/{pid}/cmdline').exists() or not Path(f'/proc/{pid}/cmdline').read_bytes():
        raise RuntimeError('Download stopped before verification completed')
    time.sleep(10)
else:
    raise RuntimeError('Download did not complete within one hour')
event('suite_start')
pool = cf.ThreadPoolExecutor(max_workers=2)
scores = []
try:
    if not (RUN / 'service_state.json').exists():
        subprocess.run([sys.executable, 'serve.py', 'start'], cwd=ROOT, check=True)
    for attempt in range(150):
        ready = True
        for base in C['target']['base_urls']:
            try:
                with urllib.request.urlopen(base + '/models', timeout=3) as r:
                    models = json.load(r)
                assert C['target']['model'] in [m['id'] for m in models['data']]
            except Exception:
                ready = False
        if ready:
            break
        state = json.loads((RUN / 'service_state.json').read_text())
        for server in state['servers']:
            p = Path('/proc') / str(server['pid']) / 'cmdline'
            if not p.exists() or not p.read_bytes():
                raise RuntimeError('Model server exited; inspect server log')
        time.sleep(10)
    else:
        raise RuntimeError('Model servers did not become ready')
    import evaluate as E
    # A trivial non-benchmark request confirms the response schema and reasoning parser.
    probe = E.request(C['target'], [{'role': 'user', 'content': 'Compute 19 + 23. Answer with the number only.'}], session_id='preflight-' + C['target']['model'])
    assert probe['returned_model'] == C['target']['model'] and '42' in probe['answer']
    E.write_json(RUN / 'preflight.json', probe)
    event('servers_ready')
    for bench in BENCHES:
        stage(bench, 'generate')
        if bench == 'ifeval':
            stage(bench, 'score')
        else:
            scores.append(pool.submit(stage, bench, 'score'))
    event('generation_complete')
finally:
    subprocess.run([sys.executable, 'serve.py', 'stop'], cwd=ROOT, check=True)
for f in scores:
    f.result()
pool.shutdown(wait=True)

# Remove failed-format diagnostics only after valid replacements were saved.
import evaluate as E
for bench in BENCHES:
    folder = ROOT / bench / 'invalid_judgments'
    if folder.exists():
        valid = E.load_lines(ROOT / bench / 'judgments_0.jsonl')
        for p in folder.glob('*.json'):
            if json.loads(p.read_text())['id'] in valid:
                p.unlink()
        if not list(folder.iterdir()):
            folder.rmdir()
with (RUN / 'verification.log').open('w') as log:
    subprocess.run([sys.executable, 'verify_results.py'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
event('suite_complete')
