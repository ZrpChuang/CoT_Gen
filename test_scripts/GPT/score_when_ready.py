#!/usr/bin/env python3
"""Score each full response set as soon as generation releases its lock."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
for bench, expected in [('multi_challenge', 273), ('eq_bench3', 45), ('arena_hard_v2', 750)]:
    out = root / bench
    deadline = time.time() + 24 * 3600
    while time.time() < deadline:
        try:
            rows = [json.loads(line) for line in (out / 'responses.jsonl').open() if line.strip()]
            ready = len({row['id'] for row in rows}) == expected
            lock_dir = Path('/home/tiger/.cache/cot-baseline/locks')
            lock_dir.mkdir(parents=True, exist_ok=True)
            with (lock_dir / (hashlib.sha256(str(out).encode()).hexdigest() + '.lock')).open('a') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            if ready:
                break
        except (OSError, ValueError):
            pass
        time.sleep(15)
    else:
        raise RuntimeError('Generation did not complete: ' + bench)
    for attempt in range(3):
        result = subprocess.run([sys.executable, str(root / 'evaluate.py'), '--bench', bench, '--stage', 'score'], cwd=root)
        if result.returncode == 0:
            break
        time.sleep(30)
    else:
        raise RuntimeError('Unresolved scoring errors: ' + bench)
print('All model benchmark scores complete.', flush=True)
