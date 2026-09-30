#!/usr/bin/env python3
"""Overlap judging with generation; publish metrics only after all 750 answers."""
import concurrent.futures as cf
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import evaluate as E

out = E.ROOT / 'arena_hard_v2'
lock_path = Path('/home/tiger/.cache/cot-baseline/locks') / (hashlib.sha256((str(out) + ':progressive').encode()).hexdigest() + '.lock')
lock = lock_path.open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
qs = E.questions('arena_hard_v2')
protocol = {'dataset_hash': E.digest(qs), 'target': E.CONFIG['target'], 'judges': E.CONFIG['judges'],
            'policy': 'Score every question once a complete answer is saved; retain all valid judgments; publish only at full denominator.'}
manifest = out / 'progressive_scoring_config.json'
if manifest.exists():
    assert json.loads(manifest.read_text()) == protocol
else:
    E.write_json(manifest, protocol)

answer_lock = threading.Lock()
answers = E.load_lines(out / 'responses.jsonl')
deadline = time.monotonic() + 24 * 3600

def get_answer(q):
    while time.monotonic() < deadline:
        with answer_lock:
            if q['id'] in answers:
                a = answers[q['id']]
                assert a['input_hash'] == E.digest(q)
                return a
            try:
                latest = E.load_lines(out / 'responses.jsonl')
            except json.JSONDecodeError:
                latest = {}  # A generation worker may still be appending its last line.
            for key, value in latest.items():
                if key in answers:
                    assert answers[key] == value, 'A saved answer changed'
                else:
                    answers[key] = value
        time.sleep(10)
    raise RuntimeError('Generation did not supply the missing answer within 24 hours')

def one_judge(i, spec):
    def score(q):
        answer = get_answer(q)
        row = E.judge(q, answer, 'arena_hard_v2', spec)
        row['candidate_answer_hash'] = E.digest(answer)
        return row
    for attempt in range(3):
        try:
            rows = E.work(qs, score, out / f'judgments_{i}.jsonl', E.CONFIG['judge_workers'])
            for q in qs:
                assert rows[q['id']]['candidate_answer_hash'] == E.digest(get_answer(q))
            return
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(30)

with cf.ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(one_judge, i, spec) for i, spec in enumerate(E.CONFIG['judges'])]
    for future in cf.as_completed(futures):
        future.result()

assert json.loads((E.ROOT / 'config.json').read_text())['judges'] == E.CONFIG['judges']
subprocess.run([sys.executable, str(E.ROOT / 'evaluate.py'), '--bench', 'arena_hard_v2', '--stage', 'score'], cwd=E.ROOT, check=True)
