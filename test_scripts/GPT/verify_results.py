#!/usr/bin/env python3
"""Audit full denominators and scores without issuing any model requests."""
import json
import math
import statistics
from collections import Counter
import evaluate as E

audit = {}
for bench in E.BENCHES:
    out = E.ROOT / bench
    questions = {q['id']: q for q in E.questions(bench)}
    answers = E.load_lines(out / 'responses.jsonl')
    assert set(questions) == set(answers), f'{bench}: missing or extra answers'
    assert all(a['input_hash'] == E.digest(questions[k]) for k, a in answers.items()), f'{bench}: changed input'
    assert not list(out.glob('*errors.json')), f'{bench}: unresolved errors'
    report = json.loads((out / 'metrics.json').read_text())
    assert report['status'] == 'complete' and report['n'] == len(questions)
    if bench == 'ifeval':
        rows = E.load_lines(out / 'judgments.jsonl')
        assert set(rows) == set(questions)
        for mode in ['strict', 'loose']:
            prompt = statistics.mean(all(r[mode]) for r in rows.values()) * 100
            instruction = statistics.mean(v for r in rows.values() for v in r[mode]) * 100
            assert math.isclose(prompt, report['metrics']['prompt_' + mode]['score'])
            assert math.isclose(instruction, report['metrics']['instruction_' + mode]['score'])
    else:
        assert set(report['judges']) == {j['model'] for j in E.CONFIG['judges']}
        for i, spec in enumerate(E.CONFIG['judges']):
            protocol = json.loads((out / f'judge_{i}_config.json').read_text())
            assert protocol['answer_hash'] == E.digest(answers)
            rows = E.load_lines(out / f'judgments_{i}.jsonl')
            assert set(rows) == set(questions)
            assert all(r.get('candidate_answer_hash', E.digest(answers[k])) == E.digest(answers[k]) for k, r in rows.items())
            assert E.metrics(bench, list(rows.values())) == report['judges'][spec['model']]
            if bench == 'multi_challenge':
                assert all(r['pass_criteria'] == questions[k]['PASS_CRITERIA'] and r['passed'] == (r['verdict'] == questions[k]['PASS_CRITERIA']) for k,r in rows.items())
    if bench == 'eq_bench3' and not E.CONFIG['target'].get('local'):
        for row in answers.values():
            turns = row['turns'] + ([row['debrief']] if row.get('debrief') else [])
            assert len({turn['session_id'] for turn in turns}) == 1
    outputs = [t for row in answers.values() for t in (row['turns'] + ([row['debrief']] if row.get('debrief') else []))] if bench == 'eq_bench3' else list(answers.values())
    token_counts = [o['usage'].get('completion_tokens', o['usage'].get('output_tokens', 0)) for o in outputs]
    diagnostics = {'returned_model_counts': dict(Counter(o['returned_model'] for o in outputs)),
                   'finish_reason_counts': dict(Counter(o['finish_reason'] for o in outputs)),
                   'empty_final_answers': sum(not o['answer'].strip() for o in outputs),
                   'mean_reported_output_tokens': statistics.mean(token_counts),
                   'max_reported_output_tokens': max(token_counts),
                   'reported_output_over_requested_limit': sum(n > E.CONFIG['target']['max_tokens'] for n in token_counts)}
    audit[bench] = {'generation_diagnostics': diagnostics, 'n': len(questions), 'verified': True, 'responses_sha256': E.hashlib.sha256((out / 'responses.jsonl').read_bytes()).hexdigest(), 'metrics_sha256': E.hashlib.sha256((out / 'metrics.json').read_bytes()).hexdigest()}
E.write_json(E.ROOT / 'audit.json', audit)
print(json.dumps(audit, indent=2))
