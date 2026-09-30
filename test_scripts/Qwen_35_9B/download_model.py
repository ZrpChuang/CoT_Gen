#!/usr/bin/env python3
"""Download the official ModelScope snapshot and check every file's SHA256."""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / 'deployment.json').read_text())
DEST = Path(C['model_dir'])
DEST.mkdir(parents=True, exist_ok=True)
manifest = ROOT / 'runtime' / 'download_manifest.json'
manifest.parent.mkdir(exist_ok=True)
if manifest.exists():
    spec = json.loads(manifest.read_text())
else:
    url = 'https://modelscope.cn/api/v1/models/' + C['repo_id'] + '/repo/files?Revision=master&Recursive=true'
    with urllib.request.urlopen(url, timeout=60) as r:
        listing = json.load(r)
    assert listing['Code'] == 200
    spec = {'repo_id': C['repo_id'], 'source': url, 'files': [f for f in listing['Data']['Files'] if f['Type'] == 'blob']}
    manifest.write_text(json.dumps(spec, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(f):
    path = DEST / f['Path']
    assert path.resolve().is_relative_to(DEST.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == f['Size'] and sha(path) == f['Sha256']:
        print('Verified existing', f['Path'], flush=True)
        return
    url = 'https://modelscope.cn/models/' + C['repo_id'] + '/resolve/' + f['Revision'] + '/' + urllib.parse.quote(f['Path'])
    partial = path.with_name(path.name + '.part')
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=180) as response:
                append = offset > 0 and response.status == 206
                with partial.open('ab' if append else 'wb') as out:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b''):
                        out.write(block)
            assert partial.stat().st_size == f['Size'], 'File size mismatch'
            if sha(partial) != f['Sha256']:
                partial.unlink()
                raise RuntimeError('SHA256 mismatch')
            partial.replace(path)
            print('Downloaded and verified', f['Path'], f['Size'], flush=True)
            return
        except Exception as e:
            print('Download retry', f['Path'], attempt + 1, type(e).__name__, flush=True)
            if attempt == 3:
                raise
            time.sleep(5)


with cf.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, spec['files']))
(ROOT / 'runtime' / 'download_complete.json').write_text(json.dumps({'repo_id': C['repo_id'], 'time': time.time(), 'files': len(spec['files']), 'bytes': sum(f['Size'] for f in spec['files']), 'sha256_verified': True}, indent=2) + '\n')
print('DOWNLOAD COMPLETE', C['repo_id'], flush=True)
