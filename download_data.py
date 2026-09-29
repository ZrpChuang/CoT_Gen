#!/usr/bin/env python3
"""Restore immutable official dataset files from the tracked manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def valid(path, item):
    return path.is_file() and path.stat().st_size == item['size'] and digest(path) == item['sha256']

def restore(item, root, verify_only):
    dest = (root / item['destination']).resolve()
    if root not in dest.parents:
        raise ValueError('Manifest path escapes output directory')
    if valid(dest, item):
        return 'verified', item['destination']
    if verify_only:
        raise ValueError('Missing or invalid file: ' + item['destination'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    for attempt in range(3):
        try:
            request = Request(item['url'], headers={'User-Agent': 'CoT_Gen-dataset-downloader'})
            with urlopen(request, timeout=90) as response, part.open('wb') as out:
                while chunk := response.read(4 * 1024 * 1024):
                    out.write(chunk)
            if not valid(part, item):
                raise ValueError('Downloaded file failed checksum: ' + item['destination'])
            part.replace(dest)
            return 'downloaded', item['destination']
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO)
    parser.add_argument('--group', choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.root.resolve()
    sources = json.loads((REPO / 'DOWNLOAD_MANIFEST.json').read_text())
    items = []
    for source in sources:
        group = 'train' if source['name'] == 'lmarena_140k' else 'test'
        if args.group in ['all', group]:
            print(source['name'], len(source['files']), 'files', flush=True)
            items.extend(source['files'])
    print(f"Total: {len(items)} files, {sum(i['size'] for i in items) / 1e9:.3f} GB", flush=True)
    if args.dry_run:
        return
    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        tasks = {pool.submit(restore, item, root, args.verify_only): item for item in items}
        for task in as_completed(tasks):
            try:
                status, name = task.result()
                print(status, name, flush=True)
            except Exception as exc:
                # Don't print redirect URLs or authorization parameters on errors.
                name = tasks[task]['destination']
                failures.append(name)
                print('FAILED', name, type(exc).__name__, flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} files failed. Rerun to retry; verified files are reused.')
    print('All selected files verified.', flush=True)

if __name__ == '__main__':
    main()
