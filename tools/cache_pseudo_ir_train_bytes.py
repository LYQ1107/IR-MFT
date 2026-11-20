#!/usr/bin/env python3
"""Bounded, byte-verified tmpfs cache of existing generated Train PNG files.

This optional I/O optimization does not transform images or create a durable
dataset copy. The original NAS files remain the source of truth.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot determine available host memory')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--cache-root', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-gib', type=float, default=10)
    parser.add_argument('--reserve-memory-gib', type=float, default=40)
    args = parser.parse_args()
    cache = args.cache_root.resolve()
    if not cache.is_relative_to(Path('/dev/shm')) or cache == Path('/dev/shm'):
        raise ValueError('Cache must be a task-specific subdirectory of /dev/shm')
    if cache.exists():
        raise FileExistsError(f'Refusing to overwrite an existing cache: {cache}')
    protocol = json.loads(args.protocol.read_text())
    # protocol_v2 stores each split's sequence list directly.
    sequences = protocol['train']
    if isinstance(sequences, dict):
        sequences = sequences['sequences']
    if sequences and isinstance(sequences[0], dict):
        sequences = [seq['sequence'] for seq in sequences]
    allowed = set(sequences)
    images = json.loads(args.annotations.read_text())['images']
    relative = []
    for row in images:
        visible = Path(row['vi_file_name'])
        sequence = visible.parts[0]
        if sequence in allowed:
            relative.append(Path('train') / sequence / 'infrared_diffv2ir' / (visible.stem + '.png'))
    if len(relative) != 27251 or len(set(relative)) != len(relative):
        raise ValueError('Unexpected or duplicated Train frame coverage')
    cache.mkdir(parents=True)
    lock = threading.Lock()
    total = 0
    started = time.perf_counter()
    limit = int(args.max_gib * 2**30)
    reserve = int(args.reserve_memory_gib * 2**30)

    def copy_one(relative_path):
        nonlocal total
        data = (args.source_root / relative_path).read_bytes()
        with lock:
            if total + len(data) > limit:
                raise RuntimeError('Temporary cache byte budget exceeded')
            if available_memory() < reserve + len(data):
                raise RuntimeError('Host memory reserve would be violated')
            if shutil.disk_usage(cache).free < len(data) + 2**30:
                raise RuntimeError('Shared-memory filesystem reserve would be violated')
            total += len(data)
        target = cache / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(data)
        source_sha = hashlib.sha256(data).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != source_sha:
            raise RuntimeError(f'Byte verification failed: {relative_path}')
        return dict(path=str(relative_path), bytes=len(data), sha256=source_sha)

    records = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for record in executor.map(copy_one, relative):
                records.append(record)
                if len(records) % 500 == 0:
                    print(json.dumps(dict(frames=len(records), expected=len(relative),
                        allocated_gib=total / 2**30, seconds=time.perf_counter()-started)), flush=True)
    except Exception:
        print(f'Cache preparation failed; original dataset is untouched: {cache}', flush=True)
        raise
    manifest = dict(complete=True, split='train', frames=len(records), bytes=total,
                    source_root=str(args.source_root), cache_root=str(cache),
                    storage='volatile_tmpfs_no_local_disk_copy', byte_identical=True,
                    max_gib=args.max_gib, reserve_memory_gib=args.reserve_memory_gib,
                    seconds=time.perf_counter()-started, files=records)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    os.replace(temporary, args.manifest)
    print('CACHE_COMPLETE', json.dumps({k:v for k,v in manifest.items() if k!='files'}), flush=True)


if __name__ == '__main__':
    main()
