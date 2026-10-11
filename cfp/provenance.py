"""Content-based provenance and atomic experiment records."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        temp = stream.name
    os.replace(temp, path)


def record_run(directory, record, *, update=False):
    required = {'run_id', 'hypothesis', 'git_commit', 'seed', 'config',
                'detector_sha256', 'reid_sha256', 'split', 'epochs',
                'checkpoint_selection', 'metrics', 'log', 'output', 'conclusion', 'status'}
    missing = required - record.keys()
    if missing:
        raise ValueError(f'Missing experiment fields: {sorted(missing)}')
    run_id = record['run_id']
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Unsafe run ID')
    path = Path(directory) / (run_id + '.json')
    if path.exists() and not update:
        raise FileExistsError(path)
    record = dict(record, config_sha256=config_hash(record['config']))
    write_json(path, record)
    return path
