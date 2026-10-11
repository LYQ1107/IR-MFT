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


def snapshot_sources(directory, paths):
    """Retain the exact small source/config bytes behind a run's hashes."""
    root = Path.cwd().resolve()
    result = {}
    for source in paths:
        source = Path(source).resolve()
        data = source.read_bytes()
        if source.suffix not in ('.py', '.json', '.md') or len(data) > 4 * 1024 * 1024:
            raise ValueError('Source snapshots accept small code/config text, never model/data arrays')
        digest = hashlib.sha256(data).hexdigest()
        try:
            name = source.relative_to(root)
        except ValueError:
            name = Path('external') / (digest[:12] + '_' + source.name)
        destination = Path(directory) / 'source_snapshots' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != data:
            raise FileExistsError('Existing source snapshot differs')
        destination.write_bytes(data)
        result[str(name)] = digest
    return result


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
