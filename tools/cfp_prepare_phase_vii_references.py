#!/usr/bin/env python3
"""Acquire source-only reference snapshots; an inventory is not a code audit.

Never runs external project code, pulls existing repositories, downloads model
weights/datasets, or marks semantic audit fields verified by filename guessing.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess

from cfp.provenance import sha256, write_json


SOURCE_SUFFIXES = {'.py', '.sh', '.md', '.txt', '.yaml', '.yml', '.json', '.toml',
                   '.cfg', '.ini', '.cpp', '.cu', '.h', '.hpp', '.cuh'}


def git(path, *arguments, input_text=None):
    return subprocess.check_output(['git', '-C', str(path), *arguments], text=True,
                                   input=input_text, timeout=120).strip()


def source_paths(path):
    result = []
    # Do not ask ls-tree for object sizes: in a partial clone that can fetch all
    # blobs, including omitted weights/media. Sizes are measured locally later.
    for row in git(path, 'ls-tree', '-r', 'HEAD').splitlines():
        header, name = row.split('\t', 1)
        mode, kind, digest = header.split()
        if kind != 'blob' or mode == '120000':
            continue
        file = Path(name)
        if file.suffix.lower() == '.json' and not any('config' in part.lower() for part in file.parts):
            continue
        if file.suffix.lower() in SOURCE_SUFFIXES or file.name.lower().startswith(('license', 'copying')):
            # No archived datasets/features/weights, even if their suffix is JSON.
            if not any(part.lower() in {'weights', 'checkpoints', 'pretrained', '.git', 'node_modules'} for part in file.parts):
                result.append((name, (path/name).stat().st_size if (path/name).is_file() else 0))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fetch-missing', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan['references_root']).resolve()
    if plan['state'] != 'FROZEN' or args.output.exists() or len(plan['repositories']) != 30:
        raise ValueError('Frozen 30-reference plan and new immutable output required')
    if not 1 <= plan['workers'] <= 2:
        raise ValueError('Bounded CPU/network concurrency required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', plan_sha256=sha256(args.plan), source_sha256=sha256(__file__),
                  audit_status='IN_PROGRESS', repositories={}, scope='Source inventory only; semantic source audit must be completed separately.')
    write_json(args.output, report)

    def job(spec):
        name, slug, relative = spec
        path = (root / relative).resolve()
        entry = dict(name=name, github=f'https://github.com/{slug}', local_path=str(path), audit_status='NOT_AUDITED')
        try:
            # Reuse existing exact-remote references that may be symlinked to
            # historical checkouts; new clones must stay inside isolated root.
            if path == root or (not path.exists() and not path.is_relative_to(root)):
                raise ValueError('New reference path escapes isolated root')
            if shutil.disk_usage(root).free < plan['reserve_gib'] * 2**30:
                raise OSError('40GiB reserve')
            if not path.exists():
                if not args.fetch_missing:
                    entry['status'] = 'SOURCE_MISSING'
                    return name, entry
                command = ['git', '-c', 'http.proxy=http://127.0.0.1:7890', '-c', 'http.version=HTTP/1.1',
                           'clone', '--depth', '1', '--filter=blob:none', '--no-checkout', f'https://github.com/{slug}.git', str(path)]
                log = args.output.parent / f'{args.output.stem}_{name}.log'
                with log.open('x') as stream:
                    for attempt in range(3):
                        completed = subprocess.run(command, timeout=180, stdout=stream, stderr=subprocess.STDOUT,
                                                   env=dict(os.environ, GIT_LFS_SKIP_SMUDGE='1'))
                        if completed.returncode == 0:
                            break
                        if path.exists():
                            raise RuntimeError('Clone failed with partial target; preserved for inspection')
                    else:
                        raise RuntimeError('Three source acquisition attempts failed; see preserved log')
                entry['acquisition_command'] = command
                paths = source_paths(path)
                if sum(n for _, n in paths) > plan['maximum_source_bytes_per_repository']:
                    raise ValueError('Source snapshot exceeds declared per-repository budget')
                # Exact sparse paths omit binary weights, images and videos.
                git(path, 'config', 'http.proxy', 'http://127.0.0.1:7890')
                git(path, 'config', 'http.version', 'HTTP/1.1')
                git(path, 'sparse-checkout', 'set', '--no-cone', '--stdin',
                    input_text='\n'.join('/' + name for name, _ in paths) + '\n')
                git(path, 'checkout', 'HEAD')
                entry['acquisition'] = 'NEW_SOURCE_ONLY_SPARSE_CLONE'
            else:
                entry['acquisition'] = 'REUSE_EXISTING_NO_PULL'
            if not (path / '.git').exists():
                raise ValueError('Existing path lacks a Git snapshot; preserve it')
            origin = git(path, 'remote', 'get-url', 'origin').removesuffix('.git').rstrip('/')
            if origin != entry['github']:
                raise ValueError('Existing remote differs from requested repository')
            entry.update(commit=git(path, 'rev-parse', 'HEAD'), origin=origin)
            paths = source_paths(path)
            if not all((path/n).is_file() for n, _ in paths) and relative.startswith('phase7/'):
                # Repair only our incomplete sparse source clones, never a
                # historical checkout. No existing source file is overwritten.
                expected = {line.split('\t',1)[1]:line.split('\t',1)[0].split()[2]
                            for line in git(path,'ls-tree','-r','HEAD').splitlines()}
                for n, _ in paths:
                    if (path/n).is_file() and git(path,'hash-object',str(path/n))!=expected[n]:
                        raise ValueError('Materialized source clone has modified content; preserve it')
                git(path, 'config', 'http.proxy', 'http://127.0.0.1:7890')
                git(path, 'config', 'http.sslVersion', 'tlsv1.2')
                git(path, 'sparse-checkout', 'set', '--no-cone', '--stdin',
                    input_text='\n'.join('/' + n for n, _ in paths)+'\n')
                git(path, 'checkout', 'HEAD')
                paths = source_paths(path)
                entry['acquisition'] = 'REPAIRED_OWN_PARTIAL_SOURCE_CLONE'
            entry['dirty'] = bool(git(path, 'status', '--porcelain'))
            entry['source_file_count'] = len(paths)
            entry['tracked_source_bytes'] = sum(n for _, n in paths)
            entry['license_files'] = [dict(path=n, sha256=sha256(path/n)) for n, _ in paths
                                      if Path(n).name.lower().startswith(('license', 'copying')) and (path/n).is_file()]
            entry['source_inventory_sha256'] = __import__('hashlib').sha256(json.dumps(paths).encode()).hexdigest()
            entry['status'] = 'SOURCE_READY' if all((path/n).is_file() for n, _ in paths) else 'SOURCE_CHECKOUT_INCOMPLETE'
            entry['license_status'] = 'REQUIRES_READING' if entry['license_files'] else 'NO_TOP_LEVEL_LICENSE_VERIFIED'
        except Exception as error:
            entry.update(status='ACQUISITION_FAILED', error=repr(error))
        return name, entry

    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        for future in as_completed([pool.submit(job, r) for r in plan['repositories']]):
            name, entry = future.result()
            report['repositories'][name] = entry
            write_json(args.output, report)
            print(name, entry['status'], entry.get('commit'), entry.get('error', ''), flush=True)
    report.update(status='INVENTORY_COMPLETE', storage_free_gib=shutil.disk_usage(root).free/2**30)
    write_json(args.output, report)


if __name__ == '__main__':
    main()
