"""Fetch bounded, manually selected public source blobs, never weights/data."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

from cfp.provenance import sha256, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.output.exists() or not 1 <= len(plan['files']) <= 48:
        raise ValueError('New, bounded selected-source acquisition required')
    root = Path(plan['source_root']).resolve()
    if not root.is_relative_to(Path('/data3/liuyeqiang/CFP-FishMOT_references/phase7')):
        raise ValueError('Only new PhaseVII reference snapshots may be materialized')
    git_root = Path(plan.get('git_object_root', root))
    if shutil.disk_usage(root).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    report = dict(status='RUNNING', commit=plan['commit'], plan_sha256=sha256(args.plan),
        tool_sha256=sha256(__file__), files={}, scope='Public exact-commit selected text blobs only. No external execution, model/data acquisition or semantic audit claim.')
    write_json(args.output, report)

    def fetch(name):
        try:
            path = (root / name).resolve()
            if not path.is_relative_to(root) or (path.suffix not in ('.py', '.md', '.txt', '.yaml', '.yml', '.json')
                    and not path.name.lower().startswith(('license', 'copying'))):
                raise ValueError('Only safe selected source text')
            if any(p.lower() in ('weights', 'checkpoints', 'images', 'assets') for p in Path(name).parts):
                raise ValueError('Model/data/media paths excluded')
            tree = subprocess.check_output(['git', '-C', str(git_root), 'ls-tree', plan['commit'], '--', name], text=True).split()
            if len(tree) < 3 or tree[1] != 'blob' or tree[0] == '120000':
                raise ValueError('Exact non-symlink Git blob absent')
            expected = tree[2]
            errors = []
            if path.exists():
                data = path.read_bytes()
                method = 'VERIFIED_EXISTING'
            else:
                opener = build_opener(ProxyHandler({'https': 'http://127.0.0.1:7890'}))
                url = 'https://api.github.com/repos/' + plan['github_slug'] + '/contents/' + quote(name, safe='/') + '?ref=' + plan['commit']
                for attempt in range(2):
                    try:
                        with opener.open(Request(url, headers={'User-Agent': 'phase7-exact-source-audit'}), timeout=8) as response:
                            payload = response.read(1500001)
                        if len(payload) > 1500000:
                            raise ValueError('Public source payload budget exceeded')
                        info = json.loads(payload)
                        if info.get('encoding') != 'base64' or info['sha'] != expected:
                            raise ValueError('Pinned publisher Git SHA differs')
                        data = base64.b64decode(info['content'], validate=False)
                        if len(data) > 1024 * 1024:
                            raise ValueError('1MiB source-file budget exceeded')
                        break
                    except Exception as error:
                        errors.append(repr(error))
                else:
                    raise RuntimeError('; '.join(errors))
                method = 'PUBLIC_PINNED_CONTENTS_API'
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != expected:
                raise ValueError('Actual source bytes differ; preserve existing source')
            if not path.exists():
                if shutil.disk_usage(root).free < 40 * 2**30:
                    raise OSError('40GiB reserve')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(data)
            return name, dict(status='EXACT_GIT_BLOB_VERIFIED', path=str(path), git_blob_sha=blob,
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), acquisition=method, transport_errors=errors)
        except Exception as error:
            return name, dict(status='ACQUISITION_FAILED', error=repr(error))

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(fetch, name) for name in plan['files']]):
            name, entry = future.result()
            report['files'][name] = entry
            write_json(args.output, report)
            print(name, entry['status'], flush=True)
    report['status'] = 'SELECTED_SOURCE_COMPLETE' if all(x['status'] == 'EXACT_GIT_BLOB_VERIFIED' for x in report['files'].values()) else 'SELECTED_SOURCE_PARTIAL'
    write_json(args.output, report)


if __name__ == '__main__':
    main()
