#!/usr/bin/env python3
"""Apply the original address audit to the registered content-guard workers."""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument('--output', required=True)
    args, _ = parser.parse_known_args()
    source = ROOT / 'tools/audit_structural_full_train_progress.py'
    original = ast.parse(source.read_text()); changed = ast.parse(source.read_text())
    matches = [node for node in ast.walk(changed) if isinstance(node, ast.Constant)
               and node.value == b'tools/train_structural_full_train.py']
    assert len(matches) == 1
    matches[0].value = b'tools/train_structural_content_verified.py'
    original_literals = [node for node in ast.walk(original) if isinstance(node, ast.Constant)
                         and node.value == b'tools/train_structural_full_train.py']
    assert len(original_literals) == 1
    original_literals[0].value = b'tools/train_structural_content_verified.py'
    assert ast.dump(original) == ast.dump(changed)
    namespace = dict(__name__='private_content_progress_audit', __file__=__file__)
    exec(compile(ast.fix_missing_locations(changed), str(source), 'exec'), namespace)
    namespace['main']()
    path = Path(args.output); result = json.loads(path.read_text())
    result.update(original_auditor=str(source.relative_to(ROOT)), original_auditor_sha256=sha256(source),
        private_binding_changes={'expected_live_worker_script_literal': 1},
        all_other_original_audit_AST_unchanged=True,
        input_content_manifest_scope='This proves actual delivered addresses and the pinned byte-verifying '
            'reader code; the final complete image-content manifest is emitted only when each fit ends.')
    dump(path, result)


if __name__ == '__main__':
    main()
