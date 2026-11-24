#!/usr/bin/env python3
"""Run the original Train auditor with one private expected-worker literal binding."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original = ROOT/'tools/audit_structural_full_train_progress.py'
    expected = '8f1f4ddef10ab7f94d2c8329e9b7f8e3d3f703f0800671c2469e4d18b411e961'
    assert sha(original) == expected
    before = ast.parse(original.read_text())
    original_dump = ast.dump(before, include_attributes=False)
    changed = [0]
    class Bind(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == b'tools/train_structural_full_train.py':
                changed[0] += 1
                return ast.copy_location(ast.Constant(value=b'tools/train_structural_content_verified.py'), node)
            return node
    after = Bind().visit(before)
    assert changed == [1]
    modified_dump = ast.dump(after, include_attributes=False)
    assert modified_dump.replace(repr(b'tools/train_structural_content_verified.py'),
                                 repr(b'tools/train_structural_full_train.py')) == original_dump
    namespace = {'__name__': '__private_expected_worker_binding__', '__file__': str(original)}
    sys.path.insert(0, str(ROOT))
    exec(compile(ast.fix_missing_locations(after), str(original), 'exec'), namespace)
    namespace['main']()
    assert '--output' in sys.argv
    output = Path(sys.argv[sys.argv.index('--output')+1])
    result = json.loads(output.read_text())
    result.update(original_auditor=str(original.relative_to(ROOT)), original_auditor_sha256=expected,
        private_binding_changes={'expected_live_worker_script_literal': 1},
        all_other_original_audit_AST_unchanged=True, private_executable_AST_sha256=hashlib.sha256(modified_dump.encode()).hexdigest(),
        binding_wrapper=str(Path(__file__).relative_to(ROOT)), binding_wrapper_sha256=sha(__file__),
        input_content_manifest_scope='Actual delivered addresses and byte-verifying reader source only; final image-content manifest emitted at fit completion.')
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
