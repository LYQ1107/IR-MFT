"""Record manually inspected semantic fields with exact Git blob proofs.

This verifies source provenance, not semantic correctness or model reproduction.
It never imports/runs reference code, downloads weights, or modifies references.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from cfp.provenance import sha256, write_json


FIELDS = (
    'paper_year_scenario_contribution', 'code_and_pretrained_license',
    'entrypoints_and_training_inference', 'core_model_files_and_shapes',
    'loss_and_positive_negative_construction', 'pretrained_sources',
    'future_information_and_online_path', 'GPU_CUDA_dependencies',
    'IR_MFT_compatibility', 'mechanisms_to_port_and_exclude', 'novelty_overlap',
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--registry', type=Path, default=Path('docs/research/PHASE_VII_REPOSITORY_REGISTRY.json'))
    ap.add_argument('--document', type=Path, default=Path('docs/research/PHASE_VII_30_PAPERS_CODE_AUDIT.md'))
    ap.add_argument('--proof', type=Path, required=True)
    args = ap.parse_args()
    audit = json.loads(args.audit.read_text())
    registry = json.loads(args.registry.read_text())
    entry = registry['repositories'][audit['name']]
    if entry['commit'] != audit['commit'] or set(audit['semantic_audit']) != set(FIELDS):
        raise ValueError('Exact registered commit and all11 manually verified fields required')
    if any(not isinstance(v, str) or len(v) < 80 for v in audit['semantic_audit'].values()):
        raise ValueError('Concrete source-level field descriptions required')
    if not audit.get('inspection_scope') or len(audit['read_files']) < 5 or args.proof.exists():
        raise ValueError('Actual reading scope, source files and new proof required')
    source = Path(audit['source_root']).resolve()
    marker = f"<!-- exact-source-audit:{audit['name']} -->"
    text = args.document.read_text()
    if marker in text:
        raise ValueError('Audit block already recorded; explicitly review before update')
    if source.is_relative_to(Path.cwd().resolve()):
        raise ValueError('Reference source must stay outsideIR-MFT')
    acquired = None
    if audit.get('source_acquisition_manifest'):
        manifest = json.loads(Path(audit['source_acquisition_manifest']).read_text())
        acquired = manifest['repositories'][audit['name']]
        if acquired['commit'] != audit['commit']:
            raise ValueError('Public source acquisition pin differs')
    proofs = {}
    for name in audit['read_files']:
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError('Inspected file absent or escapes reference')
        if acquired is not None:
            expected_blob = acquired['files'][name]['git_blob_sha']
            provenance = 'PINNED_PUBLIC_GIT_TREE_ACQUISITION_MANIFEST'
        else:
            expected = subprocess.check_output(['git', '-C', str(source), 'ls-tree', audit['commit'], '--', name], text=True).split()
            if len(expected) < 3 or expected[0] == '120000' or expected[1] != 'blob':
                raise ValueError('Exact non-symlink tracked Git blob required')
            expected_blob = expected[2]
            provenance = 'EXACT_LOCAL_GIT_TREE'
        data = path.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != expected_blob:
            raise ValueError('Working source differs from exact commit: ' + name)
        proofs[name] = dict(sha256=sha256(path), git_blob_sha1=actual, bytes=len(data), provenance=provenance,
            verified_against_commit=audit['commit'], inspection_scope=audit['inspection_scope'])
    proof = dict(status='MANUALLY_INSPECTED_FIELDS_EXACT_SOURCE_VERIFIED',
        audit=audit, audit_spec_sha256=sha256(args.audit), sources=proofs,
        verifier_sha256=sha256(__file__), scope='Hash verification does not itself read or validate algorithm semantics. Manual source findings/exclusions are recorded separately; official training/weights are not reproduced.')
    write_json(args.proof, proof)
    entry.update(audit_status='SOURCE_AUDIT_COMPLETE', required_unverified_fields=[],
        semantic_audit=audit['semantic_audit'], source_read_proofs=proofs, semantic_audit_proof=str(args.proof))
    write_json(args.registry, registry)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('| [') and '](' + entry['github'] + ')' in line:
            parts = line.split('|')
            parts[-2] = ' SOURCE AUDIT COMPLETE '
            lines[i] = '|'.join(parts)
    text = '\n'.join(lines) + '\n'
    text += f"\n{marker}\n\n### {audit['name']}: {audit['commit']}\n\n{audit['inspection_scope']}\n\n"
    for field in FIELDS:
        text += f"- **{field}**: {audit['semantic_audit'][field]}\n\n"
    complete = sum(r['audit_status'] == 'SOURCE_AUDIT_COMPLETE' for r in registry['repositories'].values())
    text = re.sub(r'(?:Five|Four|Three) rows now have completed source-level field audits[^\n]*',
        f'Source-level field audits complete: {complete}/30. Remaining: {30 - complete}. Weight/access restrictions are explicit; this is not official pretrained reproduction.', text)
    text = re.sub(r'Source-level field audits complete: \d+/30\. Remaining: \d+\.',
        f'Source-level field audits complete: {complete}/30. Remaining: {30 - complete}.', text)
    args.document.write_text(text.rstrip() + '\n')
    print(json.dumps({'name': audit['name'], 'status': entry['audit_status'], 'verified_files': len(proofs), 'complete': complete}))


if __name__ == '__main__':
    main()
