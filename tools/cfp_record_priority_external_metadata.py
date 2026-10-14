#!/usr/bin/env python3
"""Primary metadata/source audit; never acquire/execute remote payloads."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from cfp.provenance import sha256, snapshot_sources, write_json


def pinned_file(root, commit, relative):
    tree = subprocess.check_output(['git', '-C', str(root), 'ls-tree', commit, '--', relative], timeout=8, text=True).strip()
    path = root/relative
    expected = path.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(expected)).encode()+b'\0'+expected).hexdigest()
    if len(tree.split()) != 4 or tree.split()[2] != blob or tree.split()[3] != relative:
        raise ValueError('Exact primary source pin required')
    return dict(path=str(path), commit=commit, relative=relative, sha256=sha256(path), git_blob_sha1=blob)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise ValueError('New immutable primary metadata audit required')
    reference = Path('/data3/liuyeqiang/CFP-FishMOT_references')
    tide = reference/'TIDE'; sut = reference/'SU-T'; fish = reference/'phase7/FishMAE'
    tide_commit = 'fba27aefd1da11bfc7f946240dd66d85ea02a39c'
    proofs = [pinned_file(tide, tide_commit, p) for p in ['README.md', 'src/track.py', 'src/lib/tracker/matching.py']]
    proofs += [pinned_file(sut, '479be89a27129f6879c62710a83abac6a8b261ee', 'README.md'),
               pinned_file(fish, 'cffcdeaf61c8c3a9768711a4615ff4af3899ca30', 'README.md')]
    sequences = {}
    for node in ast.walk(ast.parse((tide/'src/track.py').read_text())):
        if isinstance(node, ast.If):
            condition = ast.unparse(node.test)
            if 'opt.val_MFT_Edge' in condition or 'opt.val_mft25' in condition:
                for child in node.body:
                    if isinstance(child, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'seqs_str' for t in child.targets):
                        sequences['MFT-Edge' if 'MFT_Edge' in condition else 'MFT25_test'] = ast.literal_eval(child.value).split()
    metadata = json.loads((a.metadata/'manifest.json').read_text())
    for row in metadata['rows']:
        if 'sha256' in row and sha256(row['path']) != row['sha256']:
            raise ValueError('Primary response snapshot drift')
    hf = json.loads((a.metadata/'mft25_hf_metadata.json').read_text())
    if hf['id'] != 'Vranlee/MFT25' or hf['private'] or hf['gated'] != 'auto':
        raise ValueError('Inspected public-metadata/auto-gated dataset contract changed')
    report = dict(status='COMPLETE_PRIMARY_METADATA_AUDIT_WITH_ACCESS_AND_OVERLAP_LIMITS', payloads_acquired=False,
                  metadata_manifest=str(a.metadata/'manifest.json'), metadata_manifest_sha256=sha256(a.metadata/'manifest.json'),
                  acquisition_results=metadata['rows'], exact_source_proofs=proofs, source_declared_evaluation_sequences=sequences,
                  entries=[dict(name='MFT25', primary_sources=['https://vranlee.github.io/SU-T/', 'https://huggingface.co/datasets/Vranlee/MFT25'],
                                hf_repository_sha=hf['sha'], creator_declared_license=hf['cardData']['license'],
                                access=hf['gated'], declared_total_storage_bytes=hf['usedStorage'],
                                file_names=[s['rfilename'] for s in hf['siblings']],
                                overlap='Unresolved. Source test names include BT/MSK/PF/SN families also present in inherited MFT26. Names alone neither establish nor exclude exact frame/physical-fish overlap. Require image hashes/video lineage and inherited detector/reID train lists before external claim.',
                                identity_and_cadence='Actual zip contents, split/identity lifetime, seqinfo FPS and RGB provenance not inspected.',
                                next_access_step='Creator HF gate requires an account agreeing to contact-information sharing. No automatic acceptance/contact sharing or gated file request performed.'),
                           dict(name='MFT-Edge', primary_sources=['https://vranlee.github.io/TIDE/', 'https://arxiv.org/html/2607.26412v1'],
                                code_pin=tide_commit, dataset_access='Creator Baidu link; README competition URL not verified as a functioning fish benchmark.',
                                dataset_license='No independently verified dataset grant; MIT source license does not license images/annotations.',
                                overlap='PF-01/MSK-01 names may relate to inherited PF/MSK families; no content identity established. Exact split, timestamps/visibility IDs and clip lineage pending.'),
                           dict(name='FishMAE laboratory data', primary_sources=['https://github.com/Wynne-cc/FishMAE', 'https://www.sciencedirect.com/science/article/abs/pii/S0144860926000622'],
                                code_pin='cffcdeaf61c8c3a9768711a4615ff4af3899ca30', access='Pinned README supplies user-local data_path, no public dataset/weight download or data-license terms.',
                                identity_and_cadence='Long-term individual retrieval is different from dense native MOT; session timing/physical ID/lifetime/overlap not verified. No foreign code/data acquired or executed.')],
                  paper_source_discrepancy=dict(primary='https://arxiv.org/html/2607.26412v1#S3.SS3',
                      observed='Paper AGCIoU uses normalized center distance, aspect/area alignments, and IoU<=0.7 activation with kd0.1/ks0.05/lambda0.5. Pinned source AGCv1 instead adds0.05/raw center distance; AGCv2 has no return and active ious uses bbox_ious. Prior clipped AGCv1 control is not a paper reproduction.',
                      unresolved='Paper average scale s and epsilon have no fully specified implementation in inspected main text/source. Zero-distance scores can exceed1; do not silently invent score clamping or claim faithful native geometry from prior proxy.'),
                  scope='Three priority primary metadata/source audits only. External image/annotation payload, exact license grant/lineage/dedup/native-cadence validation and heldout generalization remain pending. No external training or zero-overlap claim.',
                  sources=snapshot_sources(a.output.with_suffix('.sources'), [__file__]+[p['path'] for p in proofs]))
    write_json(a.output, report)
    print(json.dumps(dict(status=report['status'], priority_entries=len(report['entries']), source_files=len(proofs))))


if __name__ == '__main__':
    main()
