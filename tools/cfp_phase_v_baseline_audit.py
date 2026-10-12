#!/usr/bin/env python3
"""Hash-audit preserved inputs and independently reevaluate three frozen systems."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from cfp.provenance import sha256, snapshot_sources, write_json


FIELDS = ['HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSW', 'CLR_FP', 'CLR_FN', 'Frag', 'MT', 'ML', 'AssRe', 'AssPr', 'DetRe', 'DetPr']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--test-log', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Audit is immutable')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    protocol_path = Path('configs/cfp/protocol_v2.json')
    protocol = json.loads(protocol_path.read_text())
    if shutil.disk_usage(root).free < 40 * 2**30:
        raise OSError('40GiB safety reserve required')
    record = dict(status='RUNNING', source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  protocol_sha256=sha256(protocol_path), source_snapshots=snapshot_sources(args.output, [__file__, protocol_path]),
                  input_hashes={}, runtime={}, baselines={}, storage_free_gib=shutil.disk_usage(root).free / 2**30,
                  minimum_free_gib=40, gpu_limit=4, historical_test_evaluated=False)
    write_json(args.output / 'manifest.json', record)

    def verify(path, expected=None):
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Input changed: {path}')
        record['input_hashes'][str(path)] = digest
        return digest

    detector = Path('/data3/liuyeqiang/IR-MFT_assets/checkpoints/mft26ir_rgbt_v3_best_single_gpu.pth.tar')
    verify(detector, 'c448238beaa8540860d17f6a022d99b29b1407649823398a8553fa0890b6d786')
    for role in ('dev', 'train'):
        bank = root / 'phase0' / f'causal_v3_{role}_detections'
        manifest = json.loads((bank / 'manifest.json').read_text())
        if manifest['status'] != 'COMPLETE' or manifest['protocol_sha256'] != record['protocol_sha256']:
            raise ValueError('Incomplete detector bank or protocol drift')
        verify(bank / 'manifest.json')
        for seq, entry in manifest['caches'].items():
            verify(bank / (seq + '.npz'), entry['sha256'])
    for seed in range(3):
        for role, phase in [('dev', 'phase1'), ('train', 'phase2')]:
            bank = root / phase / f'causal_{role}_detection_features_postneck_seed{seed}'
            manifest = json.loads((bank / 'manifest.json').read_text())
            if manifest['status'] != 'COMPLETE' or manifest['protocol_sha256'] != record['protocol_sha256']:
                raise ValueError('Incomplete feature bank or split drift')
            verify(bank / 'manifest.json')
            for seq in protocol[role]:
                provenance = json.loads((bank / (seq + '.provenance.json')).read_text())
                verify(bank / (seq + '.npz'), provenance['features_sha256'])
            for modality in ('rgb', 'ir'):
                folder = f'reid_rgb_seed{seed}' if modality == 'rgb' else f'reid_ir_causal_seed{seed}'
                verify(root / 'phase1' / folder / 'epoch12.pt', manifest[f'{modality}_encoder']['checkpoint_sha256'])
    data = Path('/data3/liuyeqiang/datasets/mft26-ir-causal-v1')
    for seq in protocol['train'] + protocol['dev']:
        verify(data / 'train' / seq / 'gt/gt.txt', protocol['gt_hashes'][seq])
        verify(root / 'protocol_v2/gt/train' / seq / 'gt/gt.txt', protocol['derived_gt_sha256']['train/' + seq])
        if not all((data / 'train' / seq / branch / '000001.jpg').exists() for branch in ('visible', 'infrared')):
            raise FileNotFoundError('Missing paired dataset')
    for name in ('cfp-fishmot-cu117', 'cfp-fishmot'):
        python = f'/data3/liuyeqiang/.venvs/{name}/bin/python'
        command = [python, '-c', "import sys,json,torch,cv2; print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cv2=cv2.__version__)))"]
        record['runtime'][name] = json.loads(subprocess.check_output(command, text=True))
    for name, expected in [('cfp-fishmot-cu117', ('2.0.0+cu117', 8500)), ('cfp-fishmot', ('2.0.0+cu118', 8700))]:
        if (record['runtime'][name]['torch'], record['runtime'][name]['cudnn']) != expected:
            raise ValueError('Verified runtime changed')
    record['resource_snapshot'] = {key: subprocess.check_output(cmd, text=True) for key, cmd in [
        ('gpus', ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu', '--format=csv']),
        ('memory', ['free', '-h']), ('cpu_count', ['nproc']), ('storage', ['df', '-h', str(root)])]}
    record['test_log_sha256'] = verify(args.test_log)
    if '100 passed' not in args.test_log.read_text():
        raise ValueError('Baseline tests did not pass')
    families = {'ByteTrack': [('phase0', 'causal_dev_bytetrack_off')]}
    for label, phase, pattern in [('BoT-SORT', 'phase1', 'causal_dev_postneck_bot_sort_seed{s}_c0'),
                                  ('RGB-GRU', 'phase3', 'plim_v3_dev_gru_association_anchor_seed{s}_w0.3')]:
        families[label] = [(phase, pattern.format(s=s)) for s in range(3)]
    for label, runs in families.items():
        entries = []
        for phase, name in runs:
            run = root / phase / name
            old = json.loads((root / phase / (name + '_evaluation/metrics.json')).read_text())
            verify(run / 'manifest.json')
            for seq in protocol['dev']:
                verify(run / (seq + '.txt'), old['artifacts'][seq]['prediction_sha256'])
            output = args.output / (name + '_reevaluation')
            command = [sys.executable, 'tools/cfp_evaluate.py', '--data-root', str(data), '--derived-gt-root', str(root / 'protocol_v2/gt'),
                       '--protocol', str(protocol_path), '--partition', 'dev', '--predictions', str(run), '--output', str(output),
                       '--trackeval-root', '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
            with (args.output / (name + '_reevaluation.log')).open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            fresh = json.loads((output / 'metrics.json').read_text())
            if fresh['metrics'] != old['metrics'] or fresh['metric_sha256'] != old['metric_sha256']:
                raise ValueError(f'Official metric drift: {name}')
            entries.append(dict(run=name, command=command, reevaluation_sha256=sha256(output / 'metrics.json'),
                                per_sequence={seq: {key: values[key] for key in FIELDS} for seq, values in fresh['metrics'].items()}))
            print('Exact official reevaluation:', name, flush=True)
        record['baselines'][label] = dict(runs=entries, combined_mean={key: statistics.mean(x['per_sequence']['COMBINED_SEQ'][key] for x in entries) for key in FIELDS},
                                         combined_sample_std={key: statistics.stdev(x['per_sequence']['COMBINED_SEQ'][key] for x in entries) if len(entries) > 1 else 0 for key in FIELDS})
    record.update(status='BASELINE_REPRODUCED', registration_count=len(list(Path('docs/research/experiment_registry').glob('*.json'))),
                  scope='Reevaluation of preserved outputs, no new detector inference/SGD/test tuning; derived GT frozen. Detector trained on all8, identity modules heldout6/2. Short canonical terminalbatch2 vs online batch8 retained.')
    write_json(args.output / 'manifest.json', record)
    write_json('docs/research/PHASE_V_BASELINE_AUDIT.json', record)
    lines = ['# Phase V baseline audit', '', '**BASELINE_REPRODUCED**. Existing detector/feature/weight/GT hashes verified; seven preserved runs independently reevaluated with pinned official TrackEval. All metric arrays and metric-source hashes exactly match prior records. No SGD, detector/cache changes or historical test evaluation.', '',
             '| System | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |', '|---|---:|---:|---:|---:|---:|---:|']
    for label, entry in record['baselines'].items():
        lines.append('| ' + label + ' | ' + ' | '.join(f'{entry["combined_mean"][key]:.3f}' for key in FIELDS[:6]) + ' |')
    lines += ['', f'Available storage at start: {record["storage_free_gib"]:.1f}GiB; minimum reserve40GiB; maximum4 physical GPUs.100 original tests pass. Both pinned detector cu117/cuDNN8.5 and encoder cu118/cuDNN8.7 stacks verified. Detailed system snapshot, every video/seed metric, actual commands and input hashes: PHASE_V_BASELINE_AUDIT.json.', '',
              'Baseline worktree began clean at4f1b226, live remote confirmed identical; new branch research/identity-repair-jepa-v1 created without overwriting history. Current prototype source commit: ' + record['source_commit'], '',
              'The two dev videos isolate new identity modules only; historical V3 trained on all8. Biological identity overlap unverified; dense MSK excluded dev. Previous online numeric boundary remains: canonical4834frame detector bank ends withbatch2, while online uses8current copies; last2frames differ in coordinates/confidence, principal metrics/identity decisions equal. See RESEARCH_OUTCOME_20261012.json. Full per-sequence metrics remain in the machine audit; three-seed variation is not a cross-scene confidence interval.']
    Path('docs/research/PHASE_V_BASELINE_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print('Audit complete; input files verified:', len(record['input_hashes']))


if __name__ == '__main__':
    main()
