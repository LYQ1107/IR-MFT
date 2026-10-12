#!/usr/bin/env python3
"""Reverify inherited inputs and the strong current-measurement baseline.

No fitting or history/test selection. Runs use immutable new output locations.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from cfp.provenance import sha256, snapshot_sources, write_json


FIELDS = ['HOTA', 'AssA', 'DetA', 'LocA', 'AssRe', 'AssPr', 'DetRe',
          'DetPr', 'IDF1', 'IDP', 'IDR', 'MOTA', 'IDSW', 'CLR_FP',
          'CLR_FN', 'Frag', 'MT', 'ML']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Audit output is immutable')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    if shutil.disk_usage(root).free < 40 * 2**30:
        raise OSError('40 GiB reserve')
    args.output.mkdir(parents=True)
    parent = json.loads(Path('docs/research/PHASE_V_BASELINE_AUDIT.json').read_text())
    record = dict(status='RUNNING', git_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        protocol_sha256=sha256('configs/cfp/phase_vi_protocol_v1.json'),
        inherited_audit_sha256=sha256('docs/research/PHASE_V_BASELINE_AUDIT.json'),
        input_hashes={}, runs=[], source_snapshots=snapshot_sources(args.output, [
            __file__, 'configs/cfp/phase_vi_protocol_v1.json',
            'tools/cfp_replay_recovery.py', 'tools/cfp_evaluate.py',
            'cfp/predictive_association.py', 'cfp/lifecycle_crossover.py',
            'cfp/history.py', 'cfp/association.py']),
        hardware=subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv'], text=True),
        storage_free_gib=shutil.disk_usage(root).free / 2**30,
        scope='Full three-seed disabled-module byte reproduction and independent '
              'official reevaluation. Exposed historical dev, no new blind test. '
              'Baseline BoT retains native output; all new GRU modules use frozen '
              'current matched measurement output. No SGD or historical test reads.')
    write_json(args.output / 'manifest.json', record)

    def execute(command, name):
        if shutil.disk_usage(root).free < 40 * 2**30:
            raise OSError('40 GiB reserve')
        with (args.output / (name + '.log')).open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='2',
                         OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES=''))

    try:
        for name, expected in parent['input_hashes'].items():
            actual = sha256(name)
            if actual != expected:
                raise ValueError('Inherited input changed: ' + name)
            record['input_hashes'][name] = actual
        for seed in range(3):
            path = root / 'phase3' / f'predictor_v3_gru_association_anchor_seed{seed}/epoch8.pt'
            meta = json.loads(path.with_name('manifest.json').read_text())
            if sha256(path) != meta['checkpoint_sha256']:
                raise ValueError('Ordinary GRU weight drift')
            record['input_hashes'][str(path)] = sha256(path)
        print('Preserved hashes verified:', len(record['input_hashes']), flush=True)
        for seed in range(3):
            replay = args.output / f'ordinary_gru_seed{seed}'
            command = [sys.executable, 'tools/cfp_replay_recovery.py', '--kind',
                'ordinary_gru', '--seed', str(seed), '--weight', '.3', '--output', str(replay)]
            execute(command, f'replay_seed{seed}')
            run = json.loads((replay / 'manifest.json').read_text())
            if run['status'] != 'COMPLETE':
                raise ValueError('Disabled-module replay incomplete')
            for label, predictions, reference in [
                ('ordinary_gru_current_measurement', replay,
                 root / 'phase5/output_diagnosis_three_seed_v2' / f'cross_gru_cfp_s{seed}_c1_v1_measurement_evaluation/metrics.json'),
                ('BoT_SORT_RGB_native', root / 'phase1' / f'causal_dev_postneck_bot_sort_seed{seed}_c0',
                 root / 'phase1' / f'causal_dev_postneck_bot_sort_seed{seed}_c0_evaluation/metrics.json')]:
                old = json.loads(reference.read_text())
                output = args.output / f'{label}_seed{seed}_evaluation'
                command = [sys.executable, 'tools/cfp_evaluate.py', '--data-root',
                    '/data3/liuyeqiang/datasets/mft26-ir-causal-v1', '--derived-gt-root',
                    str(root / 'protocol_v2/gt'), '--protocol', 'configs/cfp/protocol_v2.json',
                    '--partition', 'dev', '--predictions', str(predictions), '--output',
                    str(output), '--trackeval-root', '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
                execute(command, f'eval_{label}_{seed}')
                fresh = json.loads((output / 'metrics.json').read_text())
                if fresh['metrics'] != old['metrics'] or fresh['metric_sha256'] != old['metric_sha256']:
                    raise ValueError('Official metrics changed: ' + label)
                for seq in fresh['sequences']:
                    if sha256(predictions / (seq + '.txt')) != old['artifacts'][seq]['prediction_sha256']:
                        raise ValueError('Baseline output changed')
                record['runs'].append(dict(system=label, seed=seed, command=command,
                    metrics=fresh['metrics'], evaluation_sha256=sha256(output / 'metrics.json'),
                    predictions=str(predictions), replay_manifest_sha256=sha256(replay / 'manifest.json')
                    if label.startswith('ordinary') else None,
                    checkpoint_sha256=sha256(root / 'phase3' /
                    f'predictor_v3_gru_association_anchor_seed{seed}/epoch8.pt') if label.startswith('ordinary') else None))
                write_json(args.output / 'manifest.json', record)
                print(label, seed, 'official parity complete', flush=True)
        record['statistics'] = {}
        for label in sorted({r['system'] for r in record['runs']}):
            entries = [r for r in record['runs'] if r['system'] == label]
            record['statistics'][label] = {seq: {field: dict(
                mean=statistics.mean(r['metrics'][seq][field] for r in entries),
                std_population=statistics.pstdev(r['metrics'][seq][field] for r in entries),
                std_sample=statistics.stdev(r['metrics'][seq][field] for r in entries))
                for field in FIELDS} for seq in entries[0]['metrics']}
        record.update(status='BASELINE_REPRODUCED', storage_free_gib=shutil.disk_usage(root).free / 2**30)
        write_json(args.output / 'manifest.json', record)
        write_json('docs/research/PHASE_VI_BASELINE_AUDIT.json', record)
    except Exception as error:
        record.update(status='FAILED', error=repr(error))
        write_json(args.output / 'manifest.json', record)
        raise


if __name__ == '__main__':
    main()
