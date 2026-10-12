#!/usr/bin/env python3
"""Measure three output policies on fixed existing internal tracking decisions."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Output diagnostic is immutable')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    plan_path = Path('configs/cfp/phase_v_output_diagnosis_v2.json')
    plan = json.loads(plan_path.read_text())
    campaign_path = root / 'phase5/crossover_campaign_v1.json'
    campaign = json.loads(campaign_path.read_text())
    if plan['state'] != 'FROZEN' or campaign['status'] != 'COMPLETE' or sha256(campaign_path) != plan['parent_campaign_sha256']:
        raise ValueError('Immutable complete parent required')
    args.output.mkdir(parents=True)
    record = dict(status='RUNNING', plan_sha256=sha256(plan_path), source_snapshots=snapshot_sources(args.output, [__file__, plan_path]),
                  runs={}, selected_means={}, scope=plan['scope'])
    write_json(args.output / 'manifest.json', record)
    for cell, candidate in plan['families'].items():
        for variant in plan['output_policies']:
            measurements = []
            for seed in plan['seeds']:
                name = f'cross_{cell}_s{seed}_c{candidate}_v1'
                run = root / 'phase5' / name
                predictions = args.output / f'{name}_{variant}'
                predictions.mkdir()
                for seq in ('BT-005', 'SN-013'):
                    (predictions / (seq + '.txt')).symlink_to(run / (seq + '.' + variant + '.txt'))
                evaluation = predictions.with_name(predictions.name + '_evaluation')
                command = [sys.executable, 'tools/cfp_evaluate.py', '--data-root', '/data3/liuyeqiang/datasets/mft26-ir-causal-v1',
                           '--derived-gt-root', str(root / 'protocol_v2/gt'), '--protocol', 'configs/cfp/protocol_v2.json', '--partition', 'dev',
                           '--predictions', str(predictions), '--output', str(evaluation), '--trackeval-root', '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
                with evaluation.with_suffix('.log').open('w') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
                metrics = json.loads((evaluation / 'metrics.json').read_text())
                measurements.append(metrics['metrics'])
                record['runs'][name + '_' + variant] = dict(command=command, internal_run_manifest_sha256=sha256(run / 'manifest.json'),
                                                           metrics=metrics['metrics'], evaluation_sha256=sha256(evaluation / 'metrics.json'))
                write_json(args.output / 'manifest.json', record)
            scalar_fields = [key for key, value in measurements[0]['COMBINED_SEQ'].items() if isinstance(value, (int, float))]
            record['selected_means'][cell + '_' + variant] = {
                seq: dict(mean={key: statistics.mean(x[seq][key] for x in measurements) for key in scalar_fields},
                          sample_std={key: statistics.stdev(x[seq][key] for x in measurements) for key in scalar_fields})
                for seq in ('BT-005', 'SN-013', 'COMBINED_SEQ')}
            write_json(args.output / 'manifest.json', record)
            print(cell, variant, 'HOTA', record['selected_means'][cell + '_' + variant]['COMBINED_SEQ']['mean']['HOTA'], flush=True)
    record['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', record)


if __name__ == '__main__':
    main()
