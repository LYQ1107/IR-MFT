#!/usr/bin/env python3
"""Execute the frozen 36-cell crossover and selected output-only interventions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from cfp.provenance import sha256, write_json, snapshot_sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError('Bounded three-process CPU campaign')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    directory = root / 'phase5'
    phase_path = Path('configs/cfp/phase_v_protocol_v1.json')
    phase_hash = sha256(phase_path)
    protocol = json.loads(phase_path.read_text())
    jobs = [(a, b, s, c) for s in (0, 1, 2) for c in (0, 1, 2) for a, b in [('bot', 'bot'), ('gru', 'cfp'), ('bot', 'cfp'), ('gru', 'bot')]]
    record = dict(status='RUNNING', phase_protocol_sha256=phase_hash, planned_jobs=jobs, runs={},
                  source_snapshots=snapshot_sources(args.output.parent / (args.output.stem + '_sources'), [__file__, phase_path]),
                  output_interventions={}, scope='36 complete dev runs, same three old association candidates per cell; no training or per-video thresholds. Original diagonal full-file parity enforced.')
    if args.output.exists():
        prior = json.loads(args.output.read_text())
        if prior['phase_protocol_sha256'] != phase_hash or prior['planned_jobs'] != [list(v) for v in jobs]:
            raise ValueError('Resume plan differs')
    write_json(args.output, record)

    def execute(command, log):
        if shutil.disk_usage(root).free < 40 * 2**30:
            raise OSError('40GiB reserve')
        with log.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='0'))
        if result.returncode:
            raise RuntimeError(f'Command failed ({result.returncode}); preserved log: {log}')

    def evaluate(predictions, output):
        command = [sys.executable, 'tools/cfp_evaluate.py', '--data-root', '/data3/liuyeqiang/datasets/mft26-ir-causal-v1',
                   '--derived-gt-root', str(root / 'protocol_v2/gt'), '--protocol', 'configs/cfp/protocol_v2.json', '--partition', 'dev',
                   '--predictions', str(predictions), '--output', str(output), '--trackeval-root', '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
        if not (output / 'metrics.json').exists():
            execute(command, output.with_suffix('.log'))
        metrics = json.loads((output / 'metrics.json').read_text())
        for seq in ('BT-005', 'SN-013'):
            if sha256(predictions / (seq + '.txt')) != metrics['artifacts'][seq]['prediction_sha256']:
                raise ValueError('Evaluation output provenance changed')
        return dict(metrics=metrics['metrics'], evaluation_sha256=sha256(output / 'metrics.json'), evaluation_command=command)

    def job(values):
        a, b, seed, candidate = values
        name = f'cross_{a}_{b}_s{seed}_c{candidate}_v1'
        output = directory / name
        command = [sys.executable, 'tools/cfp_replay_crossover.py', '--association', a, '--lifecycle', b, '--candidate', str(candidate),
                   '--seed', str(seed), '--output', str(output)]
        if not output.exists():
            execute(command, directory / (name + '.log'))
        run = json.loads((output / 'manifest.json').read_text())
        if (run['status'] != 'COMPLETE' or run['phase_protocol_sha256'] != phase_hash or run['seed'] != seed or run['candidate'] != candidate
                or run['association'] != a or run['lifecycle'] != b or run['probe']):
            raise ValueError('Existing crossover incomplete or differs; preserve it and repair before resuming: ' + name)
        for seq in ('BT-005', 'SN-013'):
            if sha256(output / (seq + '.txt')) != run['videos'][seq]['prediction_sha256'] or sha256(output / (seq + '.decisions.jsonl.gz')) != run['videos'][seq]['decisions_sha256']:
                raise ValueError('Crossover outputs changed')
        evaluation = evaluate(output, directory / (name + '_evaluation'))
        return name, dict(run_manifest_sha256=sha256(output / 'manifest.json'), replay_command=command, **evaluation)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(job, values) for values in jobs]
            for future in as_completed(futures):
                name, entry = future.result()
                record['runs'][name] = entry
                write_json(args.output, record)
                print('Complete', len(record['runs']), '/', len(jobs), name, 'HOTA', entry['metrics']['COMBINED_SEQ']['HOTA'], flush=True)
        for name in ('cross_bot_bot_s0_c0_v1', 'cross_gru_cfp_s0_c1_v1'):
            original = directory / name
            for variant in ('confirmed', 'all', 'measurement'):
                predictions = directory / (name + '_output_' + variant)
                predictions.mkdir(exist_ok=True)
                for seq in ('BT-005', 'SN-013'):
                    dest = predictions / (seq + '.txt')
                    source = original / (seq + '.' + variant + '.txt')
                    if not dest.exists():
                        dest.symlink_to(source)
                    if dest.resolve() != source.resolve():
                        raise ValueError('Output intervention symlink drift')
                record['output_interventions'][name + '_' + variant] = evaluate(predictions, predictions.with_name(predictions.name + '_evaluation'))
                write_json(args.output, record)
        record['status'] = 'COMPLETE'
        write_json(args.output, record)
    except Exception as error:
        record.update(status='FAILED', error=str(error))
        write_json(args.output, record)
        raise


if __name__ == '__main__':
    main()
