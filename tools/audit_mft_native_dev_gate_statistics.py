#!/usr/bin/env python3
"""Audit every saved gate summary in an already complete normal-stage Dev report.

Reads original production metadata and existing official Dev metrics on CPU.
It neither runs a network nor selects a model, and cannot recover per-frame
calibration or object-specific behavior from saved aggregate moments.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time


ROOT = Path(__file__).resolve().parents[1]
DEV = {'BT-005': 1834, 'SN-013': 3000}
METRICS = ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW')
MOMENTS = ('quality_gate', 'temporal_gain', 'internal_covariance')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def table(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_moments(row, count):
    require(row['count'] == count, 'Wrong recorded diagnostic sample count')
    if not count:
        require(all(row[k] is None for k in ('mean', 'std', 'min', 'max')), 'Do not invent missing statistics')
        return
    require(all(isinstance(row[k], (int, float)) and math.isfinite(row[k])
                for k in ('mean', 'std', 'min', 'max')), 'Nonfinite recorded moments')
    require(row['std'] >= 0 and row['min'] <= row['mean']+1e-12 <= row['max']+1e-12,
            'Inconsistent recorded moments')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-report', type=Path, default=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json')
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use an explicitly CPU-only process')
    started = time.monotonic()
    stat = Path('/proc/self/stat').read_text()
    command = Path('/proc/self/cmdline').read_bytes()
    identity = dict(pid=os.getpid(), start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
                    raw_argv_sha256=hashlib.sha256(command).hexdigest(), CUDA_VISIBLE_DEVICES='')
    parent = read(args.dev_report)
    stage = parent['stage']
    sequence_coverage = parent['sequences']
    require(stage in (6, 9, 12) and parent['split'] == 'dev' and parent['frames'] == 4834 and
            (sequence_coverage == DEV if isinstance(sequence_coverage, dict) else sequence_coverage == list(DEV)),
            'Complete original normal-stage Dev only')
    require(len(parent['rows']) == 10*stage+2, 'Preserve all normal, off, roll and both source cases')
    sources = {str(args.dev_report): sha(args.dev_report)}
    for path in [Path(__file__).resolve(), ROOT/'tools/native_pyramid_gate_observer.py',
                 ROOT/'yolox/models/native_pyramid_fusion.py', ROOT/'tools/evaluate_native_pyramid.py']:
        sources[str(path)] = sha(path)
    models, index, manifests, referenced = {}, {}, {}, collections.defaultdict(set)
    for row in parent['rows']:
        require(row['split'] == 'dev', 'Test accuracy is outside this audit')
        for key in ('original_metrics', 'official_CSV'):
            ref = row[key]
            path = ROOT/ref['file']
            require(sha(path) == ref['sha256'], 'Original complete Dev artifact changed')
            sources[str(path)] = ref['sha256']
        config = row['config']
        if not config.get('fit_id'):
            continue
        require(config['seed'] == 0 and config['removal'] is None and config['split'] == 'dev',
                'Only original normal-stage seed0 models and their interventions')
        key = config['mode'], config['cue'], config['epoch'], config['intervention']
        require(key not in index and row['id'] not in models, 'Duplicate original condition')
        index[key], models[row['id']] = row['id'], row
        for sequence, ref in row['original_source_manifest_references'].items():
            require(sequence in DEV, 'Unexpected sequence')
            path = ref['manifest_path']
            require(sha(path) == ref['manifest_sha256'], 'Original production manifest changed')
            sources[path] = ref['manifest_sha256']
            if path not in manifests:
                manifests[path] = read(path)
            referenced[path].add(row['id'])
    expected = {(m, c, e, 'normal') for m in ('quality', 'temporal') for c in ('rgb', 'gray', 'ir')
                for e in range(1, stage+1)}
    expected |= {(m, 'ir', e, i) for m in ('quality', 'temporal') for e in range(1, stage+1)
                 for i in ('unavailable', 'IR_image_spatial_roll')}
    require(set(index) == expected, 'Incomplete original same-budget conditions')
    for path, manifest in manifests.items():
        require(manifest['status'] == 'COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT' and
                manifest['split'] == 'dev' and manifest['sequence'] in DEV and
                manifest['frames'] == DEV[manifest['sequence']] and not manifest['GT_read'] and
                not manifest['future_frame_input'], 'Incomplete or wrong-split producer')
        require(set(manifest['gate_statistics']) == referenced[path], 'Omitted or invented producer gate condition')
    flat, selected = [], {}
    for name, row in models.items():
        config = row['config']
        for sequence, length in DEV.items():
            ref = row['original_source_manifest_references'][sequence]
            manifest = manifests[ref['manifest_path']]
            require(manifest['specification_sha256'] == config['specification_sha256'], 'Producer specification differs')
            original = manifest['models'][name]
            for key in ('id', 'fit_id', 'mode', 'cue', 'epoch', 'seed', 'intervention', 'checkpoint_sha256', 'fit_config_sha256'):
                require(original[key] == config[key], 'Actual produced model identity differs')
            stats = manifest['gate_statistics'][name]
            missing = config['intervention'] == 'unavailable'
            require(stats['frames'] == length and stats['available_frames'] == (0 if missing else length) and
                    stats['missing_frames'] == (length if missing else 0) and stats['mode'] == config['mode'] and
                    stats['quality_gate_enabled'] and stats['feature_protection_enabled'], 'Gate availability or architecture differs')
            require([v['level'] for v in stats['levels']] == ['P3', 'P4', 'P5'], 'Need all three levels')
            official = next(v for v in row['per_sequence47'] if v['seq'] == sequence)
            for level in stats['levels']:
                valid = level['valid_cells']
                require(valid == 0 if missing else valid > 0, 'Invalid feature cell coverage')
                validate_moments(level['quality_gate'], stats['available_frames'])
                validate_moments(level['temporal_gain'], valid)
                validate_moments(level['internal_covariance'], valid if config['mode'] == 'temporal' else 0)
                for key in ('nonzero_feature_correction_fraction', 'feature_correction_RMS', 'relative_feature_correction_RMS'):
                    require(level[key] is None if missing else math.isfinite(level[key]) and level[key] >= 0,
                            'Invalid feature correction statistic')
                require(0 <= level['reliable_teacher_cells'] <= valid and level['reliable_changed_feature_cells'] == 0,
                        'Protected feature cells changed')
                record = dict(id=name, mode=config['mode'], cue=config['cue'], epoch=config['epoch'],
                    intervention=config['intervention'], sequence=sequence, frames=length,
                    available_frames=stats['available_frames'], missing_frames=stats['missing_frames'], level=level['level'])
                for moment in MOMENTS:
                    record.update({moment+'_'+k: v for k, v in level[moment].items()})
                record.update({k: level[k] for k in ('valid_cells', 'nonzero_feature_correction_fraction',
                    'feature_correction_RMS', 'relative_feature_correction_RMS', 'reliable_teacher_cells', 'reliable_changed_feature_cells')})
                record['protected_teacher_cell_fraction'] = level['reliable_teacher_cells']/valid if valid else None
                record.update({k: official[k] for k in METRICS})
                flat.append(record)
                selected[name, sequence, level['level']] = record
    paired, comparisons = [], []
    for mode in ('quality', 'temporal'):
        for epoch in range(1, stage+1):
            name = index[mode, 'ir', epoch, 'normal']
            rolled = index[mode, 'ir', epoch, 'IR_image_spatial_roll']
            require(models[name]['config']['checkpoint_sha256'] == models[rolled]['config']['checkpoint_sha256'],
                    'A roll intervention changed the checkpoint')
            for sequence in DEV:
                for level in ('P3', 'P4', 'P5'):
                    a, b = selected[name, sequence, level], selected[rolled, sequence, level]
                    paired.append(dict(mode=mode, epoch=epoch, sequence=sequence, level=level,
                        normal_id=name, rolled_id=rolled, normal_gate=a['quality_gate_mean'], rolled_gate=b['quality_gate_mean'],
                        rolled_minus_normal_gate=b['quality_gate_mean']-a['quality_gate_mean'],
                        normal_relative_RMS=a['relative_feature_correction_RMS'], rolled_relative_RMS=b['relative_feature_correction_RMS'],
                        rolled_minus_normal_relative_RMS=b['relative_feature_correction_RMS']-a['relative_feature_correction_RMS'],
                        normal_HOTA=a['HOTA'], rolled_HOTA=b['HOTA'], normal_minus_rolled_HOTA=a['HOTA']-b['HOTA']))
                comparisons.append(dict(mode=mode, epoch=epoch, sequence=sequence,
                    normal_minus_rolled_HOTA=selected[name, sequence, 'P3']['HOTA']-selected[rolled, sequence, 'P3']['HOTA']))
    counts = dict(level_rows=len(flat), normal_roll_level_pairs=len(paired), model_sequence_pairs=len(comparisons),
        rolled_mean_gate_higher=sum(r['rolled_minus_normal_gate'] > 0 for r in paired),
        rolled_mean_gate_lower=sum(r['rolled_minus_normal_gate'] < 0 for r in paired),
        rolled_relative_RMS_higher=sum(r['rolled_minus_normal_relative_RMS'] > 0 for r in paired),
        normal_HOTA_higher=sum(r['normal_minus_rolled_HOTA'] > 0 for r in comparisons),
        normal_HOTA_lower=sum(r['normal_minus_rolled_HOTA'] < 0 for r in comparisons))
    output = args.output_directory
    output.mkdir(exist_ok=False)
    table(output/'all_complete_Dev_gate_levels.csv', flat)
    table(output/'all_paired_normal_vs_roll.csv', paired)
    capsules = []
    for i, (path, manifest) in enumerate(sorted(manifests.items())):
        capsule = dict(original_manifest=path, original_manifest_sha256=sources[path],
            sequence=manifest['sequence'], split='dev', frames=manifest['frames'],
            specification_sha256=manifest['specification_sha256'],
            original_image_bytes_delivery_sha256=manifest['original_image_bytes_delivery_sha256'],
            gate_statistics=manifest['gate_statistics'], extraction_scope='Exact parsed original fields, not a byte-for-byte copy of the whole producer manifest.')
        filename=f'producer_{i:02d}_{manifest["sequence"]}_gate_fields.json'
        write(output/filename, capsule)
        require(read(output/filename)['gate_statistics'] == manifest['gate_statistics'], 'Lossy extracted gate fields')
        capsules.append(dict(file=filename, sha256=sha(output/filename), original_manifest_sha256=sources[path]))
    require(all(sha(p) == h for p, h in sources.items()), 'An original source changed during analysis')
    report = dict(status='COMPLETE_ORIGINAL_ALL_NORMAL_STAGE_DEV_GATE_DIAGNOSTIC_NOT_CALIBRATION_OR_CONTRIBUTION_PROOF',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), actual_CPU_identity=identity,
        stage=stage, split='dev', frames_each_condition=4834, sequences=list(DEV), formal_conditions_with_gates=len(models),
        **counts, by_mode={m:dict(level_pairs=sum(r['mode']==m for r in paired),
            rolled_mean_gate_higher=sum(r['mode']==m and r['rolled_minus_normal_gate']>0 for r in paired),
            rolled_relative_RMS_higher=sum(r['mode']==m and r['rolled_minus_normal_relative_RMS']>0 for r in paired)) for m in ('quality','temporal')},
        paired_level_rows=paired, model_sequence_comparisons=comparisons,
        all_recorded_protected_cells_unchanged=True, unavailable_gate_fields_preserved_as_missing=True,
        last_epoch_IR_coverage=[r for r in flat if r['cue']=='ir' and r['epoch']==stage and r['intervention']=='normal'],
        original_source_hashes=sources, exact_field_extractions=capsules,
        parent_Dev_report=str(args.dev_report), parent_Dev_report_sha256=sha(args.dev_report),
        new_TrackEval_run=False, metrics_scope='Reuse already independently verified official per-sequence Dev values; this tool verifies source file hashes and does not rerun metric calculations.',
        moment_scope='Gate moments weight one scalar per available frame/level; gain and covariance moments weight valid feature cells. These cells and levels are not independent statistical replications.',
        limitations=['Only saved per-sequence moments are available; no framewise gate ordering, object attribution, calibration error or AUROC is recoverable.',
            'A higher gate mean under roll is a descriptive mismatch response, not proof that the gate caused the HOTA difference.',
            'Nonzero feature correction marks any channel difference after dtype conversion; it is not the fraction of changed detections or identities.',
            'Feature protection does not imply unchanged neighboring detector outputs; internal covariance is not calibrated physical sensor noise.',
            'Preserve all epochs and both Dev videos, including opposite effects. No Test accuracy or GT is read and no new model is selected.'],
        additional_seed_work=False, new_GPU_or_NN_work=False, new_Dev_or_Test_accuracy=False,
        supported_innovations=0, goal_met=False, actual_CPU_wall_seconds=time.monotonic()-started,
        actual_peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    write(output/'complete.json', report)
    print(json.dumps(dict(status=report['status'], counts=counts, by_mode=report['by_mode'],
                         CPU_seconds=report['actual_CPU_wall_seconds'], peak_RSS_GiB=report['actual_peak_RSS_GiB'])), flush=True)


if __name__ == '__main__':
    main()
