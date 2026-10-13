#!/usr/bin/env python3
"""Record actual six-fold/three-seed controls without selective outer tuning."""
import argparse
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json


def statistics(values):
    return dict(values=values, mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)))


def main():
    ap = argparse.ArgumentParser()
    for name in ('root', 'config', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--local-count', type=int, choices=(4, 8), required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('New immutable measured allfold report required')
    config = json.loads(args.config.read_text())
    if config['state']!='FROZEN' or config['seed_plan']!=[0, 1, 2]:
        raise ValueError('Predeclared six-fold/three-seed plan required')
    report = dict(status='IN_PROGRESS', completed_stage='COMPLETE_ALL_SIX_FOLDS_THREE_SEED_DINOV2_K'+str(args.local_count),
        config=config, config_sha256=sha256(args.config), local_count=args.local_count, runs={}, summary={},
        source=snapshot_sources(args.output.with_suffix('.sources'), [__file__, args.config]),
        scope='Six leave-two-video-out train-video crop folds, fixed8epochs/1200attempted steps, three seeds/four matched modes. No architecture/seed/checkpoint/threshold selected from outer values. Crop retrieval/discrimination only; no allnative/MOT/external-scene/novelty or completed finalgoal claim.')
    metrics = {}
    for video in config['videos']:
        inner = config['videos'][(config['videos'].index(video)+1) % 6]
        metrics[video] = {}
        for mode in config['modes']:
            metrics[video][mode] = []
            for seed in config['seed_plan']:
                name = f'dinov2_fish_{mode}_{video}_k{args.local_count}_s{seed}_v2'
                directory = args.root/name
                diagnostic_path = args.root/(name+'.diagnostic.json')
                manifest = json.loads((directory/'manifest.json').read_text())
                diagnostic = json.loads(diagnostic_path.read_text())
                guard = manifest['guard']
                if (manifest['status']!='COMPLETE' or guard['outer']!=video or guard['inner']!=inner
                        or guard['seed']!=seed or guard['mode']!=mode or guard['local_count']!=args.local_count
                        or guard['config_sha256']!=sha256(args.config) or len(guard['fit'])!=4
                        or video in guard['fit'] or inner in guard['fit'] or len(manifest['epochs'])!=8
                        or any(e['attempted_steps']!=150 for e in manifest['epochs'])
                        or not manifest['frozen_prefix_unchanged'] or manifest['trained_backbone_l2_change']<=0
                        or any(v in (video, inner) for v, identity in manifest['classifier_keys'])
                        or diagnostic['status']!='COMPLETE_MATCHED_TRAINED_DINOV2_CROP_DIAGNOSTIC'
                        or not diagnostic['exact_frozen_reference_and_training_retrieval_parity']
                        or diagnostic['training_manifest_sha256']!=sha256(directory/'manifest.json')
                        or diagnostic['guard']!=guard or sha256(directory/'final.pt')!=manifest['checkpoint_sha256']
                        or any(sha256(p)!=value for p, value in guard['source_sha256'].items())):
                    raise ValueError('Completed exact allfold fit/diagnostic/provenance required: '+name)
                primary = {pool: {view: diagnostic['videos'][video][pool][view]['embedding'] for view in ('gt', 'detector')}
                           for pool in ('actual_pool', 'paired')}
                metrics[video][mode].append(diagnostic['videos'][video])
                report['runs'][name] = dict(training=str(directory), diagnostic=str(diagnostic_path),
                    manifest_sha256=sha256(directory/'manifest.json'), diagnostic_sha256=sha256(diagnostic_path),
                    checkpoint_sha256=manifest['checkpoint_sha256'], guard=guard,
                    epochs=manifest['epochs'], trainable_parameters=manifest['trainable_parameters'],
                    trainable_head_parameters=manifest['trainable_head_parameters'],
                    trainable_backbone_parameters=manifest['trainable_backbone_parameters'],
                    training_seconds=manifest['training_seconds'], peak_gpu_allocated_bytes=manifest['peak_gpu_allocated_bytes'],
                    frozen_prefix_unchanged=manifest['frozen_prefix_unchanged'],
                    backbone_l2_change=manifest['trained_backbone_l2_change'], source_commit=manifest['source_commit'],
                    outer_primary_metrics=primary, fitting_classifier_keys_count=len(manifest['classifier_keys']),
                    full_local_and_raw_metrics='All registered current/raw/local descriptor metrics retained in exact diagnostic path/hash, not cherry-picked.')
        report['summary'][video] = {}
        for pool in ('actual_pool', 'paired'):
            report['summary'][video][pool] = {}
            for view in ('gt', 'detector'):
                reference = metrics[video]['global'][0][pool][view]['frozen_cls']
                entry = dict(frozen_cls=reference, models={})
                for mode in config['modes']:
                    values = [m[pool][view]['embedding'] for m in metrics[video][mode]]
                    for value in values:
                        for gap in ('1', '25', '100'):
                            for count in ('queries', 'supported_queries', 'target_absent'):
                                if value['retrieval'][gap][count]!=reference['retrieval'][gap][count]:
                                    raise ValueError('Changed actual/paired query support')
                        for count in ('positive_edges', 'negative_edges'):
                            if value['discrimination'][count]!=reference['discrimination'][count]:
                                raise ValueError('Changed positive/current-hardnegative edge support')
                    entry['models'][mode] = dict(retrieval={gap: {metric: statistics([v['retrieval'][gap][metric] for v in values])
                        for metric in ('rank1_percent', 'rank5_percent', 'mAP_percent')} for gap in ('1', '25', '100')},
                        discrimination={metric: statistics([v['discrimination'][metric] for v in values]) for metric in ('auc', 'eer')})
                report['summary'][video][pool][view] = entry
    report['macro_actual_detector_gap25'] = {
        mode: statistics([float(np.mean([metrics[v][mode][s]['actual_pool']['detector']['embedding']['retrieval']['25']['rank1_percent']
                              for v in config['videos']])) for s in config['seed_plan']]) for mode in config['modes']}
    report['macro_frozen_cls_actual_detector_gap25'] = float(np.mean([
        report['summary'][v]['actual_pool']['detector']['frozen_cls']['retrieval']['25']['rank1_percent'] for v in config['videos']]))
    for video in config['videos']:
        for seed in config['seed_plan']:
            coverage = []
            capacity = []
            for mode in config['modes']:
                row = report['runs'][f'dinov2_fish_{mode}_{video}_k{args.local_count}_s{seed}_v2']
                coverage.append([e['sample_sameframe_temporal_coverage'] for e in row['epochs']])
                if mode in ('global_capacity', 'patch_attention'):
                    capacity.append(row['trainable_parameters'])
            if not all(v==coverage[0] for v in coverage) or capacity[0]!=capacity[1]:
                raise ValueError('Unequal actual sampled coverage or active capacity control')
    report['completed_fits']=len(report['runs'])
    report['same_actual_sampling_coverage_and_matched_capacity']=True
    write_json(args.output, report)
    print(json.dumps(dict(completed_fits=len(report['runs']), macro=report['macro_actual_detector_gap25'],
                         frozen_macro=report['macro_frozen_cls_actual_detector_gap25'])))


if __name__=='__main__':
    main()
