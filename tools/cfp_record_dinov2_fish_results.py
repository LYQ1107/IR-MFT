#!/usr/bin/env python3
"""Record only twelve completed first-fold fits and exact paired diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json


def statistics(values):
    return dict(values=values, mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)))


def main():
    ap = argparse.ArgumentParser()
    for name in ('root', 'config', 'restart', 'test-log', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('New immutable measured report required')
    config = json.loads(args.config.read_text())
    proof = json.loads(args.restart.read_text())
    if (config['state'] != 'FROZEN' or proof['status'] != 'EXACT_RESTART_STATE_VERIFIED'
            or proof.get('producer') != 'dinov2_fish' or proof['differences']
            or '226 passed' not in args.test_log.read_text()):
        raise ValueError('Frozen plan, actual exact DINO restart and passing full suite required')
    report = dict(status='IN_PROGRESS', completed_stage='COMPLETE_FIRST_FOLD_DINOV2_FISH_THREE_SEED_CONTROLS',
        config=config, config_sha256=sha256(args.config), runs={}, summary={},
        restart=dict(path=str(args.restart), sha256=sha256(args.restart), proof=proof),
        uncontrolled_workspace_restart_failure=dict(path=str(args.root/'dinov2_fish_global_exact_restart_v2.json'),
            sha256=sha256(args.root/'dinov2_fish_global_exact_restart_v2.json'),
            scope='Original recipe independent runs already differ before epoch2 checkpoint; this failed proof is retained, not attributed solely to restoration. Exact retry, if obtained, applies to its separately declared deterministic execution policy.'),
        tests=dict(log=str(args.test_log), sha256=sha256(args.test_log), passed=226),
        failed_attempts=dict(path=str(args.root/'dinov2_fish_s0_v1_failed_record_serialization.json'),
            sha256=sha256(args.root/'dinov2_fish_s0_v1_failed_record_serialization.json')),
        sources=snapshot_sources(args.output.with_suffix('.sources'), [__file__, args.config]),
        scope='FirstouterSN015/innerBT001 only. Three registered seeds/four same-input actual visual fits; K4, fixed1200attempted steps/finalepoch8. Full allfold/K8/native-online/JEPA/strong source baseline/external audit/final20 answers remain pending; no local novelty or native MOT claim.')
    manifests, diagnostics = {}, {}
    for mode in config['modes']:
        manifests[mode], diagnostics[mode] = [], []
        for seed in config['seed_plan']:
            name = f'dinov2_fish_{mode}_SN-015_k4_s{seed}_v2'
            directory = args.root/name
            path = args.root/(name+'.diagnostic.json')
            manifest = json.loads((directory/'manifest.json').read_text())
            diagnostic = json.loads(path.read_text())
            guard = manifest['guard']
            if (manifest['status'] != 'COMPLETE' or guard['outer'] != 'SN-015' or guard['inner'] != 'BT-001'
                    or guard['mode'] != mode or guard['seed'] != seed or guard['local_count'] != 4
                    or set(guard['fit']) != {'BT-003', 'MSK-002', 'PF-001', 'SN-001'}
                    or guard['config_sha256'] != sha256(args.config)
                    or sha256(directory/'final.pt') != manifest['checkpoint_sha256']
                    or diagnostic['status'] != 'COMPLETE_MATCHED_TRAINED_DINOV2_CROP_DIAGNOSTIC'
                    or diagnostic['training_manifest_sha256'] != sha256(directory/'manifest.json')
                    or not diagnostic['exact_frozen_reference_and_training_retrieval_parity']
                    or diagnostic['guard'] != guard or len(manifest['epochs']) != 8
                    or not manifest['frozen_prefix_unchanged'] or manifest['trained_backbone_l2_change'] <= 0
                    or any(epoch['attempted_steps'] != 150 for epoch in manifest['epochs'])
                    or any(video in ('SN-015', 'BT-001') for video, identity in manifest['classifier_keys'])):
                raise ValueError('Completed guarded same-input actual fit/diagnostic required: '+name)
            manifests[mode].append(manifest)
            diagnostics[mode].append(diagnostic)
            fields = ('guard', 'classifier_keys', 'parameters', 'trainable_parameters',
                'trainable_backbone_parameters', 'trainable_head_parameters', 'epochs', 'evaluations',
                'first_step_gradient_norms', 'first_step_gradients_finite', 'first_finite_step_gradient_norms',
                'first_finite_step', 'frozen_prefix_unchanged', 'trained_backbone_l2_change',
                'training_seconds', 'peak_gpu_allocated_bytes', 'source_snapshots', 'source_commit')
            report['runs'][name] = dict(training=str(directory), training_manifest_sha256=sha256(directory/'manifest.json'),
                checkpoint_sha256=manifest['checkpoint_sha256'], diagnostic=str(path), diagnostic_sha256=sha256(path),
                training_evidence={field: manifest[field] for field in fields}, diagnostic_evidence=diagnostic['videos'])
    # Equal active architecture and budgets; same sampler/augmentation coverage
    # is checked across four modes, not inferred from nominal configuration.
    for seed in range(3):
        common = manifests['global'][seed]
        for mode in config['modes']:
            candidate = manifests[mode][seed]
            for field in ('classifier_keys',):
                if candidate[field] != common[field]:
                    raise ValueError('Different fitting physical label vocabulary')
            if [e['sample_sameframe_temporal_coverage'] for e in candidate['epochs']] != [e['sample_sameframe_temporal_coverage'] for e in common['epochs']]:
                raise ValueError('Different actual sampled/valid augmentation coverage')
        if manifests['global_capacity'][seed]['trainable_parameters'] != manifests['patch_attention'][seed]['trainable_parameters']:
            raise ValueError('Capacity-matched control is not equal active size')
    report['same_actual_sampling_coverage_across_modes'] = True
    for video in ('BT-001', 'SN-015'):
        report['summary'][video] = {}
        for pool in ('actual_pool', 'paired'):
            report['summary'][video][pool] = {}
            for view in ('gt', 'detector'):
                result = {}
                reference = diagnostics['global'][0]['videos'][video][pool][view]['frozen_cls']
                for mode in config['modes']:
                    result[mode] = {}
                    for feature in diagnostics[mode][0]['videos'][video][pool][view]:
                        metrics = [d['videos'][video][pool][view][feature] for d in diagnostics[mode]]
                        if any((m['retrieval']['25']['queries'], m['retrieval']['25']['supported_queries'],
                                m['discrimination']['positive_edges'], m['discrimination']['negative_edges']) !=
                                (reference['retrieval']['25']['queries'], reference['retrieval']['25']['supported_queries'],
                                reference['discrimination']['positive_edges'], reference['discrimination']['negative_edges']) for m in metrics):
                            raise ValueError('Changed query/gallery or positive/hardnegative edge support')
                        result[mode][feature] = dict(gap25={name: statistics([m['retrieval']['25'][name] for m in metrics])
                            for name in ('rank1_percent', 'rank5_percent', 'mAP_percent')},
                            discrimination={name: statistics([m['discrimination'][name] for m in metrics]) for name in ('auc', 'eer')})
                report['summary'][video][pool][view] = result
    write_json(args.output, report)
    print(json.dumps({mode: report['summary']['SN-015']['actual_pool']['detector'][mode]['embedding']['gap25']['rank1_percent'] for mode in config['modes']}))


if __name__ == '__main__':
    main()
