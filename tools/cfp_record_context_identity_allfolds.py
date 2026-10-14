#!/usr/bin/env python3
"""Verify all144 unique context fits and all12 exact frozen support protocols."""
import argparse
import json
from pathlib import Path
import numpy as np
from cfp.provenance import sha256, snapshot_sources, write_json


def stats(values):
    return dict(values=values, mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)))


def main():
    ap = argparse.ArgumentParser()
    for name in ['root', 'config', 'frozen', 'output']:
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise ValueError('New immutable allfold measured report required')
    c = json.loads(a.config.read_text())
    frozen = json.loads((a.frozen/'manifest.json').read_text())
    if c['state'] != 'FROZEN' or c['seed_plan'] != [0, 1, 2] or frozen['status'] != 'COMPLETE_FROZEN_ALL_SIX_MATCHED_LOCAL_SCENES' or frozen['config_sha256'] != sha256(a.config):
        raise ValueError('Complete registered allsix same-source frozen controls required')
    aliases = c['control_aliases']
    modes = [m for m in c['modes'] if m not in aliases]
    if len(modes) != 8:
        raise ValueError('Eight actual independent controls; do not count aliases')
    report = dict(status='COMPLETE_ALL_SIX_K4_CONTEXT_IDENTITY_THREE_SEED_CONTROLS', modes=modes, seeds=c['seed_plan'],
                  config_sha256=sha256(a.config), declared_aliases=aliases, runs={}, folds={}, macro={},
                  frozen_manifest=str(a.frozen/'manifest.json'), frozen_manifest_sha256=sha256(a.frozen/'manifest.json'),
                  sources=snapshot_sources(a.output.with_suffix('.sources'), [__file__, a.config]),
                  scope='144unique K4fits, six excluded outer/next-inner/fourfit-video folds, eight distinct controls, three seeds, fixed8epochs/1200successful updates. AllGT/verifieddetector histories and bothcurrent GT/detector pools, gaps1/25/100/supports reported. Clean physical-history crop identity diagnostics, not endogenous/native/rawFPS/external generalization. No prediction/localTemporal aliases are not independent evidence. Strong source/masked/candidate-target/onpolicy/fullnative/finalmethod/novelty remain unfinished.')
    all_runs = {}
    for outer in c['videos']:
        inner = c['videos'][(c['videos'].index(outer)+1) % 6]
        all_runs[outer] = {}
        for mode in modes:
            all_runs[outer][mode] = []
            for seed in c['seed_plan']:
                name = f'local_dense_context_{mode}_{outer}_k4_s{seed}_v2'
                root = a.root/name; d = json.loads((root/'manifest.json').read_text()); g = d['guard']
                if (d['status'] != 'COMPLETE' or g['outer'] != outer or g['inner'] != inner or g['mode'] != mode
                        or g['seed'] != seed or g['local_count'] != 4 or g['config_sha256'] != sha256(a.config)
                        or len(g['fit']) != 4 or outer in g['fit'] or inner in g['fit']
                        or any(v in [outer, inner] for v, identity in d['classifier_keys'])
                        or d['teacher_has_gradient'] or d['cuda_initialized'] or d['online_prediction_decoder_used']
                        or not d['student_parameters_changed'] or len(d['epochs']) != 8
                        or any(e['attempted_steps'] != 150 or e['optimizer_updates'] != 150 for e in d['epochs'])
                        or sha256(root/'final.pt') != d['checkpoint_sha256']
                        or any(sha256(p) != h for p, h in g['source_sha256'].items())):
                    raise ValueError('Actual excluded-fit/gradient/online/checkpoint/source guard failed: '+name)
                fields = ['guard', 'epochs', 'training_seconds', 'active_student_parameters', 'active_parameters_by_branch',
                          'allocated_student_parameters', 'teacher_parameters', 'local_encoder_l2_change', 'actual_teacher_updates',
                          'teacher_has_gradient', 'online_prediction_decoder_used', 'cuda_initialized', 'source_commit']
                report['runs'][name] = dict(path=str(root), manifest_sha256=sha256(root/'manifest.json'),
                                           checkpoint_sha256=d['checkpoint_sha256'], evidence={k: d[k] for k in fields})
                all_runs[outer][mode].append(d)
        for seed in c['seed_plan']:
            expected = [e['sampled_input_sha256'] for e in all_runs[outer]['full'][seed]['epochs']]
            if any([e['sampled_input_sha256'] for e in all_runs[outer][m][seed]['epochs']] != expected for m in modes):
                raise ValueError('Actual scene/current/past/target-mask sampling changed across matched modes')
        report['folds'][outer] = dict(inner=inner, outer_protocols={}, inner_protocols={})
        for evaluation_video, role in [(outer, 'outer_protocols'), (inner, 'inner_protocols')]:
            for history in ['gt', 'detector']:
                for current in ['gt', 'detector']:
                    for gap in ['1', '25', '100']:
                        key = f'{history}_history/{current}_current/gap{gap}'
                        refs = {m: frozen['models'][m][evaluation_video]['protocol'][history][current][gap] for m in frozen['models']}
                        primary = refs['cls_fifo']; entry = dict(query_support=primary['query_support'],
                            frozen={m: dict(retrieval=v['retrieval'], discrimination=v['discrimination']) for m, v in refs.items()}, models={})
                        if any(v['query_support']['sha256'] != primary['query_support']['sha256'] for v in refs.values()):
                            raise ValueError('Frozen same-scene reference support differs')
                        for mode in modes:
                            values = [d['evaluations'][evaluation_video]['protocol'][history][current][gap] for d in all_runs[outer][mode]]
                            if any(v['query_support']['sha256'] != primary['query_support']['sha256'] for v in values):
                                raise ValueError('Actual model/frozen known/unknown/target/candidate/query support differs')
                            entry['models'][mode] = dict(retrieval={k: stats([v['retrieval'][k] for v in values]) for k in ['rank1_percent', 'rank5_percent', 'mAP_percent', 'hard_known_negative_margin_mean']},
                                                        discrimination={k: stats([v['discrimination'][k] for v in values]) for k in ['auc', 'eer']})
                        report['folds'][outer][role][key] = entry
    keys = next(iter(report['folds'].values()))['outer_protocols'].keys()
    for key in keys:
        entry = dict(frozen={}, models={})
        for mode in frozen['models']:
            entry['frozen'][mode] = {section: {metric: float(np.mean([report['folds'][v]['outer_protocols'][key]['frozen'][mode][section][metric] for v in c['videos']]))
                                             for metric in (['rank1_percent', 'rank5_percent', 'mAP_percent'] if section=='retrieval' else ['auc', 'eer'])}
                                   for section in ['retrieval', 'discrimination']}
        for mode in modes:
            entry['models'][mode] = {section: {metric: stats([float(np.mean([report['folds'][v]['outer_protocols'][key]['models'][mode][section][metric]['values'][seed] for v in c['videos']])) for seed in c['seed_plan']])
                                             for metric in (['rank1_percent', 'rank5_percent', 'mAP_percent'] if section=='retrieval' else ['auc', 'eer'])}
                                   for section in ['retrieval', 'discrimination']}
        report['macro'][key] = entry
    report.update(completed_unique_fits=len(report['runs']), successful_optimizer_updates=144*1200,
                  exact_all_sample_digests_and_frozen_support_verified=True)
    write_json(a.output, report)
    print(json.dumps(dict(status=report['status'], fits=report['completed_unique_fits'], primary=report['macro']['gt_history/detector_current/gap25'])))


if __name__ == '__main__':
    main()
