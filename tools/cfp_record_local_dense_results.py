#!/usr/bin/env python3
"""Retain complete first-seed controls, strong baselines and inner diagnosis."""
import argparse
import json
from pathlib import Path

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap = argparse.ArgumentParser()
    for name in ('root', 'config', 'frozen', 'stages', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('New immutable evidence record required')
    config = json.loads(args.config.read_text())
    frozen = json.loads((args.frozen/'manifest.json').read_text())
    stages = json.loads((args.stages/'manifest.json').read_text())
    if (config['state']!='FROZEN' or frozen['status']!='COMPLETE_FROZEN_MATCHED_LOCAL_SCENES'
            or stages['status']!='COMPLETE_FITTING_INNER_LOCAL_EVIDENCE_STAGE_DIAGNOSIS'
            or config['first_outer'] in stages['videos']):
        raise ValueError('Registered real complete frozen and fitting/inner-only stages required')
    report = dict(status='IN_PROGRESS', completed_stage='COMPLETE_FIRST_SEED_NINE_LOCAL_DENSE_CONTROLS',
        config=config, config_sha256=sha256(args.config), runs={}, frozen=frozen,
        frozen_manifest_sha256=sha256(args.frozen/'manifest.json'), stages=stages,
        stage_manifest_sha256=sha256(args.stages/'manifest.json'),
        source=snapshot_sources(args.output.with_suffix('.sources'), [__file__, args.config]),
        scope='Nine actual firstfoldK4seed0 fitted controls, current GT/known-detector supervision/past-only prediction, independent stopped EMAteacher. All actual GT/verifieddetector-history and current GT/detector pools/gaps1/25/100/supports/unknowns; annotation traces stay outsideGit. Three-seed/allfold/K8/onpolicy-native/rawonline/externalnovelty remain pending. target_absent means verifiedpositive absent, not proof of NULL/physicalabsence; no_id removes explicit IDloss but retains GT temporal grouping.')
    common_epochs = None
    for mode in config['modes']:
        directory = args.root/f'local_dense_{mode}_SN-015_k4_s0_v1'
        manifest = json.loads((directory/'manifest.json').read_text())
        guard = manifest['guard']
        if (manifest['status']!='COMPLETE' or guard['mode']!=mode or guard['seed']!=0
                or guard['inner']!='BT-001' or guard['outer']!='SN-015' or len(guard['fit'])!=4
                or guard['config_sha256']!=sha256(args.config) or manifest['cuda_initialized']
                or manifest['teacher_has_gradient'] or not manifest['student_parameters_changed']
                or any(sha256(p)!=value for p, value in guard['source_sha256'].items())
                or sha256(directory/'final.pt')!=manifest['checkpoint_sha256']
                or any(v in (guard['inner'], guard['outer']) for v, i in manifest['classifier_keys'])
                or len(manifest['epochs'])!=8 or any(e['optimizer_updates']!=150 for e in manifest['epochs'])):
            raise ValueError('Actual completed genuine fit and stopped teacher required: '+mode)
        expected = 1200 if mode in ('full', 'no_ema', 'no_id', 'local_contrastive') else 0
        if manifest['actual_teacher_updates']!=expected:
            raise ValueError('Unexpected active EMA/instant teacher updates')
        epochs = [e['sampled_input_sha256'] for e in manifest['epochs']]
        if common_epochs is None:
            common_epochs = epochs
        elif epochs!=common_epochs:
            raise ValueError('Models did not train on the exact same sampled scenes/slots/masks')
        fields = ('guard', 'epochs', 'evaluations', 'first_gradient_norms', 'allocated_student_parameters',
            'active_student_parameters', 'active_parameters_by_branch', 'teacher_parameters', 'training_seconds',
            'student_parameters_changed', 'local_encoder_l2_change', 'teacher_parameters_changed',
            'teacher_has_gradient', 'actual_teacher_updates', 'cuda_initialized', 'source_commit', 'sources')
        report['runs'][mode] = dict(path=str(directory), manifest_sha256=sha256(directory/'manifest.json'),
            checkpoint_sha256=manifest['checkpoint_sha256'], evidence={name: manifest[name] for name in fields})
        for video in (guard['inner'], guard['outer']):
            for history in ('gt', 'detector'):
                for view in ('gt', 'detector'):
                    for gap in ('1', '25', '100'):
                        e = manifest['evaluations'][video]['protocol'][history][view][gap]
                        base = frozen['models']['cls_fifo'][video]['protocol'][history][view][gap]
                        if e['query_support']['sha256']!=base['query_support']['sha256']:
                            raise ValueError('Frozen and trained perquery history/current/unknown supports differ')
    report['exact_train_sample_digests_and_frozen_query_supports']=True
    report['scientific_result']='Firstseed fullJEPA beats its decoder-IDonly no_prediction onouterSN15 but remains below ordinarylocalTemporal/frozenCLS_FIFO/learnedglobal; no independent JEPA superiority established. InnerBT001fixedcheckpoint retains37.5839R1 in encodedFIFO/contextPool versus26.8456afterprediction, identifying an inference representation bottleneck without using outer diagnosis for recipe choice.'
    write_json(args.output, report)
    print(json.dumps(dict(status=report['status'], completed_models=len(report['runs']), exact_same_input=True)))


if __name__=='__main__':
    main()
