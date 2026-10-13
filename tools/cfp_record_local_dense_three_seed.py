#!/usr/bin/env python3
"""Record all models/seeds/protocols; no selective identity gain claim."""
import argparse
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json


def stats(values):
    return dict(values=values,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)))


def main():
    ap=argparse.ArgumentParser()
    for name in ('root','config','frozen','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--context-identity',action='store_true')
    args=ap.parse_args()
    if args.output.exists():
        raise ValueError('New immutable actual three-seed report required')
    config=json.loads(args.config.read_text())
    frozen=json.loads((args.frozen/'manifest.json').read_text())
    if config['state']!='FROZEN' or config['seed_plan']!=[0,1,2] or frozen['status']!='COMPLETE_FROZEN_MATCHED_LOCAL_SCENES':
        raise ValueError('Complete fixed sampled-scene reference and registered three seeds required')
    prefix='local_dense_context_' if args.context_identity else 'local_dense_'
    version='v2' if args.context_identity else 'v1'
    report=dict(status='IN_PROGRESS',completed_stage='COMPLETE_LOCAL_DENSE_FIRST_FOLD_THREE_SEED_'+version.upper(),
        config=config,config_sha256=sha256(args.config),context_identity=args.context_identity,
        runs={},summary={},frozen_reference_manifest_sha256=sha256(args.frozen/'manifest.json'),
        source=snapshot_sources(args.output.with_suffix('.sources'),[__file__,args.config]),
        scope='Allnine labels/three seeds complete firstfoldK4only. Exact same sampledscenes/targetmasks/currentcandidate/support/provenance; allGT/verifieddetector-history,currentGT/detector,gaps1/25/100 reported. These are cleanphysical diagnostic groups, notnative tracker memories/MOT. NoIDremovesexplicit loss butretainsGT temporalgrouping; targetabsent isverified-positive absence, notconfirmedNULL. Allfold/K8/onpolicy/native/rawruntime/strongsource/external/finalgoal unfinished.')
    collected={}
    for mode in config['modes']:
        collected[mode]=[]
        for seed in config['seed_plan']:
            name=f'{prefix}{mode}_SN-015_k4_s{seed}_{version}'
            directory=args.root/name
            d=json.loads((directory/'manifest.json').read_text());g=d['guard']
            if (d['status']!='COMPLETE' or g['mode']!=mode or g['seed']!=seed or g['outer']!='SN-015' or g['inner']!='BT-001'
                    or g['local_count']!=4 or g['config_sha256']!=sha256(args.config) or d['teacher_has_gradient']
                    or d['cuda_initialized'] or any(v in ('SN-015','BT-001') for v,i in d['classifier_keys'])
                    or len(d['epochs'])!=8 or any(e['optimizer_updates']!=150 for e in d['epochs'])
                    or sha256(directory/'final.pt')!=d['checkpoint_sha256']
                    or any(sha256(p)!=v for p,v in g['source_sha256'].items())):
                raise ValueError('Completed actual fitting-only model/source/checkpoint required: '+name)
            collected[mode].append(d)
            fields=('guard','epochs','evaluations','first_gradient_norms','active_student_parameters',
                'active_parameters_by_branch','allocated_student_parameters','teacher_parameters','training_seconds',
                'student_parameters_changed','local_encoder_l2_change','teacher_parameters_changed','actual_teacher_updates',
                'teacher_has_gradient','cuda_initialized','source_commit','sources')
            report['runs'][name]=dict(path=str(directory),manifest_sha256=sha256(directory/'manifest.json'),
                checkpoint_sha256=d['checkpoint_sha256'],evidence={f:d[f] for f in fields})
    for seed in config['seed_plan']:
        input_hashes=[e['sampled_input_sha256'] for e in collected['full'][seed]['epochs']]
        if any([e['sampled_input_sha256'] for e in collected[m][seed]['epochs']]!=input_hashes for m in config['modes']):
            raise ValueError('Different actual sampled scene/slot/mask sequence')
    for video in ('BT-001','SN-015'):
        report['summary'][video]={}
        for history in ('gt','detector'):
            report['summary'][video][history]={}
            for current in ('gt','detector'):
                report['summary'][video][history][current]={}
                for gap in ('1','25','100'):
                    ref=frozen['models']['cls_fifo'][video]['protocol'][history][current][gap]
                    entry=dict(frozen_cls_fifo=ref,models={})
                    for mode in config['modes']:
                        values=[d['evaluations'][video]['protocol'][history][current][gap] for d in collected[mode]]
                        if any(v['query_support']['sha256']!=ref['query_support']['sha256'] for v in values):
                            raise ValueError('Different actual known/unknown/context/candidate/query support')
                        entry['models'][mode]=dict(retrieval={key:stats([v['retrieval'][key] for v in values])
                            for key in ('rank1_percent','rank5_percent','mAP_percent','hard_known_negative_margin_mean')},
                            discrimination={key:stats([v['discrimination'][key] for v in values]) for key in ('auc','eer')})
                    report['summary'][video][history][current][gap]=entry
    report.update(completed_fits=len(report['runs']),exact_input_digests_and_frozen_support=True)
    if args.context_identity:
        report['declared_control_aliases']=config['control_aliases']
    write_json(args.output,report)
    print(json.dumps({m:report['summary']['SN-015']['gt']['detector']['25']['models'][m]['retrieval']['rank1_percent'] for m in config['modes']}))


if __name__=='__main__':
    main()
