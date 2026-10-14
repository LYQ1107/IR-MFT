#!/usr/bin/env python3
"""All fifteen unique fits, paired seeds/supports and separate target spaces."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cfp.crop_retrieval import embedding_crop_retrieval
from cfp.provenance import sha256, snapshot_sources, write_json


def stats(values):
    return dict(values=values,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('New immutable all-seed evidence required')
    torch.set_num_threads(2)
    features=a.root/'dinov2_two_crop_evidence_v1';fm=json.loads((features/'manifest.json').read_text())
    recipes=[('latent','pixel_masked_identity','configs/cfp/phase_vii_pixel_masked_identity_v1.json'),
             ('RGB','pixel_reconstruction_identity','configs/cfp/phase_vii_pixel_reconstruction_identity_v1.json')]
    report=dict(status='COMPLETE_FIRST_FOLD_PIXEL_IDENTITY_THREE_SEED_CONTROLS',runs={},summary={},frozen={},recipe_details={},
        sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,'cfp/crop_retrieval.py']),
        scope='Nine latent-target andsixRGB-target unique actual8epoch firstfold fits, eachthree seeds; two restart duplicates not counted independently. Allfourfitvideo gradients only, excludesouterSN015/innerBT001. Same immutable crop source/query/gallery support; registered RNG/sampling/augmentation verified. Teacher latentL1 and RGBpixelMSE have explicitly different outputcapacity/forwardcompute. This is past-only wholevideo same-view cropgallery retrieval, not Bsinglecurrentscene/nativeMOT/newmethod/novelty/fullgoal completion.')
    loaded={}
    for recipe,prefix,config_path in recipes:
        c=json.loads(Path(config_path).read_text());report['recipe_details'][recipe]=dict(config=c,config_sha256=sha256(config_path))
        for mode in c['modes']:
            loaded[mode]=[]
            for seed in c['seed_plan']:
                root=a.root/f'{prefix}_{mode}_SN-015_k4_s{seed}_v1';m=json.loads((root/'manifest.json').read_text());g=m['guard']
                if (m['status']!='COMPLETE' or g['mode']!=mode or g['seed']!=seed or g['outer']!='SN-015' or g['inner']!='BT-001'
                        or set(g['fit']) & {'SN-015','BT-001'} or len(g['fit'])!=4
                        or g['config_sha256']!=sha256(config_path) or g['features_sha256']!=sha256(features/'manifest.json')
                        or len(m['epochs'])!=8 or any(e['attempted_steps']!=150 for e in m['epochs'])
                        or not m['frozen_prefix_unchanged'] or m['trained_backbone_l2_change']<=0
                        or any(v in ['SN-015','BT-001'] for v,i in m['classifier_keys'])
                        or sha256(root/'final.pt')!=m['checkpoint_sha256']
                        or any(sha256(p)!=h for p,h in g['source_sha256'].items())):
                    raise ValueError('Exact completed fit/source/split/checkpoint guards required: '+str(root))
                saved=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
                rng={k:saved[k] for k in ['rng_sampler','rng_augmentation','rng_mask']}
                loaded[mode].append(dict(manifest=m,rng=rng))
                for video in ['BT-001','SN-015']:
                    e=m['evaluations'][video]
                    if e['records_sha256']!=fm['videos'][video]['records_sha256'] or any(sha256(root/video/(name+'.npy'))!=v['sha256'] for name,v in e['artifacts'].items()):
                        raise ValueError('Exact heldout current representation and records required')
                report['runs'][root.name]=dict(path=str(root),manifest_sha256=sha256(root/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256'],
                    guard=g,epochs=m['epochs'],training_seconds=m['training_seconds'],trainable_parameters=m['trainable_parameters'],
                    trainable_backbone_parameters=m['trainable_backbone_parameters'],trainable_head_parameters=m['trainable_head_parameters'],
                    first_finite_step_gradient_norms=m['first_finite_step_gradient_norms'],backbone_l2_change=m['trained_backbone_l2_change'],
                    teacher_or_success_counter=m['epochs'][-1].get('teacher_updates',m['epochs'][-1].get('successful_updates')),
                    evaluations=m['evaluations'],registered_sampling_final_rng=rng)
    modes=list(loaded)
    for seed in [0,1,2]:
        states=[loaded[mode][seed]['rng'] for mode in modes]
        if any(json.dumps(v,sort_keys=True)!=json.dumps(states[0],sort_keys=True) for v in states):
            raise ValueError('Sampler/augmentation/mask final registered sequence state differs across modes')
    for video in ['BT-001','SN-015']:
        records=json.loads((features/video/'records.json').read_text());valid=np.load(features/video/'valid.npy')
        if sha256(features/video/'cls.npy')!=fm['videos'][video]['artifacts']['cls']['sha256']:raise ValueError('Frozen same-source descriptor drift')
        report['frozen'][video]=embedding_crop_retrieval(np.load(features/video/'cls.npy',mmap_mode='r'),records,valid)
        report['summary'][video]={}
        for view in ['gt','detector']:
            report['summary'][video][view]={}
            for gap in ['1','25','100']:
                baseline=report['frozen'][video][view][gap];entry=dict(frozen_cls=baseline,models={})
                for mode in modes:
                    values=[d['manifest']['evaluations'][video]['retrieval'][view][gap] for d in loaded[mode]]
                    if any(any(v[k]!=baseline[k] for k in ['queries','supported_queries','missing_target']) for v in values):
                        raise ValueError('Actual known/unknown/past-only query/gallery support differs')
                    entry['models'][mode]={k:stats([v[k] for v in values]) for k in ['rank1_percent','rank5_percent','mAP_percent','unknown_selected']}
                report['summary'][video][view][gap]=entry
    report.update(unique_fits=len(report['runs']),same_registered_sampling_rng_and_support_verified=True)
    for file in ['pixel_masked_identity_exact_restart_v1.json','pixel_reconstruction_identity_exact_restart_v1.json']:
        d=json.loads((a.root/file).read_text())
        if d['status']!='EXACT_RESTART_STATE_VERIFIED' or d['differences']:raise ValueError('Actual completed restart proof required')
        report.setdefault('actual_restart_proofs',[]).append(dict(path=str(a.root/file),sha256=sha256(a.root/file),counts=d['counts']))
    write_json(a.output,report)
    print(json.dumps(dict(status=report['status'],unique_fits=report['unique_fits'],outer_primary=report['summary']['SN-015']['detector']['25'])))


if __name__=='__main__':main()
