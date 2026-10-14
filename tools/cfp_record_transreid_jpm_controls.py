#!/usr/bin/env python3
"""Complete all nine source-JPM crop fits; never select outer winner/seed."""
import argparse,json
from pathlib import Path
import numpy as np
from cfp.provenance import sha256,snapshot_sources,write_json


def summary(values):return dict(values=values,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('New immutable allseed report required')
    cfg=Path('configs/cfp/phase_vii_transreid_dino_jpm_v1.json');c=json.loads(cfg.read_text());result=dict(status='COMPLETE_FIRST_FOLD_SOURCE_JPM_THREE_SEED_CROP_CONTROLS',config=c,config_sha256=sha256(cfg),fits={},summary={},sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,cfg]),scope='Nine firstfold actualgradient8epoch/1200attempt fits; three sourceJPM/sourceglobal/samecapacityCLS controls; fixedfinalcheckpoint, strictpast wholevideo-gallery retrieval atGT/detector gaps1/25/100, allcounts/unknownselections/R1/R5/mAP retained. Sameinput digests/fit-onlyBN/prefix andsourcecapacity verified. NotBsingle-scene retrieval, nativeMOT/officialwholeTransReID/newnovelty/finalmethod.')
    models={}
    for mode in c['modes']:
        models[mode]=[]
        for seed in c['seed_plan']:
            path=a.root/f'transreid_dino_{mode}_SN-015_s{seed}_v1';m=json.loads((path/'manifest.json').read_text());g=m['guard']
            if (m['status']!='COMPLETE' or g['config_sha256']!=sha256(cfg) or g['mode']!=mode or g['seed']!=seed or g['outer']!='SN-015' or g['inner']!='BT-001' or set(g['fit'])!=set(['BT-003','MSK-002','PF-001','SN-001']) or not m['frozen_prefix_unchanged'] or not m['heldout_BN_buffers_unchanged'] or len(m['epochs'])!=8 or any(e['attempted_steps']!=150 for e in m['epochs']) or sha256(path/'final.pt')!=m['checkpoint_sha256'] or any(sha256(p)!=h for p,h in g['source_sha256'].items())):raise ValueError('Exact completed allfit data/source/BN/checkpoint controls required')
            for video in ['BT-001','SN-015']:
                for field,e in m['evaluations'][video]['artifacts'].items():
                    if sha256(path/video/(field+'.npy'))!=e['sha256']:raise ValueError('Actual retrieval feature artifact drift')
            result['fits'][path.name]=dict(path=str(path),manifest_sha256=sha256(path/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256'],evidence=m);models[mode].append(m)
    for seed in c['seed_plan']:
        expected=[e['sampled_input_sha256'] for e in models['jpm'][seed]['epochs']]
        if any([e['sampled_input_sha256'] for e in models[mode][seed]['epochs']]!=expected for mode in models):raise ValueError('Actual sampled/augmentation inputs differ')
        if models['jpm'][seed]['trainable_parameters']!=models['cls_capacity'][seed]['trainable_parameters']:raise ValueError('Active capacity differs')
    for video in ['BT-001','SN-015']:
        result['summary'][video]={}
        for mode,rows in models.items():
            out=result['summary'][video][mode]={}
            for view in ['gt','detector']:
                out[view]={}
                for gap in ['1','25','100']:
                    vals=[m['evaluations'][video]['retrieval'][view][gap] for m in rows];support=[{k:v[k] for k in ['queries','supported_queries','missing_target']} for v in vals]
                    if any(v!=support[0] for v in support):raise ValueError('Three-seed support differs')
                    out[view][gap]=dict(support=support[0],unknown_selected=[v['unknown_selected'] for v in vals],**{k:summary([v[k] for v in vals]) for k in ['rank1_percent','rank5_percent','mAP_percent']})
    for video in ['BT-001','SN-015']:
        for view in ['gt','detector']:
            for gap in ['1','25','100']:
                if len({json.dumps(result['summary'][video][m][view][gap]['support'],sort_keys=True) for m in models})!=1:raise ValueError('Sameview/gap query support differs acrossmodes')
    restart=a.root/'transreid_dino_jpm_exact_restart_v1.json';rr=json.loads(restart.read_text())
    if rr['status']!='EXACT_RESTART_STATE_VERIFIED' or rr['differences']:raise ValueError('Actual freshprocessrestart notexact')
    result.update(unique_fits=9,actual_restart=dict(path=str(restart),sha256=sha256(restart),proof=rr),same_sampled_inputs_and_active_capacity_verified=True)
    write_json(a.output,result);print(json.dumps(dict(status=result['status'],unique_fits=9)))

if __name__=='__main__':main()
