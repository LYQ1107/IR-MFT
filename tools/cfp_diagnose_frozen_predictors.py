#!/usr/bin/env python3
"""Compare original strong predictors on exactly the recovery calibration queries."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cfp.predictive import LatentIdentityPredictor
from cfp.recovery_training_data import load_recovery_data
from cfp.jepa_training_data import teacher_index_maps,make_jepa_batch
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Immutable baseline diagnosis')
    torch.set_num_threads(2)
    root=Path('/data3/liuyeqiang/IR-MFT_assets/cfp');query=root/'phase5/recovery_train_queries_v1'
    record=dict(status='RUNNING',query_manifest_sha256=sha256(query/'manifest.json'),models={},source_snapshots=snapshot_sources(args.output.parent/(args.output.stem+'_sources'),[__file__,'cfp/predictive.py','cfp/jepa_training_data.py','cfp/recovery_training_data.py']),scope='Offline train-calibration SN015, originalcompleted predictors, no newtraining. Equalvisible stress packs remaining strictlypast tokens for ordinarymodels; maskedJEPA keeps explicitmasktokens. Queries/currentcandidates/visiblefeatures identical; no onlineGT orfuturecontexts.')
    for seed in [0,1,2]:
        rows,features,labels,data=load_recovery_data(query,seed);rows=[r for r in rows if r['sequence']=='SN-015'];maps=teacher_index_maps(query,labels)
        for architecture,objective in [('gru','association'),('gru','predictive'),('transformer','association')]:
            checkpoint=root/'phase3'/f'predictor_v3_{architecture}_{objective}_anchor_seed{seed}'/'epoch8.pt'
            meta=json.loads(checkpoint.with_name('manifest.json').read_text())
            if meta['status']!='COMPLETE' or sha256(checkpoint)!=meta['checkpoint_sha256']:raise ValueError('Original model changed')
            fm=json.loads((Path(meta['features_root'])/'manifest.json').read_text())
            if fm['rgb_encoder']['checkpoint_sha256']!=data['rgb_encoder']['checkpoint_sha256']:raise ValueError('Baseline coordinate mismatch')
            model=LatentIdentityPredictor(512,architecture,time_encoding=meta.get('time_encoding','linear_clipped')).eval()
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
            results={}
            for view in ['clean','actual','actual_corrupted']:
                rng=np.random.default_rng(9000+seed);counts=dict(queries=0,model_correct=0,visible_mean_correct=0,N01=0,N10=0)
                with torch.inference_mode():
                    for start in range(0,len(rows),128):
                        stress=view=='actual_corrupted'
                        batch=make_jepa_batch(rows[start:start+128],features,labels,maps,'cpu',rng,view='actual' if stress else view,masking=stress,corruption=stress)
                        past,times,valid,query_frame,candidates,cv,pos,*_=batch;visible=valid&~batch[-1]
                        compact=torch.zeros_like(past);ct=torch.zeros_like(times);cm=torch.zeros_like(valid)
                        for i in range(len(past)):
                            n=int(visible[i].sum());compact[i,:n]=past[i,visible[i]];ct[i,:n]=times[i,visible[i]];cm[i,:n]=True
                        prototype,_=model(compact,ct,cm,query_frame)
                        reference=F.normalize((F.normalize(past,dim=-1)*visible[...,None]).sum(1),dim=-1)
                        def rank(p):return torch.einsum('bd,bkd->bk',p,F.normalize(candidates,dim=-1)).masked_fill(~cv,-2).argmax(1)==pos
                        good,old=rank(prototype),rank(reference)
                        counts['queries']+=len(past);counts['model_correct']+=int(good.sum());counts['visible_mean_correct']+=int(old.sum());counts['N01']+=int((good&~old).sum());counts['N10']+=int((~good&old).sum())
                counts.update(rank1=counts['model_correct']/counts['queries'],visible_mean_rank1=counts['visible_mean_correct']/counts['queries'],net_corrections=counts['N01']-counts['N10']);results[view]=counts
            name=checkpoint.parent.name;record['models'][name]=dict(checkpoint_sha256=sha256(checkpoint),manifest_sha256=sha256(checkpoint.with_name('manifest.json')),diagnostics=results);write_json(args.output,record);print(name,results,flush=True)
    record['status']='COMPLETE';write_json(args.output,record)


if __name__=='__main__':main()
