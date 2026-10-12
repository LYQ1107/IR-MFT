#!/usr/bin/env python3
"""Equal-visible-history stress comparisons; immutable legacy logs retained."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cfp.recovery_training_data import load_recovery_data
from cfp.jepa_training_data import teacher_index_maps, make_jepa_batch
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.jepa_runtime import load_jepa_student


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse',type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous stress diagnostic')
    plan=json.loads(args.plan.read_text())
    torch.set_num_threads(2)
    record=dict(status='RUNNING',plan_sha256=sha256(args.plan),models={},source_snapshots=snapshot_sources(args.output.parent/(args.output.stem+'_sources'),[__file__,args.plan]),scope='Train-calibration SN015 offline identity labels only. Model and visible-mean read exactlysame unmaskedpast values. Fullmean is extra-information comparator. N01/N10 are fixed-query rank changes, not tracking IDs.')
    previous=json.loads(args.reuse.read_text()) if args.reuse else {'models':{}}
    if args.reuse and previous['status']!='COMPLETE':raise ValueError('Only complete stress diagnostics reusable')
    if args.reuse:record['reused_source_sha256']=sha256(args.reuse)
    for job in plan['jobs']:
        path=Path(job['checkpoint']);metadata=json.loads(path.with_name('manifest.json').read_text())
        old=previous['models'].get(path.parent.name)
        if old and old['checkpoint_sha256']==sha256(path) and old['manifest_sha256']==sha256(path.with_name('manifest.json')):
            record['models'][path.parent.name]=old;write_json(args.output,record);print('Verified reuse',path.parent.name,flush=True);continue
        rows,features,labels,data=load_recovery_data(job['queries'],job['seed'])
        rows=[r for r in rows if r['sequence']=='SN-015'];maps=teacher_index_maps(job['queries'],labels)
        model,_=load_jepa_student(path,job['seed'],data['rgb_encoder']['checkpoint_sha256'])
        rng=np.random.default_rng(9000+job['seed']);counts=dict(queries=0,model_correct=0,visible_mean_correct=0,full_mean_correct=0,N01=0,N10=0);strata={}
        with torch.inference_mode():
            for start in range(0,len(rows),128):
                batch=make_jepa_batch(rows[start:start+128],features,labels,maps,'cpu',rng,view='actual',masking=True,corruption=True,future_gap=metadata['spec'].get('future_gap',0))
                past,times,valid,query,candidates,cv,pos,correct,known,*_=batch;mask=batch[-1]
                visible=valid&~mask
                prototype,_,_,_=model.training_details(past,times,valid,query,mask)
                def ranks(proto):return torch.einsum('bd,bkd->bk',proto,F.normalize(candidates,dim=-1)).masked_fill(~cv,-2).argmax(1)==pos
                good=ranks(prototype)
                visible_good=ranks(F.normalize((F.normalize(past,dim=-1)*visible[...,None]).sum(1),dim=-1))
                full_good=ranks(F.normalize((F.normalize(past,dim=-1)*valid[...,None]).sum(1),dim=-1))
                for i in range(len(past)):
                    key=f'masked_{int(mask[i].sum())}_known_correct_visible_{int((correct[i]&visible[i]).sum())}'
                    entry=strata.setdefault(key,dict(queries=0,model_correct=0,visible_mean_correct=0,full_mean_correct=0,N01=0,N10=0))
                    for bucket in [counts,entry]:
                        bucket['queries']+=1;bucket['model_correct']+=int(good[i]);bucket['visible_mean_correct']+=int(visible_good[i]);bucket['full_mean_correct']+=int(full_good[i]);bucket['N01']+=int(good[i]&~visible_good[i]);bucket['N10']+=int(~good[i]&visible_good[i])
        counts['model_rank1']=counts['model_correct']/counts['queries'];counts['visible_mean_rank1']=counts['visible_mean_correct']/counts['queries'];counts['net_corrections']=counts['N01']-counts['N10']
        record['models'][path.parent.name]=dict(checkpoint_sha256=sha256(path),manifest_sha256=sha256(path.with_name('manifest.json')),counts=counts,strata=strata)
        write_json(args.output,record);print(path.parent.name,counts,flush=True)
    record['status']='COMPLETE';write_json(args.output,record)


if __name__=='__main__':main()
