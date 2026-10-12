#!/usr/bin/env python3
"""Training-only target information and cue-stratified recovery diagnosis."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cfp.fish_jepa import FishJEPAStudent
from cfp.jepa_training_data import teacher_index_maps, make_jepa_batch
from cfp.recovery_training_data import load_recovery_data
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--queries',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Immutable diagnostic')
    torch.set_num_threads(2)
    metadata=json.loads((args.checkpoint.parent/'manifest.json').read_text())
    if metadata['status']!='COMPLETE' or sha256(args.checkpoint)!=metadata['checkpoint_sha256']:raise ValueError('Completed model required')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    spec=metadata['spec']
    if spec.get('ris'):raise ValueError('This target diagnostic is for independent JEPA')
    model=FishJEPAStudent(spec.get('architecture','gru'),spec.get('latent',128)).eval()
    model.load_state_dict(saved['model'])
    teacher=FishJEPAStudent(spec.get('architecture','gru'),spec.get('latent',128)).token_encoder.eval()
    teacher.load_state_dict(saved['teacher'])
    rows,features,labels,data=load_recovery_data(args.queries,metadata['guard']['seed'])
    rows=[r for r in rows if r['sequence']=='SN-015']
    maps=teacher_index_maps(args.queries,labels)
    points=[];strata={};target_counts={'queries':0,'future_changed':0,'future_raw_rank1':0,'teacher_current_rank1':0}
    with torch.inference_mode():
        for start in range(0,len(rows),128):
            chunk=rows[start:start+128]
            batch=make_jepa_batch(chunk,features,labels,maps,'cpu',np.random.default_rng(9000),view='actual',future_gap=5)
            past,times,valid,query,candidates,cv,pos,correct,known,future,*_=batch
            prototype,_=model(past,times,valid,query)
            actual=torch.einsum('bd,bkd->bk',prototype,F.normalize(candidates,dim=-1)).masked_fill(~cv,-2).argmax(1)==pos
            mean=F.normalize((F.normalize(past,dim=-1)*valid[...,None]).sum(1),dim=-1)
            old=torch.einsum('bd,bkd->bk',mean,F.normalize(candidates,dim=-1)).masked_fill(~cv,-2).argmax(1)==pos
            current=candidates[torch.arange(len(chunk)),pos]
            encoded=teacher(current);points.append(encoded.numpy())
            latent_score=torch.einsum('bd,bkd->bk',encoded,teacher(candidates)).masked_fill(~cv,-2)
            future_score=torch.einsum('bd,bkd->bk',F.normalize(future,dim=-1),F.normalize(candidates,dim=-1)).masked_fill(~cv,-2)
            target_counts['queries']+=len(chunk)
            target_counts['future_changed']+=int((future!=current).any(1).sum())
            target_counts['future_raw_rank1']+=int((future_score.argmax(1)==pos).sum())
            target_counts['teacher_current_rank1']+=int((latent_score.argmax(1)==pos).sum())
            for i in range(len(chunk)):
                clues=int((correct[i]&valid[i]).sum());unknown=int((~known[i]&valid[i]).sum())
                key=f'known_correct_{clues}_unknown_{unknown}'
                entry=strata.setdefault(key,dict(queries=0,model_correct=0,mean_correct=0,N01=0,N10=0))
                entry['queries']+=1;entry['model_correct']+=int(actual[i]);entry['mean_correct']+=int(old[i])
                entry['N01']+=int(actual[i]&~old[i]);entry['N10']+=int(~actual[i]&old[i])
    points=np.concatenate(points)
    write_json(args.output,dict(status='COMPLETE',checkpoint_sha256=sha256(args.checkpoint),query_manifest_sha256=sha256(args.queries/'manifest.json'),
      source_snapshots=snapshot_sources(args.output.parent/(args.output.stem+'_sources'),[__file__]),target_counts=target_counts,
      teacher_variance_sum=float(points.var(0).sum()),teacher_mean_norm=float(np.linalg.norm(points.mean(0))),strata=strata,
      scope='Training calibration SN015 only. Current teacher self-candidate rank is a coordinate consistency sanity check, not forecasting. Future5..30 targets appear only in this offline diagnostic. Unknown past labels are not wrong clues.'))


if __name__=='__main__':main()
