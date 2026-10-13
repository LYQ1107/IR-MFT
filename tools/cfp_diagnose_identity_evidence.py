#!/usr/bin/env python3
"""Explain inherited pure-retrieval versus geometry-blended row decisions.

Offline fixed actual states only; no GT history repair, no new weights, no
Hungarian or endogenous MOT claim. This is the first A0 mechanism experiment,
not the full future GT/detector-crop visual encoder evaluation platform.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from cfp.candidate_training_data import load_candidate_training, candidate_batch
from cfp.identity_evidence import query_retrieval, pair_discrimination
from cfp.predictive import LatentIdentityPredictor, bounded_identity_prototype
from cfp.provenance import sha256, snapshot_sources, write_json


def summarize(records):
    present = [r for r in records if r['target_present']]
    margins = [r['hard_negative_margin'] for r in present if r['hard_negative_margin'] is not None]
    return dict(queries=len(records), known_positive_queries=len(present),
                target_absent_queries=len(records)-len(present),
                rank1_percent=100*np.mean([r['rank1'] for r in present]) if present else None,
                rank5_percent=100*np.mean([r['rank5'] for r in present]) if present else None,
                mAP_percent=100*np.mean([r['average_precision'] for r in present]) if present else None,
                known_only_rank1_percent=100*np.mean([r['known_only_rank1'] for r in present]) if present else None,
                unknown_selected=sum(r['unknown_selected'] for r in present),
                hard_negative_accuracy_percent=100*np.mean(np.asarray(margins)>0) if margins else None,
                hard_negative_margin_quantiles=np.quantile(margins,[.1,.5,.9]).tolist() if margins else None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--queries',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Immutable diagnostic output required')
    if shutil.disk_usage(args.output.parent).free<40*2**30: raise OSError('40GiB reserve')
    rows,banks,meta=load_candidate_training(args.queries,0)
    checkpoint=Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase3/predictor_v3_gru_association_anchor_seed0/epoch8.pt')
    cm=json.loads(checkpoint.with_name('manifest.json').read_text())
    if sha256(checkpoint)!=cm['checkpoint_sha256']:raise ValueError('Frozen GRU changed')
    model=LatentIdentityPredictor().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
    torch.set_num_threads(2)
    report=dict(status='RUNNING',seed=0,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                query_manifest_sha256=sha256(args.queries/'manifest.json'),checkpoint_sha256=sha256(checkpoint),
                protocol_sha256=sha256('configs/cfp/protocol_v2.json'),videos={},
                scope='Six training-video fixed actual accepted-state pure-rank and native-geometry row controls. Old RGB/GRU exposure disclosed. '
                      'Unknown labels remain in inference pools but are excluded from known-negative AUC. No frame-level assignment or online MOT gain claimed. '
                      'Geometry eligibility is appearance blending, not a hard Hungarian gate. Density/overlap/gap are observable proxies, not annotated occlusion.',
                source_snapshots=snapshot_sources(args.output.parent/(args.output.stem+'_sources'),
                    [__file__,'cfp/identity_evidence.py','cfp/candidate_training_data.py','cfp/predictive.py',
                     'cfp/predictive_association.py','cfp/lifecycle_crossover.py']))
    write_json(args.output,report)
    with torch.inference_mode():
        for sequence in meta['videos']:
            selected=[r for r in rows if r['sequence']==sequence]
            retrieval=defaultdict(list); groups=defaultdict(lambda:defaultdict(list)); pairs=defaultdict(lambda:[[],[]])
            audit=dict(prototype_similarity_max_error=0.,baseline_cost_max_error=0.,
                       normalized_feature_norm_min=float(np.linalg.norm(banks[sequence]['features'],axis=1).min()),
                       normalized_feature_norm_max=float(np.linalg.norm(banks[sequence]['features'],axis=1).max()),
                       positive_inside_appearance_region=0,known_positive_queries=0,
                       assigned_known_correct=0,assigned_known_wrong=0,assigned_unknown=0,unmatched=0,
                       gru_last_cosines=[],gru_mean_cosines=[],history_purity=[])
            with np.load(args.queries/(sequence+'.npz'),allow_pickle=False) as archive:
                geometry=archive['geometry'].copy();offsets=archive['candidate_offsets'].copy()
            for low in range(0,len(selected),128):
                chunk=selected[low:low+128]
                past,times,valid,frame,candidates,*_=candidate_batch(chunk,banks,'cpu')
                normalized=F.normalize(candidates,dim=-1)
                prototypes={}
                for name,mode in [('last','last'),('fifo_mean','fifo'),('bounded_ema','ema')]:
                    prototypes[name]=bounded_identity_prototype(past,times,valid,frame,mode=mode)[0]
                prototypes['gru'],available=model(past,times,valid,frame)
                scores={name:torch.einsum('bd,bmd->bm',p,normalized).numpy() for name,p in prototypes.items()}
                for local,row in enumerate(chunk):
                    j=low+local;n=len(row['candidates']);lo,hi=offsets[j:j+2]
                    raw=geometry[lo:hi];confidence=row['evidence'][:,14]
                    fused_geometry=1-(1-raw)*confidence
                    eligible=raw<=.9
                    labels=row['edge_labels']
                    score_row={k:v[local,:n] for k,v in scores.items()}
                    score_row['geometry_only']=-fused_geometry
                    score_row['original_row_cost']=-row['evidence'][:,19]
                    for name in ('last','fifo_mean','bounded_ema','gru'):
                        corrected=fused_geometry.copy()
                        corrected[eligible]=.7*fused_geometry[eligible]+.3*np.clip((1-score_row[name][eligible])/2,0,1)
                        score_row[name+'_native_blend']=-corrected
                    if n:
                        audit['prototype_similarity_max_error']=max(audit['prototype_similarity_max_error'],float(np.max(np.abs(scores['gru'][local,:n]-row['evidence'][:,0]))))
                        audit['baseline_cost_max_error']=max(audit['baseline_cost_max_error'],float(np.max(np.abs(score_row['gru_native_blend']-score_row['original_row_cost']))))
                    positive=labels==0
                    audit['known_positive_queries']+=int(positive.any())
                    audit['positive_inside_appearance_region']+=int(np.any(positive&eligible))
                    a=row['assigned']
                    category='unmatched' if a<0 else ('assigned_unknown' if labels[a]<0 else 'assigned_known_correct' if labels[a]==0 else 'assigned_known_wrong')
                    audit[category]+=1
                    length=int(valid[local].sum())
                    if length:
                        audit['gru_last_cosines'].append(float((prototypes['gru'][local]*prototypes['last'][local]).sum()))
                        audit['gru_mean_cosines'].append(float((prototypes['gru'][local]*prototypes['fifo_mean'][local]).sum()))
                        audit['history_purity'].append(row['correct_history']/length)
                    gap=row['frame']-int(row['times'][length-1]) if length else None
                    group_labels=[('density','1-5' if n<=5 else '6-15' if n<=15 else '16+'),
                                  ('gap','empty' if gap is None else '1' if gap==1 else '2-15' if gap<=15 else '16+'),
                                  ('overlap_proxy','high' if n and row['evidence'][:,18].max()>=.5 else 'low'),
                                  ('retained_history_support','empty' if not length else 'all_verified_correct'
                                   if row['correct_history']==length else 'some_verified_correct' if row['correct_history']>0
                                   else 'no_verified_correct_unknown_possible')]
                    for name,values in score_row.items():
                        q=query_retrieval(values,labels);retrieval[name].append(q)
                        for axis,label in group_labels:groups[axis+'/'+label][name].append(q)
                        if name in ('last','fifo_mean','bounded_ema','gru'):
                            pairs[name][0].extend(values[labels==0].tolist());pairs[name][1].extend(values[labels==1].tolist())
                            q_eligible=query_retrieval(values[eligible],labels[eligible])
                            retrieval[name+'_eligible_pool'].append(q_eligible)
            for name in ('gru_last_cosines','gru_mean_cosines','history_purity'):
                values=audit.pop(name);audit[name+'_summary']=dict(mean=float(np.mean(values)),quantiles=np.quantile(values,[.1,.5,.9]).tolist()) if values else None
            report['videos'][sequence]=dict(
                retrieval={k:summarize(v) for k,v in retrieval.items()},
                discrimination={k:pair_discrimination(*v) for k,v in pairs.items()},
                by_observable_proxy={label:{k:summarize(v) for k,v in value.items()} for label,value in groups.items()},
                input_and_cost_audit=audit)
            write_json(args.output,report)
            print(sequence,{k:round(v['rank1_percent'],4) if v['rank1_percent'] is not None else None for k,v in report['videos'][sequence]['retrieval'].items()},flush=True)
    report.update(status='COMPLETE',storage_free_gib=shutil.disk_usage(args.output.parent).free/2**30)
    write_json(args.output,report)


if __name__=='__main__':main()
