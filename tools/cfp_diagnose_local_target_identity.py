#!/usr/bin/env python3
"""Inner-only teacher identity/easy-mean baseline, no new training or outer read."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cfp.local_dense_context_identity import ContextIdentityStudent
from cfp.local_dense_jepa import DenseLocalTarget
from cfp.local_dense_data import LocalSceneBank
from cfp.local_dense_evaluation import evaluate_local_scenes,scene_query_plan
from cfp.local_target_diagnosis import TeacherLocalFIFO,FitMeanPrediction
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser()
    for name in ['training','features','index','legacy-features','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--video',required=True);a=ap.parse_args();m=json.loads((a.training/'manifest.json').read_text());g=m['guard'];c=m['config']
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.video!=g['inner'] or g['outer']==a.video or g['mode']!='full' or m['status']!='COMPLETE':raise ValueError('Fixed fullV2 inner-only diagnostic; rejectouterbefore anyweights/banks loaded')
    if sha256(a.training/'final.pt')!=m['checkpoint_sha256'] or any(sha256(p)!=v for p,v in g['source_sha256'].items()):raise ValueError('Exactregistered immutable model required')
    torch.set_num_threads(2);state=torch.load(a.training/'final.pt',map_location='cpu',weights_only=False);student=ContextIdentityStudent(len(m['classifier_keys']),g['mode'],g['local_count'],c['dimension'],c['attention_heads']).eval();student.load_state_dict(state['student']);teacher=DenseLocalTarget(student,c['ema_decay']).eval();teacher.load_state_dict(state['teacher']);assert not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
    bank=LocalSceneBank(a.features,g['local_count'],a.index,a.legacy_features)
    if sha256(a.features/'manifest.json')!=g['feature_manifest_sha256']:raise ValueError('Exactcropfeature source required')
    a.output.mkdir();total=torch.zeros(g['local_count'],c['dimension'],dtype=torch.float64);support=torch.zeros(g['local_count'],dtype=torch.float64);fit_observations={}
    with torch.inference_mode():
        for video in g['fit']:
            ids=[i for i,r in enumerate(bank.videos[video]['records']) if r['view']=='gt'];fit_observations[video]=len(ids)
            for start in range(0,len(ids),128):
                local,fraction,_=bank.current_rows(video,ids[start:start+128]);target=teacher(local,fraction);tokens=F.normalize(target['tokens'],dim=-1);weights=fraction*(target['valid']);total+=(tokens.double()*weights[:,:,None]).sum(0);support+=weights.double().sum(0)
    mean=F.normalize((total/support.clamp_min(1)[:,None]).float(),dim=-1);np.save(a.output/'fit_target_mean.npy',mean.numpy(),allow_pickle=False)
    report=dict(status='RUNNING',video=a.video,outer=g['outer'],outer_rejected_before_load=True,training_manifest_sha256=sha256(a.training/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256'],fit_observations=fit_observations,fit_target_mean_sha256=sha256(a.output/'fit_target_mean.npy'),teacher_updates=int(teacher.updates),teacher_stopped=True,identity={},prediction_difficulty={},sources=snapshot_sources(a.output/'sources',[__file__,'cfp/local_target_diagnosis.py']),scope='ReadonlyinnerBT001 diagnostics ofsame completedcontextV2 model. Teacherforward diagnosticofflineonly, neverdeployed. Constanttargetmean builtsolelyfromfourfitGTcrop observations, noinner/outermean fitting. Currenttargetfully excludedfrompredictor; querymask selects loss-only, not visiblecontext. Supports/unknowns separate, not nativeMOT/newloss/novelty.')
    write_json(a.output/'manifest.json',report)
    for name,wrapper in [('teacher_local_fifo',TeacherLocalFIFO(teacher)),('fit_mean_prediction',FitMeanPrediction(student,mean))]:
        measured=evaluate_local_scenes(wrapper,bank,a.video,c,a.output/name/'query_support');report['identity'][name]=measured
        for h in ['gt','detector']:
            for v in ['gt','detector']:
                for gap in ['1','25','100']:
                    assert measured['protocol'][h][v][gap]['query_support']['sha256']==m['evaluations'][a.video]['protocol'][h][v][gap]['query_support']['sha256']
        write_json(a.output/'manifest.json',report);print(name,measured['protocol']['gt']['detector']['25']['retrieval'],flush=True)
    plan=scene_query_plan(bank,a.video,c['query_cap_per_physical_identity'],c['maximum_history_age_frames'])
    with torch.inference_mode():
        for gap in c['minimum_past_gaps']:
            sums={name:[] for name in ['regional_predictor','context_identity','fit_mean','teacher_past_fifo']};counts=dict(query_rows=0,known_current_GT_targets=0,missing_current_GT=0,unavailable_history=0,observed_target_tokens=0)
            for frame,identities in plan.items():
                scene=bank.scene(a.video,frame,identities,'gt','detector',c['history_length'],c['maximum_history_age_frames'],minimum_gap=gap);inp=scene['inputs'];ctx=student.history_context(inp['history'],inp['history_fraction'],inp['history_global'],inp['times'],inp['history_valid'],frame)
                # Predictions made before loading/encoding any currentGTtargets.
                predicted=student.predict(ctx);context=F.normalize(ctx['pooled'],dim=-1);constant=mean[None].expand(len(identities),-1,-1)
                n,l,k,d=inp['history'].shape;past=teacher(inp['history'].reshape(-1,k,d),(inp['history_fraction']*inp['history_valid'][:,:,None]).reshape(-1,k));p=F.normalize(past['tokens'],dim=-1).reshape(n,l,k,d);pvalid=past['valid'].reshape(n,l,k);fifo=F.normalize((p*pvalid[:,:,:,None]).sum(1),dim=-1)
                rows=[];targets=[];counts['query_rows']+=len(identities)
                for j,identity in enumerate(identities):
                    current=bank.videos[a.video]['identities']['gt'].get(identity,{}).get(frame,[])
                    if len(current)!=1:counts['missing_current_GT']+=1;continue
                    counts['known_current_GT_targets']+=1
                    if not ctx['available'][j]:counts['unavailable_history']+=1;continue
                    rows.append(j);targets.append(current[0])
                if not rows:continue
                local,fraction,_=bank.current_rows(a.video,targets);target=teacher(local,fraction);valid=target['valid'];truth=F.normalize(target['tokens'],dim=-1);counts['observed_target_tokens']+=int(valid.sum())
                for name,values in [('regional_predictor',predicted),('context_identity',context),('fit_mean',constant),('teacher_past_fifo',fifo)]:
                    loss=1-(F.normalize(values[rows],dim=-1)*truth).sum(-1);sums[name].extend(loss[valid].tolist())
            report['prediction_difficulty'][str(gap)]=dict(counts=counts,losses={name:dict(tokens=len(values),mean=float(np.mean(values)) if values else None) for name,values in sums.items()},mask_semantics='Allobserved currentGTlocal targets heldoutfrompredictioninputs; masks onlyselect lossqueries andcannotvaryvisiblecontext inthisforecast implementation')
            write_json(a.output/'manifest.json',report);print('gap',gap,report['prediction_difficulty'][str(gap)],flush=True)
    report['status']='COMPLETE_INNER_ONLY_TEACHER_IDENTITY_AND_TARGET_DIFFICULTY_DIAGNOSIS';write_json(a.output/'manifest.json',report)

if __name__=='__main__':main()
