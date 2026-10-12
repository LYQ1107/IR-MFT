#!/usr/bin/env python3
"""Fit bounded latent predictors on frozen train queries; no canonical-dev fitting."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from cfp.predictive import LatentIdentityPredictor,bounded_identity_prototype,latent_identity_loss
from cfp.provenance import sha256,write_json,snapshot_sources
from cfp.representations import identity_view


def load_queries(root,config_path,query_config_path,partition,representation='rgb',second_features_root=None):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='COMPLETE' or manifest['partition']!=partition or manifest['config_sha256']!=sha256(config_path) or manifest['query_config_sha256']!=sha256(query_config_path):
        raise ValueError('Complete matching predictive query data required')
    features_root=Path(manifest['features_root']);fm=json.loads((features_root/'manifest.json').read_text())
    if sha256(features_root/'manifest.json')!=manifest['feature_manifest_sha256'] or fm['status']!='COMPLETE' or fm.get('encoder_role','main_baseline')!='main_baseline' or fm['embedding_stage']!='post_neck':
        raise ValueError('Stopped main encoder coordinate space changed')
    rows=[];features={}
    secondary=None
    if representation=='two_rgb':
        if second_features_root is None:raise ValueError('Matched two-RGB view requires its second existing bank')
        second_features_root=Path(second_features_root);secondary=json.loads((second_features_root/'manifest.json').read_text())
        if secondary['status']!='COMPLETE' or secondary['protocol_sha256']!=fm['protocol_sha256'] or secondary['detection_manifest_sha256']!=fm['detection_manifest_sha256'] or secondary['embedding_stage']!='post_neck' or secondary.get('encoder_role','main_baseline')!='main_baseline':raise ValueError('Second RGB bank does not share original proposals/protocol')
    elif second_features_root is not None:raise ValueError('Only two-RGB controls consume a second bank')
    for sequence,report in manifest['videos'].items():
        path=root/(sequence+'.npz');fp=features_root/(sequence+'.npz')
        if sha256(path)!=report['sha256'] or sha256(fp)!=report['features_sha256']:raise ValueError('Stopped query/feature arrays changed')
        with np.load(fp) as a:rgb=a['rgb'];ir=a['ir'];valid_rgb=a['valid_rgb'];valid_ir=a['valid_ir']
        if not np.array_equal(valid_rgb,valid_ir):raise ValueError('Modality ablation must preserve identical valid-proposal availability')
        second_rgb=None
        if secondary is not None:
            sp=second_features_root/(sequence+'.npz');prov=json.loads(sp.with_suffix('.provenance.json').read_text())
            if sha256(sp)!=prov['features_sha256'] or prov['detection_sha256']!=json.loads(fp.with_suffix('.provenance.json').read_text())['detection_sha256']:raise ValueError('Second RGB feature/proposal alignment changed')
            with np.load(sp) as a:second_rgb=a['rgb'];second_valid=a['valid_rgb']
            if not np.array_equal(valid_rgb,second_valid):raise ValueError('Second RGB capacity control changes proposal availability')
        features[sequence]=identity_view(rgb,ir,representation,second_rgb)
        with np.load(path) as a:bank={k:a[k] for k in a.files}
        count=len(bank['frame']);offsets=bank['candidate_offsets']
        if len(offsets)!=count+1 or offsets[0]!=0 or offsets[-1]!=len(bank['candidate_indices']) or np.any(np.diff(offsets)<2):raise ValueError('Missing distinct same-frame training negatives')
        for i in range(count):
            lo,hi=offsets[i:i+2];candidates=bank['candidate_indices'][lo:hi];positive=int(bank['positive_index'][i])
            if not 0<=positive<len(candidates) or np.any(candidates<0) or np.any(candidates>=len(features[sequence])):raise ValueError('Invalid stopped current target index')
            row=dict(sequence=sequence,frame=int(bank['frame'][i]),identity=int(bank['anchor_id'][i]),gap=int(bank['requested_gap'][i]),positive=positive,candidates=candidates,purity=float(bank['history_purity'][i]))
            for history in ['clean','actual']+(['anchor_recent'] if 'anchor_recent_history_indices' in bank else []):
                indices=bank[history+'_history_indices'][i];times=bank[history+'_history_frames'][i];valid=indices>=0;length=int(valid.sum())
                if not 0<length<=8 or not np.array_equal(valid,np.arange(len(indices))<length) or np.any(indices[valid]>=len(features[sequence])) or np.any(times[valid]<=0) or np.any(times[valid]>row['frame']-row['gap']) or np.any(np.diff(times[valid])<=0):
                    raise ValueError('Invalid bounded causal history')
                row[history+'_indices']=indices;row[history+'_frames']=times
            rows.append(row)
    manifest=dict(manifest,identity_representation=representation,secondary_features_root=str(second_features_root) if secondary else None,secondary_feature_manifest_sha256=sha256(second_features_root/'manifest.json') if secondary else None,secondary_rgb_checkpoint_sha256=secondary['rgb_encoder']['checkpoint_sha256'] if secondary else None)
    return rows,features,manifest


def make_batch(rows,features,history,device):
    if history not in ['clean','actual','anchor_recent']:raise ValueError('Unknown observational history')
    width=max(len(r['candidates']) for r in rows);dimension=next(iter(features.values())).shape[1];length=len(rows[0][history+'_indices'])
    past=np.zeros((len(rows),length,dimension),np.float32);candidates=np.zeros((len(rows),width,dimension),np.float32)
    candidate_valid=np.zeros((len(rows),width),bool);valid=np.zeros((len(rows),length),bool);times=[]
    for i,r in enumerate(rows):
        bank=features[r['sequence']];indices=r[history+'_indices'];valid[i]=indices>=0
        past[i,valid[i]]=bank[indices[valid[i]]];n=len(r['candidates']);candidates[i,:n]=bank[r['candidates']];candidate_valid[i,:n]=True;times.append(r[history+'_frames'])
    positive=np.array([r['positive'] for r in rows]);target=candidates[np.arange(len(rows)),positive].copy()
    values=(past,np.stack(times),valid,np.array([r['frame'] for r in rows]),target,candidates,positive,candidate_valid)
    return tuple(torch.from_numpy(v).to(device) for v in values)


def sampling_groups(rows,videos):
    groups=[]
    for video in videos:
        identities=sorted({r['identity'] for r in rows if r['sequence']==video})
        if not identities:raise ValueError('Every fit video needs eligible latent examples')
        groups.append([np.array([i for i,r in enumerate(rows) if r['sequence']==video and r['identity']==identity]) for identity in identities])
    return groups


def sample_positions(groups,rng,count):
    positions=[]
    for _ in range(count):
        video=groups[int(rng.integers(len(groups)))];identity=video[int(rng.integers(len(video)))];positions.append(int(rng.choice(identity)))
    return positions


def diagnose(model,rows,features,history,device,mode=None):
    vectors=[];target_cos=[];ce=[];correct=[]
    model.eval() if model is not None else None
    with torch.inference_mode():
        for lo in range(0,len(rows),128):
            batch=make_batch(rows[lo:lo+128],features,history,device);past,times,valid,frame,target,candidates,positive,mask=batch
            predicted,available=model(past,times,valid,frame) if model is not None else bounded_identity_prototype(past,times,valid,frame,mode=mode)
            if not available.all():raise ValueError('Query builder promised nonempty history')
            target=F.normalize(target,dim=-1);candidates=F.normalize(candidates,dim=-1)
            logits=torch.einsum('bd,bkd->bk',predicted,candidates).masked_fill(~mask,-torch.inf)/.1
            target_cos.extend((predicted*target).sum(1).cpu().tolist());ce.extend(F.cross_entropy(logits,positive,reduction='none').cpu().tolist());correct.extend((logits.argmax(1)==positive).cpu().tolist());vectors.append(predicted.cpu().numpy())
    vectors=np.concatenate(vectors)
    result=dict(queries=len(rows),true_target_cosine=float(np.mean(target_cos)),same_frame_ce=float(np.mean(ce)),same_frame_rank1_percent=100*float(np.mean(correct)),prediction_variance_sum=float(vectors.var(0).sum()),mean_vector_norm=float(np.linalg.norm(vectors.mean(0))))
    for gap in sorted({r['gap'] for r in rows}):
        selected=np.array([r['gap']==gap for r in rows]);result.setdefault('by_gap',{})[str(gap)]=dict(queries=int(selected.sum()),true_target_cosine=float(np.array(target_cos)[selected].mean()),same_frame_rank1_percent=100*float(np.array(correct)[selected].mean()))
    return result


def atomic_checkpoint(path,payload):
    temp=path.with_name(path.name+'.tmp');torch.save(payload,temp);os.replace(temp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--queries',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--query-config',type=Path,required=True);p.add_argument('--training-config',type=Path,required=True)
    p.add_argument('--architecture',choices=['gru','transformer'],required=True);p.add_argument('--objective',choices=['predictive','association'],required=True);p.add_argument('--history',choices=['clean','actual','anchor_recent'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--identity-representation',choices=['rgb','ir','rgb_ir','two_rgb'],default='rgb');p.add_argument('--second-rgb-features-root',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--gpu-memory-fraction',type=float,default=.1);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    config=json.loads(args.config.read_text());training=json.loads(args.training_config.read_text())
    if config['state']!='FROZEN' or training['state']!='FROZEN' or training['trial_sha256']!=sha256(args.config) or training['query_config_sha256']!=sha256(args.query_config) or args.seed not in config['seed_plan']:raise ValueError('Frozen matched predictive budget required')
    if args.device=='cuda' and (os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or torch.cuda.device_count()!=1):raise ValueError('This first predictor experiment reserves only physical GPU2')
    if not 0<args.gpu_memory_fraction<=1:raise ValueError('Valid per-process CUDA memory fraction required')
    if args.device=='cuda':torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    if args.output.exists() and not args.resume:raise FileExistsError('Predictor runs are immutable')
    if args.resume and (not (args.output/'last.pt').is_file() or json.loads((args.output/'manifest.json').read_text())['status']=='COMPLETE'):raise ValueError('Only incomplete matching epoch checkpoints may resume')
    if training.get('identity_representation','rgb')!=args.identity_representation:raise ValueError('Latent view differs from frozen training declaration')
    rows,features,query_meta=load_queries(args.queries,args.config,args.query_config,'train',args.identity_representation,args.second_rgb_features_root)
    if query_meta['encoder_seed']!=args.seed:raise ValueError('Predictor seed must match its frozen encoder pipeline')
    fit=[r for r in rows if r['sequence'] in config['fit_videos']];cal=[r for r in rows if r['sequence'] in config['calibration_videos']]
    if not fit or not cal or len(fit)+len(cal)!=len(rows):raise ValueError('Invalid whole-video temporal fitting partition')
    groups=sampling_groups(fit,config['fit_videos']);device=torch.device(args.device);torch.set_num_threads(2);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    time_encoding=training.get('time_encoding','linear_clipped')
    dimension=next(iter(features.values())).shape[1]
    model=LatentIdentityPredictor(dimension,args.architecture,time_encoding=time_encoding).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    args.output.mkdir(parents=True,exist_ok=args.resume)
    snapshots=snapshot_sources(args.output,[__file__,'cfp/predictive.py','cfp/provenance.py','cfp/representations.py',args.config,args.query_config,args.training_config])
    guard=dict(config_sha256=sha256(args.config),query_config_sha256=sha256(args.query_config),training_config_sha256=sha256(args.training_config),query_manifest_sha256=sha256(args.queries/'manifest.json'),source_snapshots=snapshots,architecture=args.architecture,objective=args.objective,history=args.history,seed=args.seed,device=args.device,torch=torch.__version__,cudnn=torch.backends.cudnn.version(),physical_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_memory_fraction=args.gpu_memory_fraction if args.device=='cuda' else None,time_encoding=time_encoding,identity_representation=args.identity_representation,representation_dimension=dimension,secondary_features_root=query_meta['secondary_features_root'],secondary_feature_manifest_sha256=query_meta['secondary_feature_manifest_sha256'],secondary_rgb_checkpoint_sha256=query_meta['secondary_rgb_checkpoint_sha256'])
    report=dict(status='RUNNING',config=config,training_config=training,**guard,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parameters=sum(v.numel() for v in model.parameters()),protocol_sha256=query_meta['protocol_sha256'],encoder_seed=query_meta['encoder_seed'],features_root=query_meta['features_root'],feature_manifest_sha256=query_meta['feature_manifest_sha256'],fit_queries=len(fit),calibration_queries=len(cal),fit_videos=config['fit_videos'],calibration_videos=config['calibration_videos'],epochs=[],checkpoint_selection='Fixed last epoch8; calibration reports only, no checkpoint/threshold selection',scope='RGB latent stopped-target training. Canonical dev does not fit any parameter; SN-015 query holdout was exposed to the main fish encoder. Clean-vs-actual are retrospective histories, not deployment oracle. Cache-training latency is not online latency.')
    start_epoch=0;elapsed=0.
    if args.resume:
        checkpoint=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint['guard']!=guard:raise ValueError('Resume changed data/source/config/runtime/device')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);rng.bit_generator.state=checkpoint['rng_numpy'];torch.set_rng_state(checkpoint['rng_torch']);
        if args.device=='cuda':torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
        report=checkpoint['report'];start_epoch=checkpoint['epoch'];elapsed=checkpoint['elapsed']
    write_json(args.output/'manifest.json',report);start=time.monotonic();steps=math.ceil(len(fit)/config['batch_size'])
    for epoch in range(start_epoch+1,config['epochs']+1):
        model.train();losses=[];predictive=[];association=[];epoch_start=time.monotonic()
        for step in range(steps):
            selected=[fit[i] for i in sample_positions(groups,rng,config['batch_size'])];batch=make_batch(selected,features,args.history,device)
            past,times,valid,frame,target,candidates,positive,mask=batch;predicted,available=model(past,times,valid,frame)
            if not available.all():raise ValueError('Missing sampled predictive history')
            loss,parts=latent_identity_loss(predicted,target,candidates,positive,mask,prediction_weight=1. if args.objective=='predictive' else 0.)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()));predictive.append(float(parts['prediction']));association.append(float(parts['association']))
        entry=dict(epoch=epoch,steps=steps,loss=float(np.mean(losses)),prediction_loss=float(np.mean(predictive)),association_loss=float(np.mean(association)),seconds=time.monotonic()-epoch_start);report['epochs'].append(entry)
        atomic_checkpoint(args.output/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,guard=guard,report=report,rng_numpy=rng.bit_generator.state,rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if args.device=='cuda' else [],elapsed=elapsed+time.monotonic()-start))
        write_json(args.output/'manifest.json',report);print(json.dumps(entry),flush=True)
    atomic_checkpoint(args.output/'epoch8.pt',dict(model=model.state_dict(),metadata=report,epoch=config['epochs']))
    diagnostics={}
    for history in ['clean','actual']+(['anchor_recent'] if args.history=='anchor_recent' else []):
        diagnostics[history]={'model':diagnose(model,cal,features,history,device)}
        for mode in ['last','ema','fifo']:diagnostics[history][mode]=diagnose(None,cal,features,history,device,mode)
    report.update(status='COMPLETE',checkpoint_sha256=sha256(args.output/'epoch8.pt'),calibration_diagnostics=diagnostics,total_seconds=elapsed+time.monotonic()-start,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
    write_json(args.output/'manifest.json',report);print('COMPLETE',args.architecture,args.objective,args.history,flush=True)


if __name__=='__main__':main()
