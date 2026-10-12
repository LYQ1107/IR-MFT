#!/usr/bin/env python3
"""Fit OOF-supervised utility or matched ordinary gates; dev never fits a parameter."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.stats import rankdata
import torch
from torch.nn import functional as F

from cfp.provenance import sha256, write_json, snapshot_sources
from cfp.utility import UtilityMLP, EVIDENCE_FIELDS
from cfp.utility_supervision import ranking_sign_loss,decision_strata


def candidate_batch(rows, config):
    width = max(len(r['rgb']) for r in rows) + 1
    rgb = np.zeros((len(rows), width), np.float32);ir = rgb.copy()
    mask = np.zeros_like(rgb, bool);targets = []
    for i, row in enumerate(rows):
        n = len(row['rgb']);rgb[i,:n]=row['rgb'];ir[i,:n]=row['ir']
        rgb[i,n]=ir[i,n]=row.get('null_distance',config['null_distance']);mask[i,:n+1]=True;targets.append(row['target'])
    return torch.from_numpy(rgb),torch.from_numpy(ir),torch.from_numpy(mask),torch.tensor(targets,dtype=torch.long)


def utility_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    valid = truth != 0;positive = truth[valid] > 0;npos=int(positive.sum());nneg=int((~positive).sum())
    ranks=rankdata(prediction[valid])
    auc=float((ranks[positive].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else None
    return dict(mae=float(np.mean(np.abs(prediction-truth))),mse=float(np.mean((prediction-truth)**2)),
                sign_auc_nonzero=auc,truth_positive_fraction=float(np.mean(truth>0)),
                prediction_std=float(prediction.std()),truth_std=float(truth.std()),
                correlation=float(np.corrcoef(prediction,truth)[0,1]) if prediction.std()>1e-12 and truth.std()>1e-12 else None)


def load_rows(root, config_path, partition):
    config=json.loads(config_path.read_text())
    meta=json.loads((root/'manifest.json').read_text())
    if meta['status']!='COMPLETE' or meta['partition']!=partition or meta['config_sha256']!=sha256(config_path) or meta['evidence_fields']!=list(EVIDENCE_FIELDS):
        raise ValueError('Complete matching utility query inputs required')
    if partition=='train':
        for source in meta['teacher_sources']:
            path=Path(source['root'])/'manifest.json'
            teacher=json.loads(path.read_text())
            if sha256(path)!=source['manifest_sha256'] or teacher['encoder_role']!='oof_teacher':
                raise ValueError('Utility fitting requires unchanged out-of-fold teachers')
    rows=[]
    for seq,report in meta['videos'].items():
        path=root/(seq+'.npz')
        if sha256(path)!=report['sha256']:
            raise ValueError('Supervised query data changed')
        with np.load(path) as archive:
            bank={key:archive[key] for key in archive.files}
        if config.get('candidate_context_version',1)==2 and 'null_distance' not in bank:
            raise ValueError('Complete-context supervision requires explicit per-row native NULL costs')
        for i in range(len(bank['utility'])):
            lo,hi=bank['offsets'][i:i+2]
            rows.append(dict(sequence=seq,evidence=bank['evidence'][i],utility=float(bank['utility'][i]),
                             target=int(bank['target'][i]),rgb=bank['rgb'][lo:hi],ir=bank['ir'][lo:hi],
                             null_distance=float(bank['null_distance'][i]) if 'null_distance' in bank else config['null_distance']))
    return rows,meta


def predict(model, rows, batch_size=1024):
    result=[]
    model.eval()
    with torch.inference_mode():
        for lo in range(0,len(rows),batch_size):
            result.extend(model(torch.from_numpy(np.stack([r['evidence'] for r in rows[lo:lo+batch_size]]))).numpy().tolist())
    return np.asarray(result,dtype=np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--queries',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--kind',choices=['cmue','ordinary','quality'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Resume only this incomplete gate run at its last complete epoch')
    args=p.parse_args()
    if args.output.exists() and not args.resume:raise FileExistsError('Gate training outputs are immutable')
    if args.resume and not (args.output/'last.pt').is_file():raise ValueError('Gate resume requires its own saved checkpoint')
    if args.resume and json.loads((args.output/'manifest.json').read_text())['status']=='COMPLETE':
        raise ValueError('Completed gate runs are immutable')
    config=json.loads(args.config.read_text())
    if config['state']!='FROZEN' or args.seed not in config['seed_plan']:raise ValueError('Undeclared gate run')
    rows,query_meta=load_rows(args.queries,args.config,'train')
    if query_meta['encoder_seed']!=args.seed:raise ValueError('Gate seed must match its entire OOF teacher/encoder pipeline')
    fit=[r for r in rows if r['sequence'] in config['fit_videos']]
    calibration=[r for r in rows if r['sequence'] in config['calibration_videos']]
    if not fit or not calibration or len(fit)+len(calibration)!=len(rows):raise ValueError('Invalid association-training fit/calibration split')
    videos=config['fit_videos'];positions=[np.array([i for i,r in enumerate(fit) if r['sequence']==v]) for v in videos]
    if any(not len(x) for x in positions):raise ValueError('Every fit video requires eligible supervised queries')
    strata=decision_strata(fit,positions) if config.get('decision_type_balanced_sampling') else None
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    model=UtilityMLP(config['mlp']['hidden'],quality_only=args.kind=='quality')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    args.output.mkdir(parents=True,exist_ok=args.resume)
    sources=snapshot_sources(args.output,[__file__,'cfp/utility.py','cfp/provenance.py','cfp/utility_supervision.py',args.config])
    manifest=dict(status='RUNNING',kind=args.kind,seed=args.seed,config=config,config_sha256=sha256(args.config),
                 protocol_sha256=query_meta['protocol_sha256'],query_manifest_sha256=sha256(args.queries/'manifest.json'),
                 fit_videos=videos,calibration_videos=config['calibration_videos'],fit_queries=len(fit),calibration_queries=len(calibration),
                 parameters=sum(p.numel() for p in model.parameters()),evidence_fields=list(EVIDENCE_FIELDS),
                 source_sha256=sha256(__file__),model_source_sha256=sha256('cfp/utility.py'),
                 source_snapshots=sources,
                 git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 training_device='CPU',checkpoint_selection='Fixed epoch8; only association-training SN-015 calibrates utility',epochs=[],
                 scope='No canonical dev fitting, no historical test data; calibrated estimator evidence is separate from actual online tracking gains.')
    first_epoch,prior_seconds=0,0.
    if args.resume:
        previous=json.loads((args.output/'manifest.json').read_text())
        if previous['status']=='COMPLETE':raise ValueError('Completed gate runs are immutable')
        checkpoint=torch.load(args.output/'last.pt',map_location='cpu')
        for key in ['kind','seed','config_sha256','protocol_sha256','query_manifest_sha256','source_sha256','model_source_sha256']:
            if checkpoint['metadata'][key]!=manifest[key]:raise ValueError(f'Gate resume provenance changed: {key}')
        model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
        rng.bit_generator.state=checkpoint['rng_state']['sampling'];torch.set_rng_state(checkpoint['rng_state']['torch_cpu'])
        first_epoch=checkpoint['epoch'];prior_seconds=checkpoint['elapsed_seconds_before_checkpoint']
        manifest=checkpoint['metadata']
        manifest.setdefault('resumes',[]).append(dict(checkpoint_sha256=sha256(args.output/'last.pt'),completed_epoch=first_epoch))
    write_json(args.output/'manifest.json',manifest)
    start=time.perf_counter()
    steps=math.ceil(len(fit)/config['batch_size'])
    for epoch in range(first_epoch,config['epochs']):
        model.train();losses=[]
        for _ in range(steps):
            assigned=rng.integers(0,len(videos),config['batch_size']);chosen=np.empty(config['batch_size'],dtype=np.int64)
            for v,indices in enumerate(positions):
                slots=np.flatnonzero(assigned==v)
                if strata is None:chosen[slots]=rng.choice(indices,len(slots),replace=True)
                else:
                    types=rng.integers(0,len(strata[v]),len(slots))
                    for kind,group in enumerate(strata[v]):
                        subset=slots[types==kind];chosen[subset]=rng.choice(group,len(subset),replace=True)
            batch=[fit[i] for i in chosen]
            output=model(torch.from_numpy(np.stack([r['evidence'] for r in batch])))
            if args.kind=='cmue':
                truth=torch.tensor([r['utility'] for r in batch],dtype=torch.float32)
                if config.get('utility_target')=='candidate_ranking_only':loss=ranking_sign_loss(output,truth,**config['ranking_sign_loss'])
                else:loss=F.huber_loss(output,truth,delta=config['huber_delta'])
            else:
                rgb,ir,mask,target=candidate_batch(batch,config)
                alpha=output.sigmoid().unsqueeze(1)*config['maximum_ir_weight']
                mixed=(1-alpha)*rgb+alpha*ir
                logits=(-mixed/config['teacher_temperature']).masked_fill(~mask,-torch.inf)
                loss=F.cross_entropy(logits,target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite gate loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()))
        item=dict(epoch=epoch+1,loss=float(np.mean(losses)));manifest['epochs'].append(item)
        checkpoint=dict(epoch=epoch+1,kind=args.kind,seed=args.seed,config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),
                        rng_state=dict(sampling=rng.bit_generator.state,torch_cpu=torch.get_rng_state()),metadata=manifest,
                        elapsed_seconds_before_checkpoint=prior_seconds+time.perf_counter()-start)
        torch.save(checkpoint,args.output/'last.pt');write_json(args.output/'manifest.json',manifest)
        print(json.dumps(item),flush=True)
    raw=predict(model,calibration);truth=np.array([r['utility'] for r in calibration])
    if args.kind=='cmue':
        variance=float(np.var(raw));slope=max(0.,float(np.mean((raw-raw.mean())*(truth-truth.mean()))/variance)) if variance>1e-12 else 0.
        intercept=float(truth.mean()-slope*raw.mean());calibrated=slope*raw+intercept
        manifest['calibration']=dict(slope=slope,intercept=intercept,method='Nonnegative-slope least-squares affine fit on association-training calibration video only')
        manifest['calibration_metrics']=utility_metrics(calibrated,truth)
        manifest['raw_calibration_metrics']=utility_metrics(raw,truth)
        manifest['threshold_diagnosis']={str(th):dict(accepted_fraction=float(np.mean(calibrated>th)),true_utility_mean_when_accepted=float(truth[calibrated>th].mean()) if np.any(calibrated>th) else None) for th in config['utility_threshold_candidates']}
    else:
        manifest['calibration']=None
        weights=config['maximum_ir_weight']/(1+np.exp(-raw))
        manifest['calibration_gate_weight_mean_std']=[float(weights.mean()),float(weights.std())]
    manifest.update(status='COMPLETE',total_seconds=prior_seconds+time.perf_counter()-start)
    checkpoint.update(metadata=manifest,calibration=manifest['calibration']);torch.save(checkpoint,args.output/'epoch8.pt')
    manifest['checkpoint_sha256']=sha256(args.output/'epoch8.pt');write_json(args.output/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['status','kind','total_seconds','calibration']}),flush=True)


if __name__=='__main__':main()
