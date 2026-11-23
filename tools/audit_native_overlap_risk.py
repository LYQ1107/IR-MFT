#!/usr/bin/env python3
"""Cheap Train-only overlap-risk probes on existing frozen descriptors."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.audit_native_objectness import read
from tools.audit_correlation_identity_events import overlap


def targets(gt,frames,boxes,threshold):
    labels=np.full(len(boxes),-1,np.int8)
    for frame in np.unique(frames):
        indices=np.flatnonzero(frames==frame);rows=gt[(gt[:,0]==frame)&(gt[:,6]>0)&(gt[:,7]==1)]
        truth=rows[:,2:6].copy();truth[:,2:]+=truth[:,:2]
        if not len(truth):continue
        iou=overlap(boxes[indices],truth);matched=iou.argmax(1);selected=iou.max(1)>=.5
        lower=np.maximum(truth[:,None,:2],truth[None,:,:2]);upper=np.minimum(truth[:,None,2:],truth[None,:,2:])
        area=(truth[:,2:]-truth[:,:2]).prod(-1);risk=np.maximum(upper-lower,0).prod(-1)/np.maximum(np.minimum(area[:,None],area[None]),1.)
        np.fill_diagonal(risk,0);positive=risk.max(1)>=threshold
        labels[indices[selected]]=positive[matched[selected]]
    return labels


def geometry(boxes):
    size=np.maximum(boxes[:,2:]-boxes[:,:2],1.)
    return np.c_[np.log(size[:,0]),np.log(size[:,1]),np.log(size.prod(-1)),np.log(size[:,0]/size[:,1])].astype(np.float32)


def main():
    main=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(main['work_root'])
    config_path=ROOT/'configs/perceptual_overlap_risk_audit_20261123.json';config=json.loads(config_path.read_text())
    protocol=json.loads((ROOT/main['protocol']).read_text());out=work/'native_overlap_risk_audit';out.mkdir(exist_ok=True)
    if(out/'train_complete.json').exists():raise ValueError('Completed risk probe exists')
    labels=[];weights=[];references=[];sources={};indices={};counts={}
    for seq in protocol['train']:
        path=work/'native_candidate_index'/seq/'index.npz'
        with np.load(path)as z:frames=z['frame'];boxes=z['boxes']
        gt_path=Path(main['evaluation']['GT_root'])/'train'/seq/'gt/gt.txt';gt=np.loadtxt(gt_path,delimiter=',',ndmin=2)
        y=targets(gt,frames,boxes,config['threshold']);valid=y>=0;indices[seq]=valid;y=y[valid];classes=np.unique(y)
        weight=np.empty(len(y),np.float64)
        for value in classes:weight[y==value]=1/(len(protocol['train'])*len(classes)*int((y==value).sum()))
        reference_path=work/'native_candidate_features'/seq/'rgb/features.npz'
        references.append(np.c_[read(reference_path,'reference_post')[valid],geometry(boxes)[valid]])
        labels.append(y);weights.append(weight);counts[seq]=np.bincount(y,minlength=2).tolist()
        sources[seq]=dict(index_sha256=sha256(path),GT_sha256=sha256(gt_path),reference_sha256=sha256(reference_path))
    reference=np.concatenate(references);y=np.concatenate(labels);weights=np.concatenate(weights)*len(y);models={}
    for cue,seed in [('reference',0)]+[(cue,s)for cue in ['rgb','ir']for s in range(3)]:
        started=time.perf_counter();x=reference
        if cue!='reference':
            values=[]
            for seq in protocol['train']:
                path=work/'native_candidate_features'/seq/cue/'features.npz';values.append(read(path,f'native_s{seed}')[indices[seq]])
                sources[seq][cue+'_sha256']=sha256(path)
            x=np.c_[reference,np.concatenate(values)]
        mean=np.average(x,axis=0,weights=weights).astype(np.float32)
        std=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)),.001).astype(np.float32)
        model=LogisticRegression(C=config['C'],max_iter=config['max_iter'],tol=config['tolerance'],solver='lbfgs',random_state=0)
        model.fit((x-mean)/std,y,sample_weight=weights)
        if model.n_iter_[0]>=config['max_iter']:raise RuntimeError('Unconverged probe')
        key=f'{cue}_s{seed}';path=out/(key+'.npz')
        np.savez_compressed(path,mean=mean,std=std,coefficient=model.coef_.ravel(),intercept=model.intercept_)
        models[key]=dict(cue=cue,seed=seed,path=str(path),sha256=sha256(path),iterations=int(model.n_iter_[0]),seconds=time.perf_counter()-started)
        print(json.dumps(dict(model=key,Train_rows=len(y),counts=counts,iterations=int(model.n_iter_[0]),seconds=models[key]['seconds'])),flush=True)
    dump(out/'train_complete.json',dict(status='ALL7_TRAIN_FITS_COMPLETE_BEFORE_DEV_LABELS',models=models,sources=sources,Train_rows=len(y),Train_counts=counts,config_sha256=sha256(config_path),Dev_fit_rows=0,Test_read=False))
    rows=[]
    for seq in protocol['dev']:
        gt=np.loadtxt(Path(main['evaluation']['GT_root'])/'train'/seq/'gt/gt.txt',delimiter=',',ndmin=2)
        for front in ['rgb','new']:
            path=work/'native_eval_features/dev'/seq/'rgb'/(front+'.npz')
            with np.load(path)as z:
                boxes=z['boxes'];frames=np.repeat(z['frame_ids'],np.diff(z['offsets']))
                reference=np.c_[z['reference_post'],geometry(boxes)];native={s:z[f'native_s{s}']for s in range(3)}
            y=targets(gt,frames,boxes,config['threshold']);valid=y>=0
            for key,info in models.items():
                cue,seed=info['cue'],info['seed'];x=reference
                if cue=='rgb':x=np.c_[reference,native[seed]]
                elif cue=='ir':x=np.c_[reference,read(work/'native_eval_features/dev'/seq/'ir'/(front+'.npz'),f'native_s{seed}')]
                with np.load(info['path'])as m:logit=((x-m['mean'])/m['std'])@m['coefficient']+m['intercept'][0]
                rows.append(dict(sequence=seq,frontend=front,model=key,frames=int(frames.max()),evaluated_rows=int(valid.sum()),
                    positives=int((y==1).sum()),negatives=int((y==0).sum()),AUC=float(roc_auc_score(y[valid],logit[valid])),AP=float(average_precision_score(y[valid],logit[valid]))))
    report=dict(status='COMPLETE_TRAIN_FITTED_FULL_DEV_OVERLAP_RISK_DIAGNOSTIC',rows=rows,models=models,Train_counts=counts,
        config_sha256=sha256(config_path),source_sha256=sha256(__file__),Test_read=False,Dev_fit_rows=0,supported_innovations=0,goal_met=False,
        scope='Ordinary binary box-overlap risk probes only. Not actual occlusion visibility, tracking HOTA, independent identity evidence or deployed tracker.')
    dump(out/'complete.json',report);dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_OVERLAP_RISK_DEV_AUDIT.json',report)
    for row in rows:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
