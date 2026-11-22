#!/usr/bin/env python3
"""Train-only feasibility of identity evidence over temporal occlusion gaps."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import rankdata
from tools.run_pseudo_ir_comparison import dump,sha256


def auc(positive,negative):
    joined=np.r_[positive,negative];ranks=rankdata(joined,method='average');n=len(positive);m=len(negative)
    return float((ranks[:n].sum()-n*(n+1)/2)/(n*m))


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    source=w/'train_bank/train.npz';metadata=json.loads((source.parent/'complete.json').read_text())
    assert metadata['status']=='COMPLETE_TRAIN_ONLY' and not metadata['Dev_rows_used'] and not metadata['Test_rows_used']
    assert sha256(source)==metadata['feature_bank_sha256']
    with np.load(source) as z:data={k:z[k] for k in z.files}
    rng=np.random.default_rng(20261123);rows=[]
    fields=['embedding','rgb_mask_embedding','ir_mask_embedding','color','rgb_hog','ir_hog']
    features={}
    for k in fields:
        x=data[k].astype(np.float32);features[k]=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    for video,sequence in enumerate(protocol['train']):
        indices=np.flatnonzero(data['sequence']==video)
        for gap in [25,128]:
            positive=[];negative=[]
            for _ in range(8000):
                i=int(rng.choice(indices));later=indices[data['frame'][indices]>=data['frame'][i]+gap]
                if not len(later):continue
                same=later[data['label'][later]==data['label'][i]];different=later[data['label'][later]!=data['label'][i]]
                if not len(same) or not len(different):continue
                positive.append((i,int(rng.choice(same))));negative.append((i,int(rng.choice(different))))
                if len(positive)>=2000:break
            if not positive:continue
            p=np.array(positive);n=np.array(negative);statistics={}
            for key,x in features.items():
                ps=(x[p[:,0]]*x[p[:,1]]).sum(1);ns=(x[n[:,0]]*x[n[:,1]]).sum(1)
                statistics[key]=dict(AUC=auc(ps,ns),positive_median=float(np.median(ps)),negative_median=float(np.median(ns)))
            row=dict(sequence=sequence,minimum_gap=gap,positive_pairs=len(positive),negative_pairs=len(negative),statistics=statistics);rows.append(row)
            print(json.dumps(row),flush=True)
    dump(w/'long_identity_evidence_audit.json',dict(status='COMPLETE_TRAIN_ONLY_FEASIBILITY_AUDIT',scope='Same/different identity AUC on the six Train videos only, not tracking HOTA or generalization evidence. The preserved RGB encoder was trained on these identities, so this is optimistic feasibility rather than held-out recognition accuracy.',source_bank_sha256=sha256(source),rows=rows,source_sha256=sha256(__file__)))


if __name__=='__main__':main()
