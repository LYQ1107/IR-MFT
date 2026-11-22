#!/usr/bin/env python3
"""Measure raw/BN-neck identity conditioning on the same Train-only pairs."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.audit_long_identity_evidence import auc
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    path=w/f'native_identity_space_s{a.seed}_audit.json'
    if path.exists():raise ValueError('Completed native identity audit exists')
    names=['reference_s0',f'rgb_s{a.seed}',f'ir_s{a.seed}']
    while not all((w/'native_train_features'/name/'complete.json').exists() for name in names):time.sleep(5)
    data={};sources={};meta_arrays=None
    for name in names:
        directory=w/'native_train_features'/name;metadata=json.loads((directory/'complete.json').read_text());source=directory/'features.npz'
        if metadata['Dev_fit_rows'] or metadata['Test_fit_rows'] or sha256(source)!=metadata['feature_sha256']:raise ValueError('Native Train provenance')
        with np.load(source) as bank:
            index={k:bank[k].copy() for k in ['label','frame','sequence']}
            if meta_arrays is not None:
                for k in index:np.testing.assert_array_equal(index[k],meta_arrays[k])
            else:meta_arrays=index
            for key in ['pre_neck','post_neck']:data[name+'_'+key]=bank[key].astype(np.float32)
        sources[name]=metadata
    rng=np.random.default_rng(20261123);rows=[]
    for video,sequence in enumerate(protocol['train']):
        indices=np.flatnonzero(meta_arrays['sequence']==video)
        for gap in [25,128]:
            positive=[];negative=[]
            for _ in range(8000):
                i=int(rng.choice(indices));later=indices[meta_arrays['frame'][indices]>=meta_arrays['frame'][i]+gap]
                if not len(later):continue
                same=later[meta_arrays['label'][later]==meta_arrays['label'][i]];other=later[meta_arrays['label'][later]!=meta_arrays['label'][i]]
                if not len(same) or not len(other):continue
                positive.append((i,int(rng.choice(same))));negative.append((i,int(rng.choice(other))))
                if len(positive)>=2000:break
            if not positive:continue
            p=np.array(positive);n=np.array(negative);statistics={};margins={}
            for key,x in data.items():
                ps=(x[p[:,0]]*x[p[:,1]]).sum(1);ns=(x[n[:,0]]*x[n[:,1]]).sum(1)
                statistics[key]=dict(AUC=auc(ps,ns),positive_median=float(np.median(ps)),negative_median=float(np.median(ns)),negative_fraction_cosine_ge05=float((ns>=.5).mean()))
                margins[key]=ps-ns
            ref=margins['reference_s0_post_neck'];complement={}
            for cue in ['rgb','ir']:
                value=margins[f'{cue}_s{a.seed}_post_neck'];complement[cue]=dict(reference_pair_errors=int((ref<=0).sum()),correct_on_reference_errors=int(((ref<=0)&(value>0)).sum()),wrong_on_reference_correct=int(((ref>0)&(value<=0)).sum()))
            row=dict(sequence=sequence,minimum_gap=gap,pairs=len(p),statistics=statistics,complementarity=complement);rows.append(row);print(json.dumps(row),flush=True)
    dump(path,dict(status='COMPLETE_TRAIN_ONLY_NATIVE_SPACE_AUDIT',seed=a.seed,sources=sources,rows=rows,scope='Same fixed Train pair draws; all encoder identities were seen in training. This diagnoses feature conditioning and descriptive complementarity, not held-out recognition, tracking HOTA, innovation or Test improvement.',Dev_rows=0,Test_rows=0,source_sha256=sha256(__file__)))


if __name__=='__main__':main()
