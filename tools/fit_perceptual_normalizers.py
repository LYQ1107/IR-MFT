#!/usr/bin/env python3
"""Classical mean/PCA/Fisher controls, fitted strictly on the completed Train bank."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    meta=json.loads((work/'train_bank/complete.json').read_text());source=work/'train_bank/train.npz'
    if meta['Dev_rows_used'] or meta['Test_rows_used'] or sha256(source)!=meta['feature_bank_sha256']:raise ValueError('Invalid train source')
    out=work/'normalizers';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('Completed normalizers already exist')
    results={name:{} for name in ['center','pca128','fisher96']}
    with np.load(source) as z:
        labels=z['label']
        for field in ['embedding','ir_mask_embedding','rgb_mask_embedding']:
            x=z[field].astype(np.float64);mean=x.mean(axis=0);centered=x-mean
            covariance=centered.T@centered/max(len(x)-1,1)
            values,vectors=np.linalg.eigh(covariance);order=np.argsort(values)[::-1];values,vectors=values[order],vectors[:,order]
            projection=vectors[:,:128]/np.sqrt(values[:128]+.01*values.mean())[None]
            residual=x.copy();class_means=[]
            for identity in range(98):
                pick=labels==identity;m=x[pick].mean(0);residual[pick]-=m;class_means.append(m-mean)
            within=residual.T@residual/max(len(x)-98,1)
            regularized=within+np.eye(x.shape[1])*.05*np.trace(within)/x.shape[1]
            wv,wu=np.linalg.eigh(regularized)
            whitening=wu@np.diag(1/np.sqrt(np.maximum(wv,1e-8)))@wu.T
            between=np.stack(class_means);sb=between.T@between/98
            bv,bu=np.linalg.eigh(whitening@sb@whitening)
            fisher=whitening@bu[:,np.argsort(bv)[::-1][:96]]
            for name,proj in [('center',np.empty((0,0))),('pca128',projection),('fisher96',fisher)]:
                results[name][field+'_mean']=mean.astype(np.float32)
                results[name][field+'_projection']=proj.astype(np.float32)
    outputs={}
    for name,arrays in results.items():
        target=out/(name+'.npz');np.savez_compressed(target,**arrays)
        outputs[name]=dict(path=str(target),sha256=sha256(target),bytes=target.stat().st_size)
    dump(out/'complete.json',dict(status='COMPLETE_TRAIN_ONLY_CLASSICAL_CONTROLS',source_sha256=sha256(source),
                                 observations=11967,identities=98,Dev_rows=0,Test_rows=0,outputs=outputs,
                                 scope='Ordinary centering, PCA and shrinkage Fisher discriminants; not new multimodal innovations.'))
    print(json.dumps(outputs),flush=True)


if __name__=='__main__':main()
