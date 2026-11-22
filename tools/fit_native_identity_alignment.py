#!/usr/bin/env python3
"""Equal-video orthogonal alignment on paired Train crops only."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root']);protocol=json.loads((ROOT/plan['protocol']).read_text())
    if json.loads((w/'native_train_features_state.json').read_text())['status']!='COMPLETE':raise ValueError('All immutable Train features required')
    directory=w/'native_alignment';directory.mkdir(exist_ok=True)
    if (directory/'complete.json').exists():raise ValueError('Alignment already fitted')
    reference=w/'native_train_features/reference_s0/features.npz';rm=json.loads((reference.parent/'complete.json').read_text())
    if sha256(reference)!=rm['feature_sha256'] or rm['Dev_fit_rows'] or rm['Test_fit_rows']:raise ValueError('RGB Train reference provenance')
    with np.load(reference) as bank:target=bank['post_neck'].astype(np.float64);video=bank['sequence'].copy();labels=bank['label'].copy();frames=bank['frame'].copy()
    target/=np.maximum(np.linalg.norm(target,axis=1,keepdims=True),1e-8);outputs={}
    for cue in ['rgb','ir']:
        for seed in [0,1,2]:
            path=w/f'native_train_features/{cue}_s{seed}/features.npz';meta=json.loads((path.parent/'complete.json').read_text())
            if sha256(path)!=meta['feature_sha256'] or meta['Dev_fit_rows'] or meta['Test_fit_rows']:raise ValueError('Native Train feature provenance')
            with np.load(path) as bank:
                np.testing.assert_array_equal(bank['sequence'],video);np.testing.assert_array_equal(bank['label'],labels);np.testing.assert_array_equal(bank['frame'],frames);source=bank['post_neck'].astype(np.float64)
            source/=np.maximum(np.linalg.norm(source,axis=1,keepdims=True),1e-8)
            weight=np.zeros(len(video),np.float64)
            for v in range(6):weight[video==v]=1/(6*int((video==v).sum()))
            covariance=source.T@(target*weight[:,None]);u,s,vt=np.linalg.svd(covariance,full_matrices=False);rotation=(u@vt).astype(np.float32)
            before=float(((source*target).sum(1)*weight).sum());after=float((((source@rotation)*target).sum(1)*weight).sum())
            error=float(np.linalg.norm(rotation.T@rotation-np.eye(512)))
            if after+1e-5<before or error>1e-3:raise ValueError('Incorrect orthogonal alignment')
            name=f'{cue}_s{seed}';output=directory/(name+'.npz');temporary=directory/(name+'.tmp.npz');np.savez_compressed(temporary,rotation=rotation);os.replace(temporary,output)
            outputs[name]=dict(path=str(output),sha256=sha256(output),native_feature_sha256=sha256(path),reference_feature_sha256=sha256(reference),paired_cosine_before=before,paired_cosine_after=after,orthogonality_error=error,observations=len(video));print(json.dumps(dict(cue=cue,seed=seed,before=before,after=after,error=error)),flush=True)
    dump(directory/'complete.json',dict(status='COMPLETE_TRAIN_ONLY_ORTHOGONAL_ALIGNMENT',outputs=outputs,fit_sequences=protocol['train'],Dev_rows=0,Test_rows=0,labels_used_for_fit=False,weighting='Equal Train video, paired current observations within video; SVD orthogonal Procrustes',source_sha256=sha256(__file__)))


if __name__=='__main__':main()
