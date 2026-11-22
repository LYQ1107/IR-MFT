#!/usr/bin/env python3
"""Dev-only appearance diagnosis; annotations are never inference inputs."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.association_features import cosine
from yolox.tracker.matching import ious, linear_assignment

ROOT = Path(__file__).resolve().parents[1]


def statistics(score, label):
    score, label = np.asarray(score), np.asarray(label, bool)
    positive, negative = int(label.sum()), int((~label).sum())
    if not positive or not negative:
        return dict(positive=positive, negative=negative, AUC=None)
    auc = (rankdata(score)[label].sum()-positive*(positive+1)/2)/(positive*negative)
    return dict(positive=positive, negative=negative, AUC=float(auc),
                positive_median=float(np.median(score[label])),negative_median=float(np.median(score[~label])))


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    work = Path(plan['work_root']); rows=[]
    for sequence in protocol['dev']:
        gt_path = Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        gt = np.loadtxt(gt_path,delimiter=',',ndmin=2)
        gt = gt[(gt[:,6]>0)&(gt[:,7]==1)]
        for front in ['new','rgb']:
            path=work/'features/dev'/sequence/(front+'.npz')
            with np.load(path) as source:
                boxes=source['boxes'];offsets=source['offsets'];old=source['embedding'].astype(np.float32)
            vectors={'old_RGB_pre_neck':old}; sources={str(path):sha256(path),str(gt_path):sha256(gt_path)}
            for cue in ['rgb','ir']:
                path=work/'native_eval_features/dev'/sequence/cue/(front+'.npz')
                with np.load(path) as source:
                    np.testing.assert_array_equal(source['boxes'],boxes)
                    for seed in range(3):vectors[f'{cue}_native_s{seed}']=source[f'native_s{seed}']
                    if cue=='rgb':vectors['RGB_teacher_post_neck']=source['reference_post']
                sources[str(path)]=sha256(path)
            labels=np.full(len(boxes),-1,np.int64)
            for frame in range(1,len(offsets)):
                a,b=offsets[frame-1:frame+1];group=gt[gt[:,0]==frame]
                gb=group[:,2:6].copy();gb[:,2:]+=gb[:,:2]
                matches,_,_=linear_assignment(1-ious(gb,boxes[a:b]),.5)
                for gi,pi in matches:labels[a+pi]=int(group[gi,1])
            for lag in [1,8,32,128]:
                frames=np.unique(np.linspace(1,len(offsets)-lag-1,256,dtype=int))
                truth=[];scores={key:[] for key in vectors}
                for frame in frames:
                    a,b=offsets[frame-1:frame+1];c,d=offsets[frame+lag-1:frame+lag+1]
                    known=np.flatnonzero(labels[a:b]>=0)
                    if not len(known) or c==d:continue
                    truth.append(((labels[a:b][known,None]==labels[c:d][None])&(labels[c:d][None]>=0)).ravel())
                    for key,v in vectors.items():scores[key].append(cosine(v[a:b][known],v[c:d]).ravel())
                label=np.concatenate(truth)
                result={key:statistics(np.concatenate(value),label) for key,value in scores.items()}
                rows.append(dict(sequence=sequence,frontend=front,lag=lag,metrics=result,sources=sources))
                print(json.dumps(dict(sequence=sequence,frontend=front,lag=lag,AUC={key:round(value['AUC'],4) for key,value in result.items()})),flush=True)
    dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_NATIVE_DEV_IDENTITY_AUDIT.json',dict(
        status='COMPLETE_DEV_ONLY_IDENTITY_DIAGNOSIS',rows=rows,Test_read=False,
        scope='IoU0.5 Hungarian labels are posthoc Dev diagnostics only. No parameter or encoder fitting, no annotation input to tracking. Sampled query edge AUC is not HOTA.',source_sha256=sha256(__file__)))


if __name__=='__main__':
    main()
