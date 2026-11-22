#!/usr/bin/env python3
"""Train-only scale-normalized motion bounds for old-identity consultation."""
import json
from pathlib import Path

import numpy as np

from tools.run_pseudo_ir_comparison import dump,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());work=Path(plan['work_root'])
    destination=work/'identity_reachability_train.json'
    if destination.exists():raise ValueError('Reachability statistics already fitted')
    rows=[]
    for sequence in protocol['train']:
        path=Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        if sha256(path)!=protocol['derived_gt_sha256']['train/'+sequence]:raise ValueError('Train GT changed')
        gt=np.loadtxt(path,delimiter=',',ndmin=2);gt=gt[(gt[:,6]>0)&(gt[:,7]==1)]
        speeds=[];accelerations=[]
        for identity in np.unique(gt[:,1]):
            group=gt[gt[:,1]==identity];group=group[np.argsort(group[:,0],kind='stable')]
            frame=group[:,0];center=group[:,2:4]+group[:,4:6]/2
            scale=np.maximum(np.linalg.norm(group[:,4:6],axis=1),1.)
            dt=np.diff(frame);velocity=np.diff(center,axis=0)/dt[:,None]
            contiguous=dt==1;speed=np.linalg.norm(velocity,axis=1)/((scale[1:]+scale[:-1])/2)
            speeds.extend(speed[contiguous].tolist())
            if len(velocity)>1:
                eligible=contiguous[1:]&contiguous[:-1]
                acceleration=np.linalg.norm(np.diff(velocity,axis=0),axis=1)/scale[1:-1]
                accelerations.extend(acceleration[eligible].tolist())
        if not speeds or not accelerations:raise ValueError('Insufficient Train trajectory support')
        percentiles=[.5,.9,.95,.99]
        rows.append(dict(sequence=sequence,GT_sha256=sha256(path),speed_samples=len(speeds),acceleration_samples=len(accelerations),
                         speed_quantiles=dict(zip(map(str,percentiles),map(float,np.quantile(speeds,percentiles)))),
                         acceleration_quantiles=dict(zip(map(str,percentiles),map(float,np.quantile(accelerations,percentiles))))))
    result=dict(status='COMPLETE_TRAIN_ONLY_REACHABILITY',fit_sequences=protocol['train'],Dev_fit_rows=0,Test_fit_rows=0,rows=rows,
        speed_floor=float(np.mean([row['speed_quantiles']['0.5'] for row in rows])),
        unknown_speed_upper=float(max(row['speed_quantiles']['0.95'] for row in rows)),
        acceleration_upper=float(max(row['acceleration_quantiles']['0.95'] for row in rows)),
        position_slack_body_diagonals=.5,
        rule='Scale by last actual RGB box diagonal. Radius=0.5+gap*max(recent median speed+2*1.4826*MAD,Train median speed)+0.5*Train conservative95% acceleration*gap^2. Fit max per-video acceleration bound to protect fast-video support. At least two actual past observations required. Mask only memory correspondence edges; do not remove visible input proposals or renormalize their probabilities.',
        source_sha256=sha256(__file__))
    dump(destination,result)
    dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_REACHABILITY_TRAIN.json',result)
    print(json.dumps({key:result[key] for key in ['speed_floor','unknown_speed_upper','acceleration_upper','position_slack_body_diagonals']}))


if __name__=='__main__':main()
