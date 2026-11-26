#!/usr/bin/env python3
"""Existing Train-only native post-neck cosine/reachability calibration."""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.run_pseudo_ir_comparison import sha256


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    start=time.monotonic();work=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
    base=work/'tracklet_permanence_20261127';base.mkdir(exist_ok=True)
    out=base/'Train_thresholds.json';assert not out.exists()
    spec_path=ROOT/'configs/perceptual_tracklet_permanence_20261127.json';spec=json.loads(spec_path.read_text())
    arrays={};sources={};metadata=None
    for cue in ['rgb','ir']:
        folder=work/'native_train_features'/(cue+'_s0');meta=json.loads((folder/'complete.json').read_text());path=folder/'features.npz'
        assert meta['Dev_fit_rows']==meta['Test_fit_rows']==0 and meta['observations']==11967
        assert sha256(path)==meta['feature_sha256'] and sha256(meta['checkpoint'])==meta['checkpoint_sha256']==spec[cue.upper()+'_identity_checkpoint_sha256']
        with np.load(path)as bank:
            arrays[cue]=bank['post_neck'].astype(np.float32)
            current={k:bank[k].copy()for k in ['sequence','label','frame']}
        if metadata is None:metadata=current
        else:
            for k in metadata:np.testing.assert_array_equal(metadata[k],current[k])
        assert np.isfinite(arrays[cue]).all() and np.allclose(np.linalg.norm(arrays[cue],axis=1),1,atol=1e-5)
        sources[cue]=dict(path=str(path),sha256=sha256(path),metadata_sha256=sha256(folder/'complete.json'),checkpoint=meta['checkpoint'],checkpoint_sha256=meta['checkpoint_sha256'])
    positives=[];negatives=[]
    for video in np.unique(metadata['sequence']):
        indices=np.flatnonzero(metadata['sequence']==video)
        labels=np.unique(metadata['label'][indices])
        for rank,query in enumerate(indices[np.unique(np.linspace(0,len(indices)-1,min(2048,len(indices)),dtype=int))]):
            gap=metadata['frame'][query]-metadata['frame'][indices]
            pos=indices[(metadata['label'][indices]==metadata['label'][query])&(gap>=25)&(gap<=250)]
            other=labels[labels!=metadata['label'][query]]
            # Cycle wrong identities rather than repeatedly taking the lowest ID.
            neg=indices[metadata['label'][indices]==other[rank%len(other)]] if len(other) else np.empty(0,dtype=int)
            if len(pos):positives.append((int(query),int(pos[np.argmin(np.abs(metadata['frame'][pos]-(metadata['frame'][query]-64)))])))
            if len(neg):negatives.append((int(query),int(neg[np.argmin(np.abs(metadata['frame'][neg]-metadata['frame'][query]))])))
    positives=np.asarray(positives);negatives=np.asarray(negatives);thresholds={};stats={}
    for cue,quantile,floor in [('rgb',.99,.5),('ir',.95,.25)]:
        data=arrays[cue];positive=np.sum(data[positives[:,0]]*data[positives[:,1]],axis=1);negative=np.sum(data[negatives[:,0]]*data[negatives[:,1]],axis=1)
        value=max(floor,float(np.quantile(negative,quantile)));thresholds[cue.upper()+'_cosine_threshold']=value
        stats[cue]=dict(positive_pairs=len(positive),negative_pairs=len(negative),quantile=quantile,negative_quantile=float(np.quantile(negative,quantile)),threshold=value,positive_fraction_above=float((positive>=value).mean()),negative_fraction_above=float((negative>=value).mean()))
    reach=work/'identity_reachability_train.json';r=json.loads(reach.read_text());assert r['Dev_fit_rows']==r['Test_fit_rows']==0
    thresholds['Train_unknown_speed_upper']=r['unknown_speed_upper']
    value=dict(status='COMPLETE_TRAIN_ONLY_NATIVE_TRACKLET_CALIBRATION',thresholds=thresholds,statistics=stats,sources=sources,reachability_source=dict(path=str(reach),sha256=sha256(reach)),specification_sha256=sha256(spec_path),source_sha256=sha256(Path(__file__)),observations=11967,Dev_rows=0,Test_rows=0,NN_weights_changed=False,optimistic_seen_encoder_identity_calibration=True,seconds=time.monotonic()-start)
    out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(status=value['status'],thresholds=thresholds,statistics=stats,seconds=value['seconds'])),flush=True)


if __name__=='__main__':main()
