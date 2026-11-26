#!/usr/bin/env python3
"""Equal99th-percentile wrong-identity gates from existing Train vectors."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.run_pseudo_ir_comparison import sha256


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    work=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122');parent=work/'tracklet_permanence_20261127_r2/Train_thresholds.json'
    original=json.loads(parent.read_text());base=work/'ambiguity_reasoning_20261127';base.mkdir(exist_ok=True);out=base/'Train_thresholds.json';assert not out.exists()
    arrays={};metadata=None
    for cue in ['rgb','ir']:
        source=original['sources'][cue];assert sha256(source['path'])==source['sha256']
        with np.load(source['path'])as bank:
            arrays[cue]=bank['post_neck'].copy();current={k:bank[k].copy()for k in ['label','sequence','frame']}
        if metadata is None:metadata=current
        else:
            for k in metadata:np.testing.assert_array_equal(metadata[k],current[k])
    negatives=[]
    for video in np.unique(metadata['sequence']):
        indices=np.flatnonzero(metadata['sequence']==video);labels=np.unique(metadata['label'][indices])
        for rank,query in enumerate(indices[np.unique(np.linspace(0,len(indices)-1,min(2048,len(indices)),dtype=int))]):
            other=labels[labels!=metadata['label'][query]]
            if not len(other):continue
            pool=indices[metadata['label'][indices]==other[rank%len(other)]]
            negative=pool[np.argmin(np.abs(metadata['frame'][pool]-metadata['frame'][query]))]
            negatives.append((int(query),int(negative)))
    pairs=np.asarray(negatives);thresholds=dict(original['thresholds']);statistics={}
    for cue in arrays:
        d=arrays[cue];score=np.sum(d[pairs[:,0]]*d[pairs[:,1]],axis=1);q99=float(np.quantile(score,.99));q95=float(np.quantile(score,.95))
        assert len(score)==original['statistics'][cue]['negative_pairs']
        old=original['statistics'][cue]
        assert abs(float(np.quantile(score,old['quantile']))-old['negative_quantile'])<1e-7
        thresholds['primary_'+cue.upper()+'_cosine_threshold']=q99
        statistics[cue]=dict(wrong_identity_pairs=len(score),quantile=.99,threshold=q99,q95=q95,
            fraction_above=float((score>=q99).mean()))
    spec=ROOT/'configs/perceptual_ambiguity_reasoning_20261127.json'
    value=dict(status='COMPLETE_EQUAL_QUANTILE_TRAIN_AMBIGUITY_THRESHOLDS',thresholds=thresholds,statistics=statistics,
        original_late_thresholds_preserved=True,source_Train_thresholds_sha256=sha256(parent),
        source_Train_features=original['sources'],specification_sha256=sha256(spec),source_sha256=sha256(Path(__file__)),
        Dev_rows=0,Test_rows=0,GPU_work=False,new_NN_fits=0,optimistic_seen_encoder_IDs=True)
    out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(status=value['status'],thresholds=thresholds,statistics=statistics)),flush=True)


if __name__=='__main__':main()
