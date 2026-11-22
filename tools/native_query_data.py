"""Verified Train query packets shared by prior fitting and residual training."""
import json
from pathlib import Path

import numpy as np
import torch
from yolox.tracker.association_calibrator import AssociationCalibrator,prepare_calibration_cue
from tools.run_pseudo_ir_comparison import sha256

ROOT=Path(__file__).resolve().parents[1]


def load_query_group(work,sequence,regime,cue=None,seed=0):
    work=Path(work);config=json.loads((ROOT/'configs/perceptual_native_query_v2_20261123.json').read_text())
    folder=work/'native_query_bank'/sequence;metadata=json.loads((folder/'complete.json').read_text())
    if metadata['status']!='COMPLETE_TRAIN_ONLY_NATIVE_QUERY_BANK' or metadata['Dev_rows'] or metadata['Test_rows'] or metadata['config_sha256']!=config['source_feature_plan_sha256']:raise ValueError('Native query bank provenance')
    source=work/'association_train'/sequence/'pairs.npz' if regime=='local' else work/'association_long_train_v2'/(sequence+'.npz')
    if metadata['sources'][str(source)]!=sha256(source):raise ValueError('Old query bank changed')
    with np.load(source) as z:
        base=z['base'].astype(np.float32);rgb=z['rgb'].astype(np.float32);point_reference=z['reference'].astype(np.float32);labels=z['label'].copy();query=z['query'].copy()
        IR_quality=z['ir'][:,2:4].astype(np.float32) if cue=='ir' else None
    files=metadata['outputs'][regime];info=files['reference'];reference_path=Path(info['path'])
    if sha256(reference_path)!=info['sha256']:raise ValueError('Post-neck RGB packet changed')
    with np.load(reference_path) as z:identity=z['RGB3'].copy();reference7=z['reference7'].copy()
    if len(identity)!=len(base):raise ValueError('RGB query alignment')
    reference=np.c_[reference7,base[:,10:12]].astype(np.float32);features=np.c_[base,identity].astype(np.float32)
    sources=dict(old=sha256(source),reference=info['sha256'])
    if cue is None:native=reference.copy();available=np.ones(len(base),bool)
    else:
        info=files[f'{cue}_s{seed}'];path=Path(info['path'])
        if sha256(path)!=info['sha256']:raise ValueError('Native cue packet changed')
        with np.load(path) as z:packet=z['cue7'].copy()
        quality=IR_quality if cue=='ir' else base[:,10:12]
        native=np.c_[packet,quality].astype(np.float32);available=(quality[:,0]>=.15)&(quality[:,1]>=.15) if cue=='ir' else np.ones(len(base),bool);sources['native']=info['sha256']
    unique,start,count=np.unique(query,return_index=True,return_counts=True)
    np.testing.assert_array_equal(unique,np.arange(len(unique)))
    offsets=np.r_[start,len(query)];position=np.arange(len(query))-np.repeat(start,count)
    number=np.add.reduceat(labels.astype(np.int32),start)
    if (number>1).any():raise ValueError('One identity query cannot have two true current candidates')
    target=np.full(len(unique),-1,np.int64);positive=np.flatnonzero(labels);target[query[positive]]=position[positive]
    return dict(base=features,RGB_extra=prepare_calibration_cue(rgb,'rgb'),point_reference=point_reference,cue=native,reference=reference,available=available,
                label=labels.astype(np.float32),query=query,counts=count,offsets=offsets,position=position,target=target,regime=regime,sequence=sequence,sources=sources)


def frozen_RGB_model(frontend,seed):
    selection=json.loads((ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_PAIR_SELECTION.json').read_text())
    item=next(row for row in selection['rows'] if row['frontend']==frontend and row['cue']=='rgb' and row['seed']==seed)
    if sha256(item['checkpoint'])!=item['checkpoint_sha256']:raise ValueError('Fixed RGB checkpoint changed')
    checkpoint=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
    model=AssociationCalibrator(**checkpoint['architecture']);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model,item


def RGB_logits(model,group,base=None):
    base=group['base'][:,:23] if base is None else base
    result=np.zeros(len(base),np.float32)
    with torch.inference_mode():
        for lo in range(0,len(base),4096):
            b=torch.from_numpy(base[lo:lo+4096].copy());c=torch.from_numpy(group['RGB_extra'][lo:lo+4096]);r=torch.from_numpy(group['point_reference'][lo:lo+4096])
            result[lo:lo+len(b)]=model(b,c,r,torch.ones(len(b),dtype=torch.bool)).numpy()
    return result
