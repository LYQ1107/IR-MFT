#!/usr/bin/env python3
"""Bounded derived Train crops; never copy original full image datasets."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.native_cue import native_crop


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);a=p.parse_args()
    cv2.setNumThreads(1)
    config_path=ROOT/'configs/perceptual_native_cue_20261123.json';config=json.loads(config_path.read_text())
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    if a.sequence not in protocol['train']:raise ValueError('Only Train images may enter fitting crops')
    source=w/'train_bank/train.npz';meta=json.loads((source.parent/'complete.json').read_text())
    if meta['status']!='COMPLETE_TRAIN_ONLY' or meta['Dev_rows_used'] or meta['Test_rows_used'] or meta['sequences']!=protocol['train'] or sha256(source)!=meta['feature_bank_sha256']:raise ValueError('Train source changed')
    directory=w/'native_crops'/a.sequence;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():raise ValueError('Native crops are immutable')
    if shutil.disk_usage(directory).free<40*2**30:raise RuntimeError('Work filesystem reserve')
    with np.load(source) as bank:
        indices=np.flatnonzero(bank['sequence']==protocol['train'].index(a.sequence));frames=bank['frame'][indices].copy()
        boxes=bank['boxes'][indices].astype(np.float32);boxes[:,2:]+=boxes[:,:2];label=bank['label'][indices].copy()
    count=len(indices)
    arrays=dict(indices=indices,frame=frames,boxes=boxes,label=label,rectangles=np.zeros((count,4),np.int16),
                rgb=np.zeros((count,128,256,3),np.uint8),ir=np.zeros((count,128,256,1),np.uint8))
    started=time.perf_counter();unique=np.unique(frames)
    for position,frame in enumerate(unique):
        frame=int(frame);rgb_path=Path(plan['data_root'])/'train'/a.sequence/'visible'/f'{frame:06d}.jpg'
        ir_path=Path(plan['generated_ir_root'])/'train'/a.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
        rgb=cv2.imread(str(rgb_path));ir=cv2.imread(str(ir_path),cv2.IMREAD_UNCHANGED)
        if rgb is None or ir is None or ir.ndim!=2 or ir.dtype!=np.uint8 or rgb.shape[:2]!=ir.shape:raise ValueError('Actual November Train RGB/IR pair missing')
        for index in np.flatnonzero(frames==frame):
            arrays['rgb'][index],rectangle=native_crop(rgb,boxes[index]);arrays['ir'][index],other=native_crop(ir,boxes[index])
            np.testing.assert_array_equal(rectangle,other);arrays['rectangles'][index]=rectangle
        if (position+1)%100==0 or position+1==len(unique):
            progress=dict(sequence=a.sequence,frames=position+1,total_frames=len(unique),seconds=time.perf_counter()-started)
            dump(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
    path=directory/'crops.npz';temporary=directory/'crops.tmp.npz';np.savez_compressed(temporary,**arrays);os.replace(temporary,path)
    dump(directory/'complete.json',dict(status='COMPLETE_TRAIN_ONLY',sequence=a.sequence,observations=count,frames=len(unique),source_bank_sha256=sha256(source),config_sha256=sha256(config_path),crop_sha256=sha256(path),bytes=path.stat().st_size,raw_pixel_bytes=arrays['rgb'].nbytes+arrays['ir'].nbytes,Dev_rows=0,Test_rows=0,source_sha256=sha256(__file__),pixel_source_sha256=sha256(ROOT/'yolox/tracker/native_cue.py'),seconds=time.perf_counter()-started,scope='Derived compact Train crop representations, no copied full images or legacy IR pixels. Identity labels copied only for Train losses; not pixel-preparation inputs.'))


if __name__=='__main__':main()
