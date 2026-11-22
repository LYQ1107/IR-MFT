#!/usr/bin/env python3
"""Compact real-pixel Train crops for every exactly reconstructed proposal."""
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
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);a=p.parse_args();cv2.setNumThreads(1)
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root']);config_path=ROOT/'configs/perceptual_native_candidates_20261123.json'
    if a.sequence not in protocol['train']:raise ValueError('Only Train candidates')
    source=w/'native_candidate_index'/a.sequence/'index.npz';meta=json.loads((source.parent/'complete.json').read_text())
    if meta['status']!='COMPLETE_EXACT_TRAIN_INDEX' or meta['Dev_rows'] or meta['Test_rows'] or not meta['every_saved_geometry_field_exact'] or sha256(source)!=meta['index_sha256']:raise ValueError('Exact proposal reconstruction required')
    directory=w/'native_candidate_crops'/a.sequence;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():raise ValueError('Candidate crop bank already complete')
    if shutil.disk_usage(directory).free<40*2**30+meta['raw_candidate_pixel_bytes']:raise RuntimeError('Keep40GiB reserve even for an uncompressed candidate bank')
    with np.load(source) as bank:frames=bank['frame'].copy();boxes=bank['boxes'].copy();labels=bank['label'].copy()
    count=len(frames);arrays=dict(frame=frames,boxes=boxes,label=labels,rectangles=np.zeros((count,4),np.int16),rgb=np.zeros((count,128,256,3),np.uint8),ir=np.zeros((count,128,256,1),np.uint8))
    started=time.perf_counter();unique=np.unique(frames)
    for position,frame in enumerate(unique):
        frame=int(frame);rgb=cv2.imread(str(Path(plan['data_root'])/'train'/a.sequence/'visible'/f'{frame:06d}.jpg'))
        ir=cv2.imread(str(Path(plan['generated_ir_root'])/'train'/a.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'),cv2.IMREAD_UNCHANGED)
        if rgb is None or ir is None or ir.ndim!=2 or ir.dtype!=np.uint8 or rgb.shape[:2]!=ir.shape:raise ValueError('Actual current November pixel pair missing')
        for index in np.flatnonzero(frames==frame):
            arrays['rgb'][index],rectangle=native_crop(rgb,boxes[index]);arrays['ir'][index],other=native_crop(ir,boxes[index]);np.testing.assert_array_equal(rectangle,other);arrays['rectangles'][index]=rectangle
        if (position+1)%100==0 or position+1==len(unique):
            progress=dict(sequence=a.sequence,frames=position+1,total_frames=len(unique),observations=count,seconds=time.perf_counter()-started)
            dump(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
    path=directory/'crops.npz';temporary=directory/'crops.tmp.npz';np.savez_compressed(temporary,**arrays);os.replace(temporary,path)
    dump(directory/'complete.json',dict(status='COMPLETE_EXACT_TRAIN_CANDIDATE_CROPS',sequence=a.sequence,observations=count,frames=len(unique),index_sha256=sha256(source),config_sha256=sha256(config_path),crop_sha256=sha256(path),bytes=path.stat().st_size,raw_pixel_bytes=arrays['rgb'].nbytes+arrays['ir'].nbytes,source_sha256=sha256(__file__),pixel_source_sha256=sha256(ROOT/'yolox/tracker/native_cue.py'),Dev_rows=0,Test_rows=0,labels_used_in_pixel_preparation=False,seconds=time.perf_counter()-started,scope='Only derived resized Train proposals with jitter and backgrounds, preserving exact original pair indexing; no copied full images, no legacy IR or evaluation images.'))


if __name__=='__main__':main()
