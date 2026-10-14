#!/usr/bin/env python3
"""Exercise real IPC tail validation without changing normal padded8 outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import cv2
import numpy as np
from cfp.raw_profile_tail_runtime import RawProfileTailValidatedRuntime
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if a.output.exists() or physical not in ['0','1','2','3']:raise ValueError('New approved idle-GPU runtime proof required')
    memory,utilization=map(int,subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
    if memory>64 or utilization:raise ValueError('Never take an active GPU')
    c=json.loads(Path('configs/cfp/phase_vii_whole_raw_current_profile_v1.json').read_text());runtime=RawProfileTailValidatedRuntime(c['detector_python'],c['detector_checkpoint'],c['scripts_root'])
    rows=json.loads((a.root/'identity_two_crop_index_v1/SN-015.frames.json').read_text());proof=json.loads((a.root/'detector_cache_tail_batch_diagnosis_v1.json').read_text());results=[]
    with np.load(Path(c['detections'])/'SN-015.npz',allow_pickle=False) as bank:
        offsets,boxes,scores=[bank[k].copy() for k in ['offsets','boxes','scores']]
    try:
        for entry in proof['results']:
            frame=entry['frame'];row=rows[frame-1];rgb=cv2.imread(str(Path(c['data_root'])/'train'/row['rgb_file']));ir=cv2.imread(str(Path(c['data_root'])/'train'/row['ir_file']))
            before,_=runtime.process(rgb,ir);validated=runtime.validate_tail(rgb,ir,3,entry['original_tail_position']);after,_=runtime.process(rgb,ir);lo,hi=offsets[frame-1:frame+1]
            exact=np.array_equal(validated['boxes'],boxes[lo:hi]) and np.array_equal(validated['scores'],scores[lo:hi])
            preserved=np.array_equal(before['boxes'],after['boxes']) and np.array_equal(before['scores'],after['scores'])
            if not exact or not preserved:raise ValueError('Actual tail validation changed normal output or failed exact cache reconstruction')
            results.append(dict(frame=frame,normal_pad8_before_after_byte_equal=preserved,current_only_tail3_reconstructs_oldcache=exact,
                normal_boxes_sha256=__import__('hashlib').sha256(before['boxes'].tobytes()).hexdigest(),normal_scores_sha256=__import__('hashlib').sha256(before['scores'].tobytes()).hexdigest()))
    finally:shutdown=runtime.close()
    write_json(a.output,dict(status='VERIFIED_ACTUAL_TAIL_IPC_AND_UNCHANGED_NORMAL_PAD8',results=results,runtime=shutdown,
        sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,'cfp/raw_profile_tail_runtime.py','tools/cfp_detector_worker_raw_profile_v2.py']),
        scope='Isolated currentRGB/currentcachedIR numerical IPC proof only, no rawFPS/GT/tracker/no-cache inference claim. Actualwhole raw v3 stillgeneratesIR beforecurrent normalpad8 andcalls exacttail3 validation afteroutput. Validation never changes normalpad8 output or supplies online proposals.'))


if __name__=='__main__':main()
