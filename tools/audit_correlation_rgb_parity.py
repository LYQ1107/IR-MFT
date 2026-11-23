#!/usr/bin/env python3
"""Quantify complete frozen RGB numerical parity without reading labels."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.extract_correlation_policies import detections
from yolox.exp import get_exp
from yolox.tracker.foveated_regions import exact_half_preproc
from yolox.utils import postprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only physical2/3')
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    config_path=ROOT/'configs/perceptual_correlation_repair_20261123.json';config=json.loads(config_path.read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text())
    if a.sequence not in protocol['dev']:raise ValueError('Full Dev numerical audit only')
    work=Path(plan['work_root']);out=work/'correlation_RGB_parity'/(a.sequence+'.json')
    if out.exists():raise ValueError('Completed parity proof exists')
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.45,0)
    checkpoint=Path(config['source_RGB_checkpoint'])
    if sha256(checkpoint)!=config['source_RGB_sha256']:raise ValueError('Frozen checkpoint changed')
    model=get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'),None).get_model()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);del saved
    model=model.cuda().half().eval().requires_grad_(False)
    source=Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(a.sequence+'.npz')
    with np.load(source)as z:reference={key:z[key]for key in ['frame_ids','offsets','boxes','scores']}
    mismatches=[];started=time.perf_counter()
    with torch.inference_mode():
        for first in range(1,len(reference['frame_ids'])+1,4):
            frames=list(range(first,min(first+4,len(reference['frame_ids'])+1)));inputs=[];ratios=[]
            for frame in frames:
                image=cv2.imread(str(Path(plan['data_root'])/'train'/a.sequence/'visible'/f'{frame:06d}.jpg'))
                value,ratio=exact_half_preproc(image,config['inference_input_hw'],(.485,.456,.406),(.229,.224,.225))
                inputs.append(value);ratios.append(ratio)
            outputs=model(torch.from_numpy(np.stack(inputs)).cuda())
            for frame,value,ratio in zip(frames,postprocess(outputs,1,.1,.7),ratios):
                current=detections(value,ratio);lo,hi=reference['offsets'][frame-1:frame+1]
                old=np.c_[reference['boxes'][lo:hi],reference['scores'][lo:hi]]
                if np.array_equal(current,old):continue
                row=dict(frame=frame,current_count=len(current),original_count=len(old),batch_size=len(frames))
                if current.shape==old.shape and current.size:
                    row.update(max_box_difference=float(abs(current[:,:4]-old[:,:4]).max()),
                               max_score_difference=float(abs(current[:,4]-old[:,4]).max()),
                               high_threshold_changed=int(((current[:,4]>.6)!=(old[:,4]>.6)).sum()),
                               birth_threshold_changed=int(((current[:,4]>.7)!=(old[:,4]>.7)).sum()))
                mismatches.append(row)
    dump(out,dict(status='COMPLETE_FULL_DEV_RGB_NUMERICAL_AUDIT',sequence=a.sequence,frames=len(reference['frame_ids']),
        exact_frames=len(reference['frame_ids'])-len(mismatches),mismatches=mismatches,source_sha256=sha256(__file__),
        checkpoint_sha256=config['source_RGB_sha256'],reference_sha256=sha256(source),physical_gpu=physical,
        GT_read=False,Test_read=False,seconds=time.perf_counter()-started))


if __name__=='__main__':main()
