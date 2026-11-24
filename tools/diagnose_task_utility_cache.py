#!/usr/bin/env python3
"""Compare a cached CPU head reference with fresh per-step RGB, no GT/updates."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import numpy as np
import torch
from tools.content_verified_train_reader import ContentVerifiedTrainImageReader
from tools.diagnose_task_utility_zero import difference
from tools.fusion_memory_evaluation import read,sha
from tools.fusion_memory_model_loading import load_source_rgb_cpu
from tools.fusion_memory_observation import write_json
from tools.native_pyramid_train_data import preprocess_observation
from tools.native_pyramid_training_loop import state_sha256
from tools.task_utility_fusion import TaskUtilityFusion
from yolox.models.frozen_head_repair import head_predictions


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    spec=read(ROOT/'configs/perceptual_task_utility_prototype_20261125.json');plan=read(ROOT/'configs/perceptual_mft_20261122.json')
    protocol=read(ROOT/plan['protocol']);teacher=load_source_rgb_cpu(spec['source_RGB_checkpoint'],spec['source_RGB_sha256']);before=state_sha256(teacher.state_dict())
    reader=ContentVerifiedTrainImageReader(plan['data_root'],plan['generated_ir_root'],protocol['train'],max_bytes=16*2**20,reserve_bytes=40*2**30)
    address=spec['CPU']['addresses'][0];rgb=reader.read(address['sequence'],address['frame'],'rgb')
    value,cue,valid,_,_,_=preprocess_observation(rgb,None,'rgb',spec['CPU']['input_hw'],read(spec['calibration']))
    with torch.no_grad():
        features=tuple(teacher.backbone(torch.from_numpy(value[None].astype(np.float32))))
        cached,metadata=head_predictions(teacher.head,features)
    torch.manual_seed(0);module=TaskUtilityFusion().train();optimizer=torch.optim.AdamW(module.parameters(),lr=.0003)
    rows=[]
    for index in range(3):
        with torch.no_grad():fresh,metadata=head_predictions(teacher.head,features)
        output,_,diagnostics=module(features,torch.from_numpy(cue[None].copy()),metadata['scores'],
            valid=torch.from_numpy(valid[None].copy()),return_proposals=True)
        predicted,_=head_predictions(teacher.head,output)
        with torch.no_grad():
            proposal,_=head_predictions(teacher.head,diagnostics['detached_proposal_features'])
            repeated,_=head_predictions(teacher.head,features)
        rows.append(dict(repetition=index,cached_vs_fresh=difference(cached,fresh),student_vs_fresh=difference(predicted,fresh),
            proposal_vs_fresh=difference(proposal,fresh),repeated_source_vs_fresh=difference(repeated,fresh),
            source_FPN_bits_unchanged=all(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)) for a,b in zip(features,output))))
    assert before==state_sha256(teacher.state_dict()) and all(p.grad is None for p in teacher.parameters()) and not torch.cuda.is_initialized()
    path=Path(plan['work_root'])/'task_utility_fresh_reference_diagnosis_20261125.json'
    write_json(path,dict(status='COMPLETE_ACTUAL_FRESH_STEP_REFERENCE_DIAGNOSIS_NOT_QUALIFICATION',rows=rows,
        source_hashes={p:sha(ROOT/p) for p in ['tools/diagnose_task_utility_cache.py','tools/task_utility_fusion.py','tools/probe_task_utility_cpu.py']},
        model_updates=0,GT_read=False,new_GPU_work=False,source_teacher_unchanged=True,goal_met=False,supported_innovations=0),exclusive=True)
    print(json.dumps(dict(path=str(path),rows=rows)),flush=True)


if __name__=='__main__':main()
