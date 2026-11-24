#!/usr/bin/env python3
"""Isolate actual first-frame CPU zero-path layouts and autograd modes; no GT."""
import hashlib
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.content_verified_train_reader import ContentVerifiedTrainImageReader
from tools.fusion_memory_evaluation import read,sha
from tools.fusion_memory_model_loading import load_source_rgb_cpu
from tools.fusion_memory_observation import write_json
from tools.native_pyramid_train_data import preprocess_observation
from tools.native_pyramid_training_loop import state_sha256
from tools.task_utility_fusion import TaskUtilityFusion
from yolox.models.frozen_head_repair import head_predictions


def difference(a,b):
    return dict(exact=torch.equal(a,b),different_values=int((a!=b).sum()),max_abs=float((a-b).abs().max()))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    path=ROOT/'configs/perceptual_task_utility_prototype_20261125.json';spec=read(path)
    plan=read(ROOT/'configs/perceptual_mft_20261122.json');protocol=read(ROOT/plan['protocol'])
    teacher=load_source_rgb_cpu(spec['source_RGB_checkpoint'],spec['source_RGB_sha256']);before=state_sha256(teacher.state_dict())
    reader=ContentVerifiedTrainImageReader(plan['data_root'],plan['generated_ir_root'],protocol['train'],max_bytes=16*2**20,reserve_bytes=40*2**30)
    a=spec['CPU']['addresses'][0];rgb=reader.read(a['sequence'],a['frame'],'rgb')
    value,cue,valid,_,_,_=preprocess_observation(rgb,None,'rgb',spec['CPU']['input_hw'],read(spec['calibration']))
    tensor=torch.from_numpy(value[None].astype(np.float32));pixels=torch.from_numpy(cue[None].copy())
    rows=[]
    for layout,image in [('original_view',tensor),('contiguous',tensor.contiguous())]:
        torch.manual_seed(0);module=TaskUtilityFusion().train()
        with torch.no_grad():
            features=tuple(v.detach() for v in teacher.backbone(image));reference,metadata=head_predictions(teacher.head,features)
        output,_,diagnostics=module(features,pixels,metadata['scores'],valid=torch.from_numpy(valid[None].copy()),return_proposals=True)
        raw,_=head_predictions(teacher.head,output)
        with torch.no_grad():
            no_grad,_=head_predictions(teacher.head,output);again,_=head_predictions(teacher.head,features)
            detached,_=head_predictions(teacher.head,tuple(v.detach() for v in output))
        differentiable=tuple(v.detach().requires_grad_(True) for v in features)
        grad_reference,_=head_predictions(teacher.head,differentiable)
        copied=tuple(torch.empty_like(base).copy_(new) for base,new in zip(features,output))
        copied_raw,_=head_predictions(teacher.head,copied)
        rows.append(dict(input_layout=layout,input_stride=list(image.stride()),feature_rows=[dict(
            source_stride=list(base.stride()),output_stride=list(new.stride()),source_requires_grad=base.requires_grad,
            output_requires_grad=new.requires_grad,feature_values=difference(base,new),
            feature_bits_exact=torch.equal(base.contiguous().view(torch.uint8),new.contiguous().view(torch.uint8))) for base,new in zip(features,output)],
            initial_gradient_output_vs_inference_reference=difference(raw,reference),repeat_inference_reference=difference(again,reference),
            no_grad_output_vs_inference_reference=difference(no_grad,reference),detached_output_vs_inference_reference=difference(detached,reference),
            gradient_output_vs_gradient_reference=difference(raw,grad_reference),copied_layout_output_vs_reference=difference(copied_raw,reference)))
    assert state_sha256(teacher.state_dict())==before and all(p.grad is None for p in teacher.parameters()) and not torch.cuda.is_initialized()
    out=Path(plan['work_root'])/'task_utility_zero_layout_diagnosis_20261125.json'
    write_json(out,dict(status='COMPLETE_ACTUAL_FIRST_TRAIN_RGB_CPU_ZERO_LAYOUT_MODE_DIAGNOSIS',source_hashes={p:sha(ROOT/p) for p in
        ['tools/diagnose_task_utility_zero.py','tools/task_utility_fusion.py','tools/probe_task_utility_cpu.py','tools/fusion_memory_model_loading.py','yolox/models/yolo_head.py']},
        rows=rows,source_image=reader.content_manifest(),original_teacher_unchanged=True,GT_read=False,model_updates=0,new_GPU_work=False,
        scope='OneactualTrainRGBframe,zero-initmodule,twophysicaltensorlayoutsandheadautogradmodes. NotIRlearning,GPUqualificationoraccuracy.',goal_met=False,supported_innovations=0),exclusive=True)
    import json
    print(json.dumps(dict(output=str(out),rows=rows)),flush=True)


if __name__=='__main__':main()
