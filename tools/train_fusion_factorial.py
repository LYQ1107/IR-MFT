#!/usr/bin/env python3
"""Joint-disable worker: CPU canary, gated GPU engineering, full-Train fit."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from tools.fusion_factorial_workflow import (
    REGISTRY, SCIENCE, context, fit_metadata, make_joint_arm, read, require_gpu,
)
from tools.native_pyramid_content_guard import PinnedTrainImageReader, make_image_reader
from tools.native_pyramid_fit import optimizer_step
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_train_data import OrderedTrainPixels, epoch_clips, preprocess_observation
from tools.native_pyramid_training_loop import run_training_to_epoch, state_sha256
from tools.neck72_runtime import require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.models.native_pyramid_fusion import NativePyramidFusion


def canary(teacher, reader, plan, study, protocol, calibration, arm, device, inference_frames):
    """Run actual shared fit code and independently encoded B1 source checks."""
    if inference_frames not in [2, 8] or arm != make_joint_arm(study, arm['mode']):
        raise ValueError('Only the preregistered joint-disable canary is supported')
    before = state_sha256(teacher.state_dict())
    torch.manual_seed(0)
    module = NativePyramidFusion(**arm['module_kwargs']).to(device=device, dtype=torch.float32)
    initial = state_sha256(module.state_dict())
    optimizer = torch.optim.AdamW(module.parameters(), lr=arm['fit']['learning_rate'],
                                 weight_decay=arm['fit']['weight_decay'])
    provider = OrderedTrainPixels(plan, protocol, arm['Train_delivery']['lengths'], calibration,
                                  reader, arm['training_input_hw'], device)
    provider.begin_epoch(1)
    clips = list(epoch_clips(provider.lengths, 1))[:2]
    assert clips[0].frames == (1,2,3,4) and clips[1].frames == (5,6,7,8)
    assert clips[0].sequence == clips[1].sequence
    state, gray, rows = None, None, []
    for position, clip in enumerate(clips):
        batch = provider.prepare(clip, 'ir')
        if position and arm['mode'] == 'temporal':
            assert state.frame == 4 and all(not v.requires_grad for v in state.latent+state.covariance)
        row, state, gray = optimizer_step(module, teacher.backbone, teacher.head,
            batch, optimizer, arm, position, state, gray)
        rows.append(row)
    assert rows[0]['initial_full_RGB_exact'] and rows[0]['native_stem_gradient_L1'] == 0
    assert rows[1]['native_stem_gradient_L1'] > 0 and rows[1]['decoded_box_max_change'] > 0
    assert all(r['optimizer_updated'] for r in rows)
    assert all(p.grad is None for p in module.quality.parameters())
    assert state_sha256(teacher.state_dict()) == before and all(p.grad is None for p in teacher.parameters())
    if arm['mode'] == 'temporal': assert state.sequence == clips[0].sequence and state.frame == 8
    temporal_end_frame = state.frame if state is not None else None
    # Preserve the explicit old convention: FP32 head for fitting; FP16 head
    # for production GPU inference. CPU engineering stays entirely FP32.
    if str(device).startswith('cuda'): teacher.head.half()
    inference_teacher = state_sha256(teacher.state_dict())
    module.eval().requires_grad_(False)
    torch.manual_seed(0)
    zero = NativePyramidFusion(**arm['module_kwargs']).to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    observations = [dict(id='source',cue='rgb',baseline='fresh_RGB_B1',seed=0),
        dict(id='trained',cue='ir',seed=0),dict(id='zero',cue='ir',seed=0,expect_RGB_exact=True),
        dict(id='missing',cue='ir',seed=0,intervention='unavailable')]
    engine = OnlineNativeDetector(teacher, observations,
        dict(source=None,trained=module,zero=zero,missing=module),study['inference_input_hw'],calibration)
    from tools.extract_structural_relative import detections
    from yolox.utils import postprocess
    checks = []
    sequence = clips[0].sequence
    with torch.no_grad():
        for frame in range(1,inference_frames+1):
            rgb, ir = reader.read(sequence,frame,'rgb'),reader.read(sequence,frame,'ir')
            packets, info = engine.step(rgb,ir,sequence,frame)
            np.testing.assert_array_equal(packets['source'],packets['zero'])
            np.testing.assert_array_equal(packets['source'],packets['missing'])
            pixels,_,_,_,ratio,_ = preprocess_observation(rgb,None,'rgb',study['inference_input_hw'],calibration)
            tensor = torch.from_numpy(pixels[None]).to(device=device,dtype=next(teacher.parameters()).dtype)
            direct = detections(postprocess(teacher(tensor).clone(),1,.1,.7)[0],ratio)
            np.testing.assert_array_equal(direct,packets['source'])
            assert info['full_head_exact_to_fresh_RGB']['zero']
            assert info['full_head_exact_to_fresh_RGB']['missing']
            checks.append(dict(sequence=sequence,frame=frame,direct_B1_packets_exact=True,
                zero_full_head_exact=True,trained_missing_full_head_exact=True))
    assert state_sha256(teacher.state_dict()) == inference_teacher
    return dict(mode=arm['mode'],cell='00',device=str(device),initial_native_sha256=initial,
        module_kwargs=arm['module_kwargs'],parameter_inventory=sum(p.numel() for p in module.parameters()),
        delivered_Train_frames=8,actual_updates=2,optimizer_rows=rows,
        delivered_prefix_sha256=provider.digest.hexdigest(),temporal_training_end_frame=temporal_end_frame,
        inference_B1_frames=inference_frames,source_zero_missing_checks=checks,
        frozen_RGB_parameters_and_BN_unchanged_before_explicit_head_precision_conversion=True,
        inference_RGB_state_unchanged=True,weights_saved=False,Dev_rows=0,Test_rows=0,
        inference_precision='FP16 frozen RGB encoder/head and FP32 native' if str(device).startswith('cuda') else 'CPU FP32 only; no GPU parity claim')


def cpu_probe():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    plan,study,protocol,calibration,work,checkpoint,base = context()
    require_host_memory(8*2**30,maximum_workers=1); require_storage(work)
    output = work/'fusion_factorial_training_worker_CPU_proof.json'
    assert not output.exists()
    sources = {}
    for name in ['native_pyramid_pipeline_registered.json','neck72_fresh_runtime_registered.json']:
        sources.update(read(work/name)['source_hashes'])
    for name in ['tools/fusion_factorial_workflow.py','tools/train_fusion_factorial.py',
                 'tests/test_fusion_factorial_workflow.py',str(SCIENCE.relative_to(ROOT))]:
        sources[name] = sha256(ROOT/name)
    assert all(sha256(ROOT/n)==h for n,h in sources.items())
    started = time.monotonic()
    log = work/'fusion_factorial_training_worker_CPU_pytest.log'
    with log.open('x') as stream:
        subprocess.run([sys.executable,'-m','pytest','-q','tests/test_fusion_factorial_workflow.py'],
                       cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    match = re.search(r'\b(\d+) passed\b',log.read_text()); assert match
    lengths = study['Train_delivery']['lengths']; clips = list(epoch_clips(lengths,1))[:2]
    expected = {}
    for clip in clips:
        for frame in clip.frames:
            for modality,data_root,folder,suffix in [('rgb',plan['data_root'],'visible','jpg'),
                    ('ir',plan['generated_ir_root'],'infrared_diffv2ir','png')]:
                path=Path(data_root)/'train'/clip.sequence/folder/f'{frame:06d}.{suffix}'
                expected[clip.sequence,frame,modality]=dict(sha256=sha256(path),bytes=path.stat().st_size)
    reader=PinnedTrainImageReader(plan['data_root'],plan['generated_ir_root'],protocol['train'],
        expected_content=expected,max_bytes=256*2**20,reserve_bytes=40*2**30)
    from yolox.exp import get_exp
    teacher=get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'),None).get_model().float().eval().requires_grad_(False)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    teacher.load_state_dict(saved['model'],strict=True);del saved
    previous=read(work/'native_contrast_quality_factorial_CPU_proof.json')
    records=[]
    for mode in study['modes']:
        row=canary(teacher,reader,plan,study,protocol,calibration,make_joint_arm(study,mode),'cpu',2)
        prior=next(r for r in previous['records'] if r['mode']==mode)
        assert row['initial_native_sha256']==prior['four_cells_identical_initial_tensor_state']['00']
        assert row['delivered_prefix_sha256']==prior['actual_delivered_prefix_sha256']
        records.append(row);print(json.dumps(dict(mode=mode,shared_worker_cpu_canary=True)),flush=True)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30;assert peak<=8
    assert all(sha256(ROOT/n)==h for n,h in sources.items()) and not torch.cuda.is_initialized()
    require_storage(work)
    result=dict(status='COMPLETE_CPU_FACTORIAL_TRAIN_WORKER_AND_B1_SOURCE_CANARY',
        source_hashes=sources,science_sha256=sha256(SCIENCE),tests_passed=int(match.group(1)),
        test_log_sha256=sha256(log),prior_factorial_cpu_proof_sha256=sha256(work/'native_contrast_quality_factorial_CPU_proof.json'),
        actual_Train_input_content=reader.content_manifest(),records=records,
        unique_Train_RGB_frames=8,unique_current_November_IR_frames=8,total_actual_updates=4,
        B1_inference_frame_conditions=4,GPU_work=False,CUDA_initialized=False,
        new_accuracy=False,formal_fits_started=0,weights_saved=0,
        peak_process_RSS_GiB=peak,seconds=time.monotonic()-started,
        full_followup_pipeline_implemented=False,registered_GPU_queue=False,
        scope='Actual shared worker on two ordered Train clips per mode and two independently encoded full-resolution B1 Train frames per mode. CPU FP32 only; no GPU/FP16 parity, full Train fit, Dev/Test accuracy, runtime FPS or innovation. Full independent evaluation/controller/export remains required before registration.',
        goal_met=False,supported_innovations=0)
    dump(output,result);print(result['status'],result['tests_passed'],'tests',flush=True)


def gpu_worker(action):
    plan,study,protocol,calibration,work,checkpoint,base=context()
    authorization,arm,registration,physical,uuid=require_gpu(work)
    from tools.native_pyramid_experiment import load_teacher
    teacher=load_teacher(checkpoint,training=True)
    reader,reference=make_image_reader(work,plan,protocol,arm['Train_delivery']['lengths'],'ir')
    reader.max_bytes=2**30  # Bound encoded cache within the whole8GiB worker.
    if action=='gpu-probe':
        path=base/'engineering.json';assert not path.exists()
        row=canary(teacher,reader,plan,study,protocol,calibration,arm,'cuda',8)
        assert row['initial_native_sha256']==authorization['initial_native_sha256']
        dump(path,dict(status='COMPLETE_REAL_GPU_JOINT_FACTORIAL_ENGINEERING',record=row,
            authorization_sha256=sha256(base/'authorization.json'),registry_sha256=sha256(work/REGISTRY),
            physical_gpu=physical,GPU_uuid=uuid,input_content_reference=reference,
            peak_own_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
            formal_fit=False,goal_met=False,supported_innovations=0))
        return
    proof=read(base/'engineering.json')
    assert proof['status']=='COMPLETE_REAL_GPU_JOINT_FACTORIAL_ENGINEERING'
    assert proof['authorization_sha256']==sha256(base/'authorization.json')
    assert proof['registry_sha256']==sha256(work/REGISTRY)
    torch.manual_seed(0)
    module=NativePyramidFusion(**arm['module_kwargs']).cuda().float()
    initial=state_sha256(module.state_dict());assert initial==authorization['initial_native_sha256']
    optimizer=torch.optim.AdamW(module.parameters(),lr=arm['fit']['learning_rate'],weight_decay=arm['fit']['weight_decay'])
    provider=OrderedTrainPixels(plan,protocol,arm['Train_delivery']['lengths'],calibration,reader,arm['training_input_hw'],'cuda')
    metadata=fit_metadata(authorization,arm,registration,provider,initial,physical,uuid,reference)
    run_training_to_epoch(module,teacher.backbone,teacher.head,optimizer,provider,arm,metadata,
                          authorization['joint_arm_path'],base/'fit',authorization['stage'],work)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30;assert peak<=8
    dump(base/'fit_complete.json',dict(status='COMPLETE_ONE_JOINT_DISABLE_FULL_TRAIN_FIT',
        authorization_sha256=sha256(base/'authorization.json'),progress_sha256=sha256(base/'fit/progress.json'),
        stage=authorization['stage'],actual_input_content=reader.statistics(),peak_process_RSS_GiB=peak,
        peak_own_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,goal_met=False,supported_innovations=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['cpu-probe','gpu-probe','fit'])
    action=parser.parse_args().action
    cpu_probe() if action=='cpu-probe' else gpu_worker(action)
