#!/usr/bin/env python3
"""Actual one-arm Train/B1 CPU worker check, not formal GPU eligibility."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import torch

from tools.fusion_memory_evaluation import read,sha
from tools.fusion_memory_model_loading import load_source_rgb_cpu
from tools.fusion_memory_observation import write_json
from tools.fusion_memory_pipeline import current_source_hashes
from tools.fusion_memory_workflow import FusionMemoryWorkflow
from tools.native_pyramid_content_guard import PinnedTrainImageReader
from tools.native_pyramid_train_data import OrderedTrainPixels
from tools.neck72_runtime import require_host_memory,require_storage
from tools.task_utility_fusion import TaskUtilityFusion
from tools.task_utility_study import STUDY,context,make_arm,predecessor_ready,registered
from tools.task_utility_training_loop import state_sha256
from tools.train_task_utility import canary

SPEC=ROOT/'configs/perceptual_task_utility_worker_components_20261125.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pytest-receipt',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    started=time.monotonic();spec=read(SPEC);spec_sha=sha(SPEC);sources=spec['source_hashes']
    assert all(sha(ROOT/p)==s for p,s in sources.items())
    unit=read(args.pytest_receipt)
    assert unit['exit_code']==0 and unit['all_sources_unchanged'] and sha(unit['log'])==unit['log_sha256']
    assert '10 passed' in Path(unit['log']).read_text() and all(sources[p]==s for p,s in unit['source_hashes'].items())
    plan,study,protocol,calibration,work,checkpoint,formal_base=context();assert sha(STUDY)==spec['study_sha256']
    require_host_memory(8*2**30,maximum_workers=1);require_storage(work)
    old=FusionMemoryWorkflow().registered();assert old['source_hashes']==current_source_hashes() and len(old['source_hashes'])==384
    assert sha(work/'fusion_memory/pipeline_registered.json')==spec['F4_registration_sha256']
    try:
        registered(work)
    except (FileNotFoundError,ValueError) as error:
        eligibility=dict(complete_pipeline_registered=False,exception=repr(error))
    else:raise RuntimeError('This component probe expects the original incomplete pipeline state')
    eligibility['predecessor_ready']=predecessor_ready(work)
    base=Path(spec['output_directory']);base.mkdir(exist_ok=False)
    proc=Path('/proc/self');identity=dict(pid=os.getpid(),start_ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
        command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
    state=dict(status='RUNNING_REAL_TASK_UTILITY_WORKER_CPU_COMPONENT',identity=identity,specification_sha256=spec_sha,
        completed_optimizer_updates=0,completed_B1_inference_frames=0,events=[],formal_GPU_eligibility=False,new_GPU_work=False)
    def save():
        temporary=base/'state.tmp.json';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(base/'state.json')
    def progress(event):
        state['events'].append(event)
        if event['phase']=='optimizer':state['completed_optimizer_updates']+=int(event['row']['optimizer_updated'])
        if event['phase']=='inference':state['completed_B1_inference_frames']+=1
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('Own worker exceeded8GiB')
        save();print(json.dumps(dict(phase=event['phase'],updates=state['completed_optimizer_updates'],
            B1_frames=state['completed_B1_inference_frames'])),flush=True)
    save()
    try:
        prototype=read(ROOT/'configs/perceptual_task_utility_prototype_20261125.json')
        manifest=prototype['completed_IR_content_manifest'];assert sha(manifest['path'])==manifest['sha256']
        content={(r['sequence'],r['frame'],r['modality']):r for r in read(manifest['path'])}
        reader=PinnedTrainImageReader(plan['data_root'],plan['generated_ir_root'],protocol['train'],expected_content=content,
            max_bytes=64*2**20,reserve_bytes=40*2**30)
        provider=OrderedTrainPixels(plan,protocol,study['Train_delivery']['lengths'],calibration,reader,study['training_input_hw'],'cpu')
        arm=make_arm(study,spec['fit_id'],spec['variant'],spec['cue']);write_json(base/'CPU_only_arm.json',arm,exclusive=True)
        teacher=load_source_rgb_cpu(checkpoint,study['source_RGB_sha256']);frozen=state_sha256(teacher.state_dict())
        torch.manual_seed(0);module=TaskUtilityFusion(**arm['module_kwargs']).float()
        assert sum(p.numel() for p in module.parameters())==553705
        assert state_sha256(module.state_dict())==study['architecture']['initial_state_sha256']
        record=canary(module,teacher,provider,arm,inference_frames=12,progress=progress)
        assert state['completed_optimizer_updates']==3 and state['completed_B1_inference_frames']==12
        assert state_sha256(teacher.state_dict())==frozen and all(p.grad is None for p in teacher.parameters())
        images=reader.content_manifest();assert len(images)==24
        assert all(r['sha256']==content[r['sequence'],r['frame'],r['modality']]['sha256'] for r in images)
        assert sha(SPEC)==spec_sha and all(sha(ROOT/p)==s for p,s in sources.items())
        assert FusionMemoryWorkflow().registered()==old and old['source_hashes']==current_source_hashes()
        assert not torch.cuda.is_initialized();require_storage(work)
        value=dict(status='COMPLETE_ONE_IR_TASK_UTILITY_WORKER_TRAIN_B1_CPU_NOT_GPU_ELIGIBILITY',identity=identity,
            specification_sha256=spec_sha,study_sha256=sha(STUDY),source_hashes=sources,pytest_receipt=str(args.pytest_receipt.resolve()),
            pytest_receipt_sha256=sha(args.pytest_receipt),fit_id=arm['fit_id'],actual_record=record,
            original_RGB_state_sha256=frozen,actual_Train_frames=12,Train_domains=1,actual_optimizer_updates=3,
            actual_B1_inference_frames=12,complete_Train_epochs=0,Train_GT_hashes=provider.gt_hashes,
            image_content_records=images,reader_statistics=reader.statistics(),eligibility_preflight=eligibility,
            actual_device='cpu',new_GPU_work=False,formal_GPU_eligibility=False,full_workflow_registered=False,
            original_RGB_weights_BN_gradients_unchanged=True,saved_weights=False,new_image_FPN_disk_cache=False,
            new_Dev_Test_accuracy=False,goal_met=False,supported_innovations=0,
            seconds=time.monotonic()-started,peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
            scope='One IR joint-supervision arm through actual newworker three T4 Train updates plus12B1frames,source/zero/missing/NMS/ByteTrack/geometry/image-roll checks. Not12armGPU qualification,completeTrain/formalworkflow,officialaccuracy,causalefficacy or individualFPS. Earlier CPU numerical anomaly cause remains unknown.')
        write_json(base/'complete.json',value,exclusive=True)
        state.update(status=value['status'],result_sha256=sha(base/'complete.json'));save()
        print(json.dumps(dict(status=value['status'],seconds=value['seconds'],peak_process_RSS_GiB=value['peak_process_RSS_GiB'])),flush=True)
    except Exception as error:
        state.update(status='FAILED_REAL_TASK_UTILITY_WORKER_CPU_COMPONENT_PRESERVE',error=repr(error));save();raise


if __name__=='__main__':main()
