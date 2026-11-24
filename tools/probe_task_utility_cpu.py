#!/usr/bin/env python3
"""Twelve matched six-update Train engineering checks, no formal fit or efficacy."""
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
import numpy as np
import torch

from tools.content_verified_train_reader import ContentVerifiedTrainImageReader
from tools.fusion_memory_evaluation import read,sha
from tools.fusion_memory_model_loading import load_source_rgb_cpu
from tools.fusion_memory_observation import write_json
from tools.fusion_memory_workflow import FusionMemoryWorkflow
from tools.native_pyramid_train_data import preprocess_observation
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory,require_storage
from tools.task_utility_fusion import TaskUtilityFusion
from tools.task_utility_objectives import task_supervision
from yolox.models.frozen_head_repair import head_predictions,original_supervised_loss,preserve_reference

SPEC=ROOT/'configs/perceptual_task_utility_prototype_20261125.json'


def direct_gradient(loss,parameters):
    gradients=torch.autograd.grad(loss,tuple(parameters),retain_graph=True,allow_unused=True)
    return sum(float(g.detach().abs().sum()) for g in gradients if g is not None)


def gradient(module,fragment):
    return sum(float(p.grad.detach().abs().sum()) for n,p in module.named_parameters() if fragment in n and p.grad is not None)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pytest-receipt',type=Path,required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    started=time.monotonic();spec=read(SPEC);spec_sha=sha(SPEC);sources=spec['source_hashes']
    assert all(sha(ROOT/p)==s for p,s in sources.items())
    tests=read(args.pytest_receipt)
    assert tests['exit_code']==0 and tests['all_sources_unchanged'] and sha(tests['log'])==tests['log_sha256']
    assert '12 passed' in Path(tests['log']).read_text()
    assert all(sources[p]==s for p,s in tests['source_hashes'].items())
    plan_path=ROOT/'configs/perceptual_mft_20261122.json';assert sha(plan_path)==spec['main_plan_sha256']
    plan=read(plan_path);protocol_path=ROOT/plan['protocol'];assert sha(protocol_path)==spec['protocol_sha256'];protocol=read(protocol_path)
    work=Path(plan['work_root']);require_host_memory(4*2**30,maximum_workers=1);require_storage(work)
    registered=FusionMemoryWorkflow().registered();registration_sha=sha(work/'fusion_memory/pipeline_registered.json')
    assert len(registered['source_hashes'])==384
    base=Path(spec['output_directory']);base.mkdir(exist_ok=False)
    process=Path('/proc/self');identity=dict(pid=os.getpid(),start_ticks=int((process/'stat').read_text().rsplit(')',1)[1].split()[19]),
        command_sha256=hashlib.sha256((process/'cmdline').read_bytes()).hexdigest())
    state=dict(status='RUNNING_TASK_UTILITY_CPU_ENGINEERING_NOT_FORMAL_FIT',identity=identity,completed_conditions=[],
        completed_optimizer_updates=0,specification_sha256=spec_sha,new_GPU_work=False)
    def save():
        temp=base/'state.tmp.json';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(base/'state.json')
    def guard():
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*2**30
    save();calibration=read(spec['calibration']);assert sha(spec['calibration'])==spec['calibration_sha256']
    teacher=load_source_rgb_cpu(spec['source_RGB_checkpoint'],spec['source_RGB_sha256']);frozen=state_sha256(teacher.state_dict())
    reader=ContentVerifiedTrainImageReader(plan['data_root'],plan['generated_ir_root'],protocol['train'],max_bytes=64*2**20,reserve_bytes=40*2**30)
    content_info=spec['completed_IR_content_manifest'];assert sha(content_info['path'])==content_info['sha256']
    content={(r['sequence'],r['frame'],r['modality']):r for r in read(content_info['path'])}
    cached=[];gt_hashes={}
    for address in spec['CPU']['addresses']:
        sequence,frame=address['sequence'],address['frame'];assert sequence in protocol['train']
        rgb,ir=reader.read(sequence,frame,'rgb'),reader.read(sequence,frame,'ir')
        gt_path=Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        assert sha(gt_path)==protocol['derived_gt_sha256']['train/'+sequence];gt_hashes[sequence]=sha(gt_path)
        gt=np.loadtxt(gt_path,delimiter=',',ndmin=2);gt=gt[(gt[:,0]==frame)&(gt[:,6]>0)&(gt[:,7]==1)]
        pixels={}
        for cue in spec['matched_cues']:
            value,auxiliary,valid,_,ratio,present=preprocess_observation(rgb,ir if cue=='ir' else None,cue,spec['CPU']['input_hw'],calibration)
            assert present;pixels[cue]=torch.from_numpy(auxiliary[None].copy())
        boxes=gt[:,2:6].astype(np.float32).copy();boxes[:,:2]+=boxes[:,2:]/2;boxes*=ratio
        targets=torch.zeros(1,500,5);assert 0<len(boxes)<=500;targets[0,:len(boxes),1:]=torch.from_numpy(boxes)
        with torch.no_grad():
            features=tuple(v.detach() for v in teacher.backbone(torch.from_numpy(value[None].astype(np.float32))))
            reference,metadata=head_predictions(teacher.head,features)
        roll=np.roll(ir,(ir.shape[0]//4,ir.shape[1]//4),(0,1));roll_pixels=preprocess_observation(rgb,roll,'ir',spec['CPU']['input_hw'],calibration)[1]
        cached.append(dict(address=address,pixels=pixels,valid=torch.from_numpy(valid[None].copy()),targets=targets,
            features=features,reference=reference,metadata=metadata,IR_roll=torch.from_numpy(roll_pixels[None].copy())))
        guard();print(json.dumps(dict(prepared=address,GT_boxes=len(boxes))),flush=True)
    assert len(cached)==6 and set(gt_hashes)==set(protocol['train'])
    image_records=reader.content_manifest();assert len(image_records)==12
    assert all(r['sha256']==content[r['sequence'],r['frame'],r['modality']]['sha256'] for r in image_records)
    conditions=[];initials=set();counts=set()
    assert len({json.dumps(v['kwargs'],sort_keys=True) for v in spec['variants']})==1
    for variant in spec['variants']:
        for cue in spec['matched_cues']:
            torch.manual_seed(0);module=TaskUtilityFusion(**variant['kwargs']).float().train()
            count=sum(p.numel() for p in module.parameters());assert count==spec['expected_parameters'];counts.add(count)
            initials.add(state_sha256(module.state_dict()));optimizer=torch.optim.AdamW(module.parameters(),lr=spec['CPU']['learning_rate'],weight_decay=spec['CPU']['weight_decay'])
            steps=[]
            for position,data in enumerate(cached):
                output,_,diagnostics=module(data['features'],data['pixels'][cue],data['metadata']['scores'],valid=data['valid'],return_proposals=True)
                predictions,_=head_predictions(teacher.head,output)
                if position==0:assert torch.equal(predictions,data['reference'])
                with torch.no_grad():proposal,_=head_predictions(teacher.head,diagnostics.pop('detached_proposal_features'))
                foreground,utility,labels=task_supervision(teacher.head,data['reference'],proposal,data['metadata'],data['targets'],diagnostics,
                    spec['CPU']['input_hw'],temperature=spec['CPU']['utility_temperature'])
                original=original_supervised_loss(teacher.head,predictions,data['metadata'],data['targets'],spec['CPU']['input_hw'])[0]
                preservation=preserve_reference(predictions,data['reference'],data['metadata']['strides'],reliable_only=True)
                fg_weight=spec['CPU']['lambda_foreground'] if variant['supervise_foreground'] else 0.
                utility_weight=spec['CPU']['lambda_utility'] if variant['supervise_utility'] else 0.
                fg_gradient=direct_gradient(fg_weight*foreground,(p for t in module.task for p in t.foreground.parameters()))
                utility_gradient=direct_gradient(utility_weight*utility,(p for t in module.task for p in t.utility.parameters()))
                loss=original+spec['CPU']['lambda_reliable_RGB']*preservation+fg_weight*foreground+utility_weight*utility
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(module.parameters(),spec['CPU']['gradient_clip'],error_if_nonfinite=True));optimizer.step()
                steps.append(dict(address=data['address'],loss=float(loss.detach()),original_YOLOX=float(original.detach()),
                    foreground_loss=float(foreground.detach()),utility_loss=float(utility.detach()),
                    weighted_foreground_direct_gradient_L1=fg_gradient,weighted_utility_direct_gradient_L1=utility_gradient,
                    gradient_norm=norm,native_stem_gradient_L1=gradient(module,'encoder.blocks.0.0'),
                    decoded_box_max_change=float((predictions[:,:,:4]-data['reference'][:,:,:4]).detach().abs().max()),
                    local_utility_labels=labels,foreground_gate_std=[float(r['foreground_logits'].detach().std()) for r in diagnostics['levels']]))
                assert bool(torch.isfinite(loss)) and all(p.grad is None for p in teacher.parameters())
                guard();state['completed_optimizer_updates']+=1;save()
            assert any(r['decoded_box_max_change']>0 for r in steps[1:])
            if variant['supervise_foreground']:assert any(r['weighted_foreground_direct_gradient_L1']>0 for r in steps)
            else:assert all(r['weighted_foreground_direct_gradient_L1']==0 for r in steps)
            if variant['supervise_utility']:assert any(r['weighted_utility_direct_gradient_L1']>0 for r in steps[1:])
            else:assert all(r['weighted_utility_direct_gradient_L1']==0 for r in steps)
            assert all(r['weighted_utility_direct_gradient_L1']==0 for r in steps[:1])
            module.eval().requires_grad_(False);checks=[]
            with torch.no_grad():
                for data in cached:
                    missing=module(data['features'],None,data['metadata']['scores'],available=False)[0]
                    missing_raw,_=head_predictions(teacher.head,missing);assert torch.equal(missing_raw,data['reference'])
                    checks.append(dict(address=data['address'],missing_raw_head_bit_exact=True))
                data=cached[0];normal=module(data['features'],data['pixels'][cue],data['metadata']['scores'],valid=data['valid'])[0]
                normal_raw,_=head_predictions(teacher.head,normal);roll_changed=None
                if cue=='ir':
                    rolled=module(data['features'],data['IR_roll'],data['metadata']['scores'],valid=data['valid'])[0]
                    roll_raw,_=head_predictions(teacher.head,rolled);roll_changed=not torch.equal(normal_raw,roll_raw)
                    assert roll_changed
            assert state_sha256(teacher.state_dict())==frozen
            identifier=variant['id']+'_'+cue
            conditions.append(dict(id=identifier,variant=variant,cue=cue,seed=0,initial_raw_RGB_exact=True,steps=steps,
                missing_checks=checks,actual_IR_image_roll_changes_raw_head=roll_changed,formal_fit=False,tracking_accuracy_measured=False))
            state['completed_conditions'].append(identifier);save();print(json.dumps(dict(completed=identifier,updates=6)),flush=True)
    assert len(initials)==len(counts)==1 and len(conditions)==12 and not torch.cuda.is_initialized()
    assert sha(SPEC)==spec_sha and all(sha(ROOT/p)==s for p,s in sources.items())
    assert FusionMemoryWorkflow().registered()==registered
    require_storage(work);guard()
    result=dict(status='COMPLETE_ACTUAL_TASK_UTILITY_CPU_ENGINEERING_NOT_FORMAL_FIT_OR_GAIN',identity=identity,
        specification_sha256=spec_sha,source_hashes=sources,pytest_receipt=str(args.pytest_receipt.resolve()),pytest_receipt_sha256=sha(args.pytest_receipt),
        previous_F4_registration_sha256=registration_sha,actual_conditions=conditions,common_initial_state_sha256=next(iter(initials)),
        parameters_each=next(iter(counts)),actual_Train_frames=6,Train_domains=6,optimizer_updates_total=72,complete_Train_epochs=0,
        image_content_records=image_records,Train_GT_hashes=gt_hashes,source_RGB_state_sha256=frozen,
        original_RGB_weights_BN_and_gradients_unchanged=True,past_or_future_frame_input=False,
        actual_device='cpu',new_GPU_work=False,saved_weights=False,new_image_FPN_ROI_disk_cache=False,
        formal_launch_ready=False,new_Dev_Test_accuracy=False,goal_met=False,supported_innovations=0,
        seconds=time.monotonic()-started,peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        scope='Twelve engineering conditions,6Trainframes/6updates each. Dense fixed-reference task-loss labels are not causal efficacy/calibrated reliability. No formal fit, convergence, heldout accuracy or innovation evidence.')
    write_json(base/'complete.json',result,exclusive=True);state.update(status=result['status'],result_sha256=sha(base/'complete.json'));save()
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],peak_process_RSS_GiB=result['peak_process_RSS_GiB'])),flush=True)


if __name__=='__main__':main()
