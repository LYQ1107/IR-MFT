#!/usr/bin/env python3
"""Real ordered T4 clips through the new optimizer API; CPU engineering only."""
import argparse
import copy
import gc
import hashlib
import itertools
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
from tools.fusion_memory_workflow import FusionMemoryWorkflow
from tools.native_pyramid_content_guard import PinnedTrainImageReader
from tools.native_pyramid_train_data import OrderedTrainPixels,epoch_clips,address_bytes
from tools.neck72_runtime import require_host_memory,require_storage
from tools.task_utility_fit import optimizer_step
from tools.task_utility_fusion import TaskUtilityFusion
from tools.task_utility_training_loop import state_sha256

SPEC=ROOT/'configs/perceptual_task_utility_training_components_20261125.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pytest-receipt',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    started=time.monotonic();spec=read(SPEC);spec_sha=sha(SPEC)
    assert all(sha(ROOT/p)==s for p,s in spec['source_hashes'].items())
    unit=read(args.pytest_receipt)
    assert unit['exit_code']==0 and unit['all_sources_unchanged'] and sha(unit['log'])==unit['log_sha256']
    assert '6 passed' in Path(unit['log']).read_text()
    assert all(spec['source_hashes'][p]==s for p,s in unit['source_hashes'].items())
    prototype=read(ROOT/spec['prototype']['path']);assert sha(ROOT/spec['prototype']['path'])==spec['prototype']['sha256']
    proof_path=ROOT/spec['prototype_proof']['path'];proof=read(proof_path)
    assert sha(proof_path)==spec['prototype_proof']['sha256'] and proof['optimizer_updates_total']==72
    assert all(sha(ROOT/p)==s for p,s in proof['source_hashes'].items())
    plan_path=ROOT/'configs/perceptual_mft_20261122.json';assert sha(plan_path)==spec['main_plan_sha256']
    plan=read(plan_path);protocol_path=ROOT/plan['protocol'];assert sha(protocol_path)==spec['protocol_sha256']
    protocol=read(protocol_path);work=Path(plan['work_root'])
    require_host_memory(8*2**30,maximum_workers=1);require_storage(work)
    registration=FusionMemoryWorkflow().registered();assert len(registration['source_hashes'])==384
    base=Path(spec['output_directory']);base.mkdir(exist_ok=False)
    process=Path('/proc/self');identity=dict(pid=os.getpid(),start_ticks=int((process/'stat').read_text().rsplit(')',1)[1].split()[19]),
        command_sha256=hashlib.sha256((process/'cmdline').read_bytes()).hexdigest())
    state=dict(status='RUNNING_ACTUAL_T4_TASK_SUPERVISION_CPU_API_CHECK',identity=identity,
        specification_sha256=spec_sha,completed_optimizer_updates=0,completed_conditions=[],new_GPU_work=False)
    def save():
        temp=base/'state.tmp.json';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(base/'state.json')
    def guard():
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*2**30
    save();calibration=read(prototype['calibration']);assert sha(prototype['calibration'])==prototype['calibration_sha256']
    teacher=load_source_rgb_cpu(prototype['source_RGB_checkpoint'],prototype['source_RGB_sha256'])
    frozen=state_sha256(teacher.state_dict())
    manifest=prototype['completed_IR_content_manifest'];assert sha(manifest['path'])==manifest['sha256']
    content={(r['sequence'],r['frame'],r['modality']):r for r in read(manifest['path'])}
    lengths=spec['training']['Train_delivery']['lengths'];assert len(content)==2*sum(lengths.values())
    assert set(content)=={(s,f,m) for s,n in lengths.items() for f in range(1,n+1) for m in ['rgb','ir']}
    reader=PinnedTrainImageReader(plan['data_root'],plan['generated_ir_root'],protocol['train'],
        expected_content=content,max_bytes=64*2**20,reserve_bytes=40*2**30)
    addresses=list(itertools.islice(epoch_clips(lengths,1),spec['CPU']['clips_per_condition']))
    assert [dict(sequence=c.sequence,frames=list(c.frames)) for c in addresses]==spec['CPU']['ordered_clips']
    rows=[];initials=set();gt_hashes=None
    for variant in prototype['variants']:
        for cue in prototype['matched_cues']:
            torch.manual_seed(0);module=TaskUtilityFusion(**variant['kwargs']).float().train()
            assert sum(p.numel() for p in module.parameters())==spec['parameters_each']
            initials.add(state_sha256(module.state_dict()))
            optimizer=torch.optim.AdamW(module.parameters(),lr=spec['training']['fit']['learning_rate'],
                weight_decay=spec['training']['fit']['weight_decay'])
            config=copy.deepcopy(spec['training']);config['task_supervision'].update(
                foreground=variant['supervise_foreground'],utility=variant['supervise_utility'])
            provider=OrderedTrainPixels(plan,protocol,lengths,calibration,reader,config['training_input_hw'],'cpu')
            if gt_hashes is not None:assert gt_hashes==provider.gt_hashes
            gt_hashes=provider.gt_hashes;provider.begin_epoch(1);steps=[]
            for position,address in enumerate(addresses):
                batch=provider.prepare(address,cue)
                assert all(batch.available) and len(batch.available)==4
                result,temporal,previous=optimizer_step(module,teacher.backbone,teacher.head,batch,optimizer,config,position)
                assert temporal is previous is None and result['optimizer_updated']
                assert result['foreground_supervision_direct_gradient_L1'] is not None
                expected=result['full_YOLOX']+.02*result['reliable_RGB']
                expected+=.1*variant['supervise_foreground']*result['foreground']+.1*variant['supervise_utility']*result['utility']
                assert abs(expected-result['loss'])<=2e-6*max(1.,abs(expected))
                if position==0:
                    assert result['initial_full_RGB_exact'] and result['initial_proposal_full_RGB_exact']
                    assert result['utility_label_mass']==result['utility_supervision_direct_gradient_L1']==0
                steps.append(dict(sequence=address.sequence,frame_indices=list(address.frames),**result))
                state['completed_optimizer_updates']+=1;save();guard()
                print(json.dumps(dict(condition=variant['id']+'_'+cue,position=position,row=result)),flush=True)
            fg=sum(v['foreground_supervision_direct_gradient_L1'] for v in steps)
            utility=sum(v['utility_supervision_direct_gradient_L1'] for v in steps)
            assert (fg>0)==variant['supervise_foreground']
            assert (utility>0)==variant['supervise_utility']
            assert any(v['decoded_box_max_change']>0 for v in steps[1:])
            missing=copy.copy(batch);missing.available=(False,)*4
            before=state_sha256(module.state_dict());moments=copy.deepcopy(optimizer.state_dict()['state'])
            missing_row,_,_=optimizer_step(module,teacher.backbone,teacher.head,missing,optimizer,config,len(addresses))
            assert not missing_row['optimizer_updated'] and missing_row['decoded_box_max_change']==0
            assert missing_row['valid_task_cells']==missing_row['utility_label_mass']==0
            assert state_sha256(module.state_dict())==before
            for key,value in moments.items():
                for name,tensor in value.items():assert torch.equal(tensor,optimizer.state_dict()['state'][key][name])
            assert state_sha256(teacher.state_dict())==frozen
            identifier=variant['id']+'_'+cue
            rows.append(dict(id=identifier,cue=cue,supervision=config['task_supervision'],steps=steps,
                delivered_Train_clips=provider.prepared,actual_address_prefix_sha256=provider.digest.hexdigest(),
                all_missing_clip=missing_row,all_missing_skips_AdamW_with_existing_moments_exact=True,
                formal_fit=False,complete_Train_epochs=0))
            state['completed_conditions'].append(identifier);save()
            del module,optimizer,provider,batch,missing,moments;gc.collect();guard()
    assert len(initials)==1 and len(rows)==12 and state['completed_optimizer_updates']==36
    records=reader.content_manifest();assert len(records)==24
    assert all(v['sha256']==content[v['sequence'],v['frame'],v['modality']]['sha256'] for v in records)
    assert sha(SPEC)==spec_sha and all(sha(ROOT/p)==s for p,s in spec['source_hashes'].items())
    assert FusionMemoryWorkflow().registered()==registration and not torch.cuda.is_initialized()
    require_storage(work);guard()
    result=dict(status='COMPLETE_ACTUAL_T4_TASK_SUPERVISION_API_CPU_NOT_FORMAL_FIT_OR_GAIN',identity=identity,
        specification_sha256=spec_sha,source_hashes=spec['source_hashes'],pytest_receipt=str(args.pytest_receipt.resolve()),
        pytest_receipt_sha256=sha(args.pytest_receipt),conditions=rows,common_initial_state_sha256=next(iter(initials)),
        source_RGB_state_sha256=frozen,Train_GT_hashes=gt_hashes,image_content_records=records,reader_statistics=reader.statistics(),
        parameters_each=spec['parameters_each'],optimizer_updates_total=36,missing_clip_optimizer_calls=12,
        actual_unique_Train_frames=12,Train_domains=1,actual_Train_frame_visits=144,all_missing_frame_rechecks=48,complete_Train_epochs=0,
        actual_device='cpu',new_GPU_work=False,formal_launch_ready=False,new_Dev_Test_accuracy=False,
        source_RGB_weights_BN_and_gradients_unchanged=True,saved_weights=False,new_image_FPN_disk_cache=False,
        seconds=time.monotonic()-started,peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        original_F4_registration_sha256=sha(work/'fusion_memory/pipeline_registered.json'),goal_met=False,supported_innovations=0,
        scope='Three real chronological T4 clips in one Train video per condition, full production input resolution and LR warmup; original encoder and head are invoked in each new optimizer call. All-missing calls repeat the final clip. This is not complete Train, formal-loop delivery/resume on actual full data, GPU qualification, NMS/ByteTrack verification, convergence or accuracy evidence.')
    write_json(base/'complete.json',result,exclusive=True)
    state.update(status=result['status'],result_sha256=sha(base/'complete.json'));save()
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],peak_process_RSS_GiB=result['peak_process_RSS_GiB'])),flush=True)


if __name__=='__main__':main()
