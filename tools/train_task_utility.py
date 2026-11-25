#!/usr/bin/env python3
"""Independent task-supervision worker; complete release/registration before CUDA."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from tools.native_pyramid_experiment import gpu_settings,load_teacher,read
from tools.native_pyramid_content_guard import make_image_reader
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line
from tools.native_pyramid_train_data import OrderedTrainPixels,epoch_clips,preprocess_observation
from tools.neck72_runtime import require_host_memory,require_storage
from tools.probe_fusion_memory import state_digest
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.task_utility_fit import optimizer_step
from tools.task_utility_fusion import TaskUtilityFusion
from tools.task_utility_observer import TaskUtilityObserver
from tools.task_utility_study import STUDY,REGISTRY,arm_path,computation_key,context,predecessor_ready,registered,validate_history
from tools.task_utility_training_loop import run_training_to_epoch,state_sha256


def array_sha(value):
    value=np.ascontiguousarray(value);digest=hashlib.sha256()
    digest.update(str(value.shape).encode());digest.update(str(value.dtype).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def tensor_sha(value):return array_sha(value.detach().cpu().numpy())


@torch.no_grad()
def inference_canary(module,teacher,reader,calibration,sequence,arm,frames,progress=None):
    """Current image interface only; no targets/provider/GT argument."""
    from tools.extract_structural_relative import detections
    from yolox.utils import postprocess
    module.eval().requires_grad_(False);device=next(teacher.parameters()).device
    if device.type=='cuda':teacher.head.half()
    frozen=state_sha256(teacher.state_dict());auxiliary_before=state_sha256(module.state_dict())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0);zero=TaskUtilityFusion(**arm['module_kwargs']).to(device).float().eval().requires_grad_(False)
    rows=[dict(id='RGB',cue='rgb',baseline=True),dict(id='trained',cue=arm['cue']),
        dict(id='zero',cue=arm['cue'],expect_RGB_exact=True),dict(id='missing',cue='ir',intervention='unavailable')]
    modules=dict(RGB=None,trained=module,zero=zero,missing=copy.deepcopy(module))
    if arm['cue']=='ir':
        rows.append(dict(id='roll',cue='ir',intervention='IR_image_spatial_roll'));modules['roll']=copy.deepcopy(module)
    rows.append(dict(id='geometry',cue=arm['cue'],geometry_fixed=True));modules['geometry']=copy.deepcopy(module)
    engine=OnlineNativeDetector(teacher,rows,modules,arm['inference_input_hw'],calibration)
    observer=TaskUtilityObserver();encoder_records={};handles=[];capture=dict(enabled=True,head=[])
    for name,current in modules.items():
        if current is None:continue
        observer.attach(name,current)
        def encoded(net,args,output,name=name):
            latent,_,_=output
            encoder_records[name]=dict(cue_sha256=tensor_sha(args[0]),latent_sha256=[tensor_sha(v) for v in latent])
        handles.append(current.encoder.register_forward_hook(encoded))
    def decoded(net,args,output):
        if capture['enabled']:capture['head'].append(output.detach().clone())
    handles.append(teacher.head.register_forward_hook(decoded))
    trackers={r['id']:IndependentByteTrack() for r in rows};trackers['direct']=IndependentByteTrack()
    mot_hashes={k:hashlib.sha256() for k in trackers};checks=[]
    nonbaseline=[r['id'] for r in rows if not r.get('baseline')]
    try:
        for frame in range(1,frames+1):
            rgb=reader.read(sequence,frame,'rgb');ir=reader.read(sequence,frame,'ir') if engine.needs_actual_IR_image else None
            encoder_records.clear();capture.update(enabled=True,head=[])
            values,information=engine.step(rgb,ir,sequence,frame);raw=capture['head']
            if len(raw)!=1+len(nonbaseline):raise ValueError('Every actual B1 source and repaired head must be observed')
            raw_by_name=dict(zip(nonbaseline,raw[1:]));capture['enabled']=False
            pixels,auxiliary,_,_,ratio,_=preprocess_observation(rgb,ir,arm['cue'],arm['inference_input_hw'],calibration)
            direct_raw=teacher(torch.from_numpy(pixels[None]).to(device=device,dtype=next(teacher.parameters()).dtype))
            if not torch.equal(direct_raw,raw[0]):raise RuntimeError('Independent direct RGB raw head differs')
            if any(not torch.equal(raw_by_name[name],direct_raw) for name in ['zero','missing']):
                raise RuntimeError('Observed zero/missing raw head differs from independent direct RGB')
            direct=detections(postprocess(direct_raw.clone(),1,.1,.7)[0],ratio)
            for name in ['RGB','zero','missing']:np.testing.assert_array_equal(values[name],direct)
            if encoder_records['trained']['cue_sha256']!=array_sha(auxiliary[None]):
                raise RuntimeError('Native cue encoder did not consume the independently prepared current image')
            geometry=torch.cat((raw[0][:,:,:4],raw_by_name['geometry'][:,:,4:]),-1)
            np.testing.assert_array_equal(values['geometry'],detections(postprocess(geometry.clone(),1,.1,.7)[0],ratio))
            roll=None
            if 'roll' in modules:
                moved=np.roll(ir,(ir.shape[0]//4,ir.shape[1]//4),(0,1))
                expected=preprocess_observation(rgb,moved,'ir',arm['inference_input_hw'],calibration)[1]
                if encoder_records['roll']['cue_sha256']!=array_sha(expected[None]):raise RuntimeError('Wrong IR image intervention')
                if (encoder_records['roll']['cue_sha256']==encoder_records['trained']['cue_sha256'] or
                        encoder_records['roll']['latent_sha256']==encoder_records['trained']['latent_sha256']):
                    raise RuntimeError('Actual IR roll must change input and at least one learned cue feature')
                roll=dict(actual_input_and_latent_changed=True,
                    raw_head_changed=not torch.equal(raw_by_name['roll'],raw_by_name['trained']),
                    raw_head_max_abs=float((raw_by_name['roll']-raw_by_name['trained']).abs().max()),
                    packet_changed=not np.array_equal(values['roll'],values['trained']))
            current_tracks={name:[mot_line(v) for v in tracker.update(direct if name=='direct' else values[name],frame)]
                for name,tracker in trackers.items()}
            for name,lines in current_tracks.items():
                for line in lines:mot_hashes[name].update(line.encode())
            for name in ['RGB','zero','missing']:
                if (current_tracks[name]!=current_tracks['direct'] or trackers[name].counter!=trackers['direct'].counter or
                        state_digest(trackers[name].tracker)!=state_digest(trackers['direct'].tracker)):
                    raise RuntimeError('Zero/missing source tracking emission,ID or KF state differs')
            checks.append(dict(sequence=sequence,frame=frame,direct_zero_missing_raw_head_and_NMS_exact=True,
                direct_zero_missing_MOT_ID_KF_exact=True,geometry_packet_matches_independent_coordinate_intervention=True,
                actual_cue_input_sha256=encoder_records['trained']['cue_sha256'],IR_roll=roll,
                observed_head_shapes=[list(v.shape) for v in raw],detections={k:len(v) for k,v in values.items()},
                GT_read=False,future_frame_input=False))
            if progress is not None:progress(dict(phase='inference',row=checks[-1]))
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('Own worker exceeded8GiB')
        summaries={name:observer.summary(name) for name in observer.rows}
    finally:
        observer.close()
        for handle in handles:handle.remove()
    if state_sha256(teacher.state_dict())!=frozen or state_sha256(module.state_dict())!=auxiliary_before:
        raise RuntimeError('Frozen inference source or auxiliary weights/BN changed')
    return dict(B1_frames=frames,checks=checks,gate_statistics=summaries,
        MOT_prefix_sha256={k:v.hexdigest() for k,v in mot_hashes.items()},
        actual_device=str(device),raw_precision=str(next(teacher.parameters()).dtype),
        frozen_inference_states_exact=True,GT_read=False,future_frame_input=False,
        scope='Actual current-image B1/NMS/ByteTrack prefix, not complete sequences,official metrics,causal gain or independentFPS.')


def canary(module,teacher,provider,arm,inference_frames=12,progress=None):
    frozen=dict(encoder=state_sha256(teacher.backbone.state_dict()),head=state_sha256(teacher.head.state_dict()))
    config=copy.deepcopy(arm);config['audit_supervision_gradients']=True
    optimizer=torch.optim.AdamW(module.parameters(),lr=arm['fit']['learning_rate'],weight_decay=arm['fit']['weight_decay'])
    provider.begin_epoch(1);clips=list(epoch_clips(provider.lengths,1))[:3];rows=[]
    if len(clips)!=3 or [f for c in clips for f in c.frames]!=list(range(1,13)) or len({c.sequence for c in clips})!=1:
        raise ValueError('The registered first three ordered T4 clips required')
    for position,clip in enumerate(clips):
        batch=provider.prepare(clip,arm['cue'])
        result,state,previous=optimizer_step(module,teacher.backbone,teacher.head,batch,optimizer,config,position)
        if state is not None or previous is not None or not result['optimizer_updated']:raise RuntimeError('Current-frame optimizer path failed')
        rows.append(dict(sequence=clip.sequence,frame_indices=list(clip.frames),**result))
        if progress is not None:progress(dict(phase='optimizer',row=rows[-1]))
    for flag,key in [('foreground','foreground_supervision_direct_gradient_L1'),('utility','utility_supervision_direct_gradient_L1')]:
        if (sum(r[key] for r in rows)>0)!=arm['task_supervision'][flag]:raise RuntimeError('Auxiliary supervision gradient switch failed')
    if rows[0]['utility_supervision_direct_gradient_L1']!=0:raise RuntimeError('Zero proposal cannot create a utility update')
    before=state_sha256(module.state_dict());moments=copy.deepcopy(optimizer.state_dict()['state'])
    missing=copy.copy(batch);missing.available=(False,)*len(batch.available)
    unavailable,_,_=optimizer_step(module,teacher.backbone,teacher.head,missing,optimizer,config,3)
    if (unavailable['optimizer_updated'] or not unavailable['initial_full_RGB_exact'] or
            not unavailable['initial_proposal_full_RGB_exact'] or state_sha256(module.state_dict())!=before):
        raise RuntimeError('All-missing training clip failed exact RGB/skip checks')
    for key,value in moments.items():
        for name,tensor in value.items():
            if not torch.equal(tensor,optimizer.state_dict()['state'][key][name]):raise RuntimeError('Missing clip changed AdamW moments')
    if progress is not None:progress(dict(phase='all_missing_training',row=unavailable))
    if frozen!=dict(encoder=state_sha256(teacher.backbone.state_dict()),head=state_sha256(teacher.head.state_dict())):
        raise RuntimeError('Original RGB training source changed')
    online=inference_canary(module,teacher,provider.reader,provider.calibration,clips[0].sequence,arm,inference_frames,progress=progress)
    return dict(optimizer_rows=rows,actual_updates=3,delivered_Train_frames=12,complete_Train_epochs=0,
        all_missing_training=unavailable,existing_AdamW_moments_unchanged=True,
        diagnostic_only_override=dict(audit_supervision_gradients=True),inference=online,saved_weights=False,
        source_training_state_unchanged_before_explicit_CUDA_head_precision_conversion=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['probe','fit'])
    parser.add_argument('--fit-id',required=True);parser.add_argument('--through-epoch',type=int,choices=[6,9,12]);args=parser.parse_args()
    plan,study,protocol,calibration,work,checkpoint,base=context();registration=registered(work)
    if not predecessor_ready(work):raise RuntimeError('Wait for full F4 export and actual original process exits before CUDA')
    path=arm_path(base,args.fit_id);arm=read(path)
    if (registration['normal_arms'].get(args.fit_id)!=sha256(path) or arm['fit_id']!=args.fit_id or
            arm['parent_study_sha256']!=sha256(STUDY)):raise ValueError('Registered original task-supervision arm required')
    require_host_memory(8*2**30,maximum_workers=1);require_storage(work,growth_bytes=int(.5*2**30))
    physical,uuid=gpu_settings();teacher=load_teacher(checkpoint,training=True);torch.manual_seed(0)
    module=TaskUtilityFusion(**arm['module_kwargs']).cuda().float()
    if (sum(p.numel() for p in module.parameters())!=553705 or
            state_sha256(module.state_dict())!=study['architecture']['initial_state_sha256']):raise ValueError('Common seeded auxiliary initialization changed')
    reader,reference=make_image_reader(work,plan,protocol,arm['Train_delivery']['lengths'],arm['cue'])
    reader.max_bytes=2*2**30
    if reader.cache or reader.bytes:raise RuntimeError('Set the2GiB encoded RAM limit before the first read')
    provider=OrderedTrainPixels(plan,protocol,arm['Train_delivery']['lengths'],calibration,reader,arm['training_input_hw'],'cuda')
    metadata=dict(family='task_utility',fit_id=args.fit_id,variant=arm['variant'],cue=arm['cue'],mode='quality',removal=None,
        task_supervision=arm['task_supervision'],seed=0,fit_config_sha256=sha256(path),study_sha256=sha256(STUDY),
        registry_sha256=sha256(base/REGISTRY),computation_key=computation_key(arm),source_hashes=registration['source_hashes'],
        source_RGB_sha256=study['source_RGB_sha256'],protocol_sha256=study['protocol_sha256'],calibration_sha256=study['calibration_sha256'],
        Train_GT_sha256=provider.gt_hashes,initial_native_sha256=state_sha256(module.state_dict()),input_content_reference=reference,
        physical_gpu=physical,GPU_uuid=uuid,original_RGB_frozen=True,Dev_fit_rows=0,Test_fit_rows=0)
    proof_path=base/'engineering'/args.fit_id/'proof.json';started=time.monotonic()
    if args.mode=='probe':
        if proof_path.exists():raise ValueError('Preserve prior actual engineering proof')
        result=canary(module,teacher,provider,arm,inference_frames=12,progress=lambda r:print(json.dumps(r),flush=True));registered(work)
        dump(proof_path,dict(status='COMPLETE_TASK_UTILITY_GPU_TRAIN_B1_ENGINEERING',metadata=metadata,record=result,
            image_reader_statistics=reader.statistics(),seconds=time.monotonic()-started,GPU_work=True,
            peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
            Dev_rows=0,Test_rows=0,goal_met=False,supported_innovations=0))
    else:
        if args.through_epoch not in study['budget_and_Dev']['stages']:raise ValueError('Registered common Train stage required')
        for fit in study['normal_fits']:
            proof=read(base/'engineering'/fit['id']/'proof.json')
            if (proof['status']!='COMPLETE_TASK_UTILITY_GPU_TRAIN_B1_ENGINEERING' or not proof['GPU_work'] or
                    proof['metadata']['registry_sha256']!=metadata['registry_sha256'] or
                    proof['metadata']['fit_config_sha256']!=sha256(arm_path(base,fit['id']))):
                raise ValueError('All twelve actual GPU proofs required before fitting')
        authorization=read(base/'stage_authorized.json')
        if (authorization['study_sha256']!=sha256(STUDY) or authorization['through_epoch']!=args.through_epoch or
                args.fit_id not in authorization['fit_ids'] or authorization['arm_hashes'][args.fit_id]!=sha256(path)):
            raise ValueError('Complete Dev-based common-stage authorization required')
        optimizer=torch.optim.AdamW(module.parameters(),lr=arm['fit']['learning_rate'],weight_decay=arm['fit']['weight_decay'])
        run_training_to_epoch(module,teacher.backbone,teacher.head,optimizer,provider,arm,metadata,path,
            base/'fits'/args.fit_id,args.through_epoch,work)
        saved=torch.load(base/'fits'/args.fit_id/f'epoch_{args.through_epoch:02d}.pt',map_location='cpu',weights_only=False)
        validate_history(saved,args.through_epoch,arm);registered(work)


if __name__=='__main__':main()
