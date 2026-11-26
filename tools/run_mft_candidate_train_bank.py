#!/usr/bin/env python3
"""Bounded all-Train real proposal/cue extraction on physical GPUs4/5."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.fusion_memory_observation import FixedSequenceImages,LENGTHS
from tools.mft_candidate_bank_storage import PayloadBudget,payload_bytes,quantize_frame
from tools.native_pyramid_experiment import context,frozen_json,read
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.probe_mft_native_memory_priority import tensor_digest
from tools.run_mft_pose_continuity import registered as parent_registered
from tools.run_mft_tracklet_permanence_r2 import gpu_models,encode,replay_arrays
from tools.mft_tracklet_permanence_r2 import cases as reference_cases
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_queue import Queue

SPEC=ROOT/'configs/perceptual_candidate_train_bank_20261127.json'
PREFIX='CANDIDATE_TRAIN_BANK_20261127'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    values=context()
    return(*values,values[4]/'candidate_train_bank_20261127')


def ready(work):
    pose=work/'pose_continuity_20261127'
    execution=read(pose/'controller.execution.json')
    assert execution['actual_child_exit_code']==0 and execution['source_unchanged']
    assert identity(execution['identity']['PID'])is None
    state=read(pose/'state.json');jobs=[j for rows in state['phases'].values()for j in rows.values()]
    assert len(jobs)==20 and all(j['status']=='COMPLETE'and type(j.get('exit_code'))is int
        and j['exit_code']==0 and j['argv_identity_observed']and identity(j['pid'])is None for j in jobs)
    for stem in ['independent_Dev_audit_r1','independent_Test_audit_r1']:
        actual=read(pose/(stem+'.actual_completion.json'))
        assert actual['actual_tool_exit_code']==0 and actual['source_unchanged']
    audit=read(pose/'independent_Test_audit_complete.json')
    assert audit['official_numeric_fields_checked']==2256
    report=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261127_POSE_CONTINUITY_FULL_TEST.json'
    assert sha256(report)==audit['report_sha256']
    previous=work/'tracklet_permanence_20261127_r2'
    for name in ['controller.execution.json','independent_test_audit_r1.execution.json']:
        value=read(previous/name)
        assert value['actual_child_exit_code']==0 and value['source_unchanged']and identity(value['identity']['PID'])is None
    return audit


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,study,protocol,calibration,work,checkpoint,base=locations()
    prior=ready(work);parent=parent_registered();spec=read(SPEC)
    proof=read(base/'CPU_storage_regressions_r1.json')
    assert proof['tests_passed']==4 and proof['actual_tool_exit_code']==0
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n
    sources=dict(parent['source_hashes'])
    for n in ['tools/run_mft_candidate_train_bank.py','tools/mft_candidate_bank_storage.py',
              'tests/test_mft_candidate_bank_storage.py',str(SPEC.relative_to(ROOT))]:sources[n]=sha256(ROOT/n)
    maximum=int(spec['maximum_uncompressed_payload_GiB']*2**30)
    used=sum(p.stat().st_size for p in work.rglob('*')if p.is_file())
    guard_storage(work)
    assert used+int(1.1*maximum)+int(.1*2**30)<64*2**30,'New bank plus explicit overhead must fit64GiB'
    pose=work/'pose_continuity_20261127'
    paths=[base/'CPU_storage_regressions_r1.json',base/'CPU_storage_regressions_r1.stdout.txt',
        pose/'controller.execution.json',pose/'complete_Dev.json',pose/'full_Test_results.json',
        pose/'independent_Dev_audit_r1.actual_completion.json',pose/'independent_Test_audit_r1.actual_completion.json',
        pose/'independent_Test_audit_complete.json']
    value=dict(status='REGISTERED_ALL_TRAIN_COMPACT_CAUSAL_CANDIDATE_BANK',UTC=utc(),
        source_hashes=sources,artifact_hashes={str(p):sha256(p)for p in paths},specification_sha256=sha256(SPEC),
        maximum_uncompressed_payload_bytes=maximum,initial_work_bytes=used,full_Train_frames=27251,
        prior_pose_Test_HOTA=prior['HOTA'],prior_pose_Test_gain_pp=prior['gain_over_original_RGB_pp'],
        skip_if_existing_plus5_met=prior['target_plus5_met'],physical_gpus=['4','5'],maximum_cards=2,
        fixed_seed=0,new_NN_fits=0,new_accuracy=False,historical_Test_previously_exposed=True,
        this_stage_designed_after_prior_Test_review=True,goal_met=False,supported_innovations=0)
    assert not(base/'pipeline_registered.json').exists();frozen_json(base/'pipeline_registered.json',value)
    print(json.dumps(dict(status=value['status'],sources=len(sources),initial_work_GiB=used/2**30,
        maximum_extra_payload_GiB=maximum/2**30,skip_existing_target_met=value['skip_if_existing_plus5_met'])),flush=True)


def registered():
    plan,study,protocol,calibration,work,checkpoint,base=locations();parent_registered()
    value=read(base/'pipeline_registered.json');assert value['specification_sha256']==sha256(SPEC)
    for n,h in value['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in value['artifact_hashes'].items():assert sha256(n)==h,n
    return value


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def load_gpu():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')in['4','5']
    free,total=torch.cuda.mem_get_info()
    assert free>=16*2**30,'Preserve worker: less than16GiB free on its allowed card'
    return gpu_models()


def collect(sequence,models,frames=None,reserve=False):
    teacher,module,visual,encoders,physical,uuid=models
    plan,study,protocol,calibration,work,checkpoint,base=locations()
    images=FixedSequenceImages(plan,'train',sequence,frames=frames)
    detector=OnlineNativeDetector(teacher,[visual['model']],{visual['id']:module},(800,1440),calibration)
    maximum=read(base/'pipeline_registered.json')['maximum_uncompressed_payload_bytes']
    budget=PayloadBudget(base,maximum)
    parts={k:[]for k in ['detections','rgb','ir']};offsets=[0]
    image_hash,packet_hash=hashlib.sha256(),hashlib.sha256()
    source_features={k:hashlib.sha256()for k in ['rgb','ir']}
    errors={k:dict(maximum_absolute_error=0.,maximum_unit_norm_error=0.)for k in ['rgb','ir']}
    started=time.monotonic()
    for frame in range(1,images.frames+1):
        rgb,ir,row=images.read_frame(frame,True)
        image_hash.update(json.dumps(row,separators=(',',':')).encode()+b'\n')
        result,_=detector.step(rgb,ir,sequence,frame);d=result[visual['id']]
        encoded=encode(encoders,rgb,ir,d[:,:4]);packed,loss=quantize_frame(d,encoded)
        header=np.asarray([frame,len(d)],dtype='<i8').tobytes()
        packet_hash.update(header);packet_hash.update(np.asarray(d,dtype='<f4').tobytes())
        for cue in ['rgb','ir']:
            source_features[cue].update(header);source_features[cue].update(np.asarray(encoded[cue],dtype='<f4').tobytes())
            for key,value in loss[cue].items():errors[cue][key]=max(errors[cue][key],value)
        for key in parts:parts[key].append(packed[key])
        offsets.append(offsets[-1]+len(d))
        if frame%100==0 or frame==images.frames:
            amount=payload_bytes(offsets[-1],frame)
            if reserve:budget.reserve(sequence,amount)
            guard_storage(work)
            atomic_json(base/'progress'/((sequence if frames is None else sequence+'_engineering_'+physical)+'.json'),
                dict(frame=frame,frames=images.frames,proposals=offsets[-1],uncompressed_payload_bytes=amount,
                    physical_gpu=physical,seconds=time.monotonic()-started,GT_labels_parsed=False))
    arrays={k:np.concatenate(v,axis=0)for k,v in parts.items()}
    arrays.update(offsets=np.asarray(offsets,np.int64),frame_ids=np.arange(1,images.frames+1,dtype=np.int64))
    actual=sum(v.nbytes for v in arrays.values());assert actual==payload_bytes(offsets[-1],images.frames)
    info=dict(sequence=sequence,split='train',frames=images.frames,proposals=offsets[-1],uncompressed_payload_bytes=actual,
        image_delivery_sha256=image_hash.hexdigest(),detection_packet_sha256=packet_hash.hexdigest(),
        FP32_source_descriptor_packet_sha256={k:h.hexdigest()for k,h in source_features.items()},
        descriptor_FP16_roundtrip_errors=errors,seconds=time.monotonic()-started,
        GT_labels_parsed=False,oracle_GT_crops_used=False,Dev_or_Test_images_read=0,
        future_frame_input=False,full_frame_feature_copy=False,image_copy=False,
        inference_only=True,new_NN_fits=0,physical_gpu=physical,GPU_uuid=uuid)
    return arrays,info


def frozen_models_digest(models):
    teacher,module,visual,encoders,physical,uuid=models
    return {k:tensor_digest(v)for k,v in dict(teacher=teacher,module=module,**encoders).items()}


def engineering():
    reg=registered();plan,study,protocol,calibration,work,checkpoint,base=locations()
    models=load_gpu();physical,uuid=models[-2:];before=frozen_models_digest(models)
    full,source=collect('BT-003',models,frames=12)
    prefix,prefix_source=collect('BT-003',models,frames=6)
    cut=int(full['offsets'][6]);assert cut==len(prefix['detections'])
    np.testing.assert_array_equal(full['offsets'][:7],prefix['offsets'])
    maxima={}
    for key in ['detections','rgb','ir']:
        a=full[key][:cut].astype(np.float32);b=prefix[key].astype(np.float32)
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-4 if key=='detections'else 1e-3)
        maxima[key]=float(np.max(np.abs(a-b)))if a.size else 0.
    thresholds=read(work/'tracklet_permanence_20261127_r2/Train_thresholds.json')['thresholds']
    rows,_=replay_arrays(full,[reference_cases()[0]],thresholds,'BT-003')
    original=IndependentByteTrack();expected=[]
    for i,frame in enumerate(full['frame_ids']):
        lo,hi=map(int,full['offsets'][i:i+2]);expected.extend(original.update(full['detections'][lo:hi],int(frame)))
    assert ''.join(map(mot_line,rows['TP_native_zero']))==''.join(map(mot_line,expected))
    assert frozen_models_digest(models)==before
    registered();frozen_json(base/('gpu_engineering_'+physical+'.json'),dict(
        status='COMPLETE_REAL_GPU_TRAIN_COMPACT_BANK_ENGINEERING',UTC=utc(),physical_gpu=physical,GPU_uuid=uuid,
        actual_identity=identity(os.getpid()),Train_frames=12,reset_neural_prefix_frames=6,
        reset_neural_prefix_max_absolute_error=maxima,neural_prefix_tolerance=dict(detections=1e-4,FP16_descriptors=1e-3),
        original_zero_tracker_MOT_exact=True,all_model_weights_and_BN_unchanged=True,
        descriptor_quantization_verified=True,source=source,prefix_source=prefix_source,
        FP32_detections_FP16_descriptors=True,new_accuracy=False,new_NN_fits=0,
        pipeline_registration_sha256=sha256(base/'pipeline_registered.json')))
    print(json.dumps(dict(status='COMPLETE_REAL_GPU_TRAIN_COMPACT_BANK_ENGINEERING',physical_gpu=physical)),flush=True)


def require_engineering(base):
    jobs=read(base/'state.json')['phases'][PREFIX+'_GPU_ENGINEERING']
    assert set(jobs)=={'4','5'}
    for physical,job in jobs.items():
        assert job['status']=='COMPLETE'and job['exit_code']==0 and identity(job['pid'])is None
        value=read(base/('gpu_engineering_'+physical+'.json'))
        assert value['physical_gpu']==physical and value['Train_frames']==12
        assert value['all_model_weights_and_BN_unchanged']and value['descriptor_quantization_verified']
        assert value['original_zero_tracker_MOT_exact']


def produce(sequence):
    assert sequence in LENGTHS['train']
    reg=registered();plan,study,protocol,calibration,work,checkpoint,base=locations();require_engineering(base)
    folder=base/'cache';folder.mkdir(exist_ok=True)
    path=folder/(sequence+'.npz');meta=folder/(sequence+'.json')
    assert not path.exists()and not meta.exists(),'Preserve prior completed or partial work'
    models=load_gpu();before=frozen_models_digest(models)
    arrays,info=collect(sequence,models,reserve=True)
    assert info['frames']==LENGTHS['train'][sequence]
    assert arrays['detections'].dtype==np.float32 and arrays['rgb'].dtype==arrays['ir'].dtype==np.float16
    assert frozen_models_digest(models)==before
    PayloadBudget(base,reg['maximum_uncompressed_payload_bytes']).reserve(sequence,info['uncompressed_payload_bytes'])
    guard_storage(work)
    temporary=folder/(sequence+'.tmp.npz');assert not temporary.exists()
    np.savez_compressed(temporary,**arrays);os.replace(temporary,path)
    registered();guard_storage(work)
    frozen_json(meta,dict(status='COMPLETE_ALL_FRAMES_TRAIN_CAUSAL_COMPACT_CANDIDATE_BANK',UTC=utc(),**info,
        path=str(path),sha256=sha256(path),bytes=path.stat().st_size,
        source_versions=reg['source_hashes'],pipeline_registration_sha256=sha256(base/'pipeline_registered.json'),
        actual_identity=identity(os.getpid()),all_model_weights_and_BN_unchanged=True,
        dtype='FP32 detections, FP16 stored descriptors encoded in FP32, INT64 frame/offset indices',
        downstream_read_policy='Restore descriptors to FP32 and normalize; quantization is not claimed exact to FP32',
        checksum_qualification_can_read_prior_training_metadata=True,historical_full_Train_packet_anchor_available=False,
        no_image_crop_or_FPN_copy=True,new_accuracy=False,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_ALL_FRAMES_TRAIN_CAUSAL_COMPACT_CANDIDATE_BANK',sequence=sequence,
        frames=info['frames'],proposals=info['proposals'],bytes=path.stat().st_size)),flush=True)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,study,protocol,calibration,work,checkpoint,base=locations();ready(work)
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_ALL_TRAIN_CANDIDATE_BANK',controller_pid=os.getpid(),phases={},data_root=plan['data_root'],
            physical_gpus=['4','5'],maximum_cards=2,fixed_seed=0,new_NN_fits=0,new_accuracy=False,goal_met=False,supported_innovations=0)
        queue=Queue(work,base,state,plan['data_root'],registered);queue.save()
        if reg['skip_if_existing_plus5_met']:
            state.update(status='COMPLETE_SKIP_TRAIN_EXTRACTION_EXISTING_POSE_PLUS5_MET',GPU_work=False);queue.save();return
        try:
            command=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
            queue.jobs(PREFIX+'_GPU_ENGINEERING',[dict(id=g,physical_gpu=g,command=command+['engineering'])for g in ['4','5']],
                True,lambda j:read(base/('gpu_engineering_'+j['id']+'.json'))['descriptor_quantization_verified'])
            require_engineering(base)
            queue.jobs(PREFIX+'_TRAIN_NN',[dict(id=s,command=command+['produce','--sequence',s])
                for s in sorted(LENGTHS['train'],key=lambda s:-LENGTHS['train'][s])],True,
                lambda j:read(base/'cache'/(j['id']+'.json'))['frames']==LENGTHS['train'][j['id']])
            metadata=[read(base/'cache'/(s+'.json'))for s in LENGTHS['train']]
            assert sum(m['frames']for m in metadata)==27251 and len(metadata)==6
            ledger=read(base/'payload_reservations.json')
            assert set(ledger['sequences'])==set(LENGTHS['train'])
            assert ledger['total_reserved_bytes']==sum(m['uncompressed_payload_bytes']for m in metadata)<=reg['maximum_uncompressed_payload_bytes']
            for m in metadata:assert sha256(m['path'])==m['sha256']and m['all_model_weights_and_BN_unchanged']
            frozen_json(base/'complete_Train.json',dict(status='COMPLETE_ALL6_REAL_TRAIN_CANDIDATE_BANKS_READY_FOR_SUPERVISED_RISK_FIT',
                UTC=utc(),frames=27251,sequences=6,rows=metadata,total_proposals=sum(m['proposals']for m in metadata),
                total_private_compressed_bytes=sum(m['bytes']for m in metadata),total_uncompressed_payload_bytes=ledger['total_reserved_bytes'],
                all8_related_GPU_jobs_actual_exit_zero=True,new_NN_fits=0,new_tracking_accuracy=False,
                GT_labels_parsed_by_producers=False,Dev_or_Test_images_read=0,goal_met=False,supported_innovations=0))
            state.update(status='COMPLETE_TRAIN_BANK_READY_NO_NEW_MODEL_OR_ACCURACY');queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_TRAIN_BANK_ATTEMPT',error=repr(error));queue.save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['register','run','engineering','produce'])
    p.add_argument('--sequence',choices=list(LENGTHS['train']))
    args=p.parse_args()
    if args.mode=='register':register()
    elif args.mode=='run':run()
    elif args.mode=='engineering':engineering()
    else:produce(args.sequence)


if __name__=='__main__':main()
