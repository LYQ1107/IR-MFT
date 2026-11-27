#!/usr/bin/env python3
"""Two-card fixed-Train DINO descriptor pilot; no detector/tracker changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.run_mft_birth_only import registered as parent_registered,locations as parent_locations,utc
from tools.native_pyramid_experiment import read,frozen_json,gpu_settings
from tools.native_pyramid_training_loop import guard_storage
from tools.mft_candidate_risk_data import load_arrays
from tools.mft_frozen_tokens import encode,load_model
from tools.probe_mft_native_memory_priority import tensor_digest
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_candidate_train_bank import atomic_json
from tools.task_utility_queue import Queue

SPEC=ROOT/'configs/perceptual_frozen_tokens_20261127.json'
OWN=['tools/mft_frozen_tokens.py','tools/run_mft_frozen_tokens.py','tests/test_mft_frozen_tokens.py',str(SPEC.relative_to(ROOT))]


def locations():
    p,protocol,w,b,t=parent_locations();return p,protocol,w,w/'frozen_tokens_20261127',t


def vendor_manifest(source):
    source=Path(source);spec=read(SPEC)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip();assert head==spec['source_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source).strip()
    names=subprocess.check_output(['git','ls-files','-z'],cwd=source).decode().split('\0')
    files={n:sha256(source/n)for n in names if n}
    aggregate=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(commit=head,tracked_files=files,aggregate_sha256=aggregate)


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    parent=parent_registered();spec=read(SPEC);plan,protocol,work,base,train=locations()
    proof=read(base/'CPU_regressions.json');assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==3
    sources=dict(parent['source_hashes'])
    for n in OWN:sources[n]=sha256(ROOT/n)
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h
    checkpoint=Path(spec['existing_checkpoint']);assert checkpoint.stat().st_size==spec['checkpoint_bytes']and sha256(checkpoint)==spec['checkpoint_sha256']
    vendor=vendor_manifest(spec['source_directory']);qualified=read(work/'proposal_quality_20261127/fit_registered.json')['qualified_Train']
    prior_path=work/'descriptor_location_probe_20261127/registered.json';prior=read(prior_path)
    selected=prior['selected_frame_ids'];counts={}
    for sequence,q in qualified.items():
        a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256'])
        counts[sequence]=sum(int(a['offsets'][f]-a['offsets'][f-1])for f in selected[sequence])
        assert len(selected[sequence])==64 and q['bank_sha256']==prior['qualified_Train'][sequence]['bank_sha256']
    assert len(counts)==6 and sum(counts.values())==6752
    payload=sum(counts.values())*6*768*2;assert payload<spec['private_descriptor_payload_cap_bytes']
    frozen_json(base/'registered.json',dict(status='REGISTERED_ACTUAL_CURRENT_PROPOSAL_DINO_PILOT_BEFORE_GPU',UTC=utc(),
        source_hashes=sources,source_vendor=vendor,spec_sha256=sha256(SPEC),CPU_regressions_sha256=sha256(base/'CPU_regressions.json'),qualified_Train=qualified,selected_counts=counts,selected_frame_ids=selected,
        prior_selection_path=str(prior_path),prior_selection_sha256=sha256(prior_path),
        physical_GPUs=['4','5'],Train_frames=384,Train_proposals=6752,selected_frames_per_video=64,private_descriptor_payload_bytes=payload,
        source='https://github.com/facebookresearch/dinov2',checkpoint_sha256=spec['checkpoint_sha256'],
        no_weight_download_or_copy=True,NN_training=False,labels_used_in_feature_producer=False,Dev_or_Test_rows=0,
        goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='REGISTERED_ACTUAL_CURRENT_PROPOSAL_DINO_PILOT_BEFORE_GPU',sources=len(sources),private_bytes=payload)),flush=True)


def registered():
    parent_registered();plan,protocol,work,base,train=locations();r=read(base/'registered.json');spec=read(SPEC)
    assert sha256(SPEC)==r['spec_sha256']
    assert sha256(base/'CPU_regressions.json')==r['CPU_regressions_sha256']
    assert sha256(r['prior_selection_path'])==r['prior_selection_sha256']
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    assert sha256(spec['existing_checkpoint'])==r['checkpoint_sha256']
    assert vendor_manifest(spec['source_directory'])==r['source_vendor']
    return r


def gpu():
    os.environ['XFORMERS_DISABLED']='1'
    import torch
    import cv2
    reg=registered();spec=read(SPEC);plan,protocol,work,base,train=locations();physical,uuid=gpu_settings()
    model=load_model(spec['source_directory'],spec['existing_checkpoint']);before=tensor_digest(model)
    own=list(reg['qualified_Train'])[0::2]if physical=='4'else list(reg['qualified_Train'])[1::2]
    manifests={}
    for sequence in own:
        q=reg['qualified_Train'][sequence];a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);n=reg['selected_counts'][sequence]
        selected=reg['selected_frame_ids'][sequence];offsets=[0];detections=[]
        values={view+'_'+kind:np.empty((n,768),np.float16)for view in ['RGB','IR','gray']for kind in ['cls','mean']}
        packets=hashlib.sha256();images=hashlib.sha256();quantization={k:0. for k in values};repeat_checked=False
        for frame in selected:
            lo,hi=map(int,a['offsets'][frame-1:frame+1]);d=a['detections'][lo:hi];original=d.copy()
            rp=Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'
            ip=Path(plan['generated_ir_root'])/'train'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
            rb,ib=rp.read_bytes(),ip.read_bytes();rgb=cv2.imdecode(np.frombuffer(rb,np.uint8),cv2.IMREAD_COLOR);ir=cv2.imdecode(np.frombuffer(ib,np.uint8),cv2.IMREAD_UNCHANGED)
            assert rgb is not None and ir is not None and ir.ndim==2 and rgb.shape[:2]==ir.shape
            image=dict(frame=frame,RGB_path=str(rp),IR_path=str(ip),RGB_sha256=hashlib.sha256(rb).hexdigest(),IR_sha256=hashlib.sha256(ib).hexdigest())
            outlo,outhi=offsets[-1],offsets[-1]+len(d)
            f=encode(model,rgb,ir,d[:,:4],spec['batch_size'])
            if not repeat_checked:
                second=encode(model,rgb,ir,d[:,:4],spec['batch_size'])
                assert all(np.array_equal(f[k],second[k])for k in f);repeat_checked=True
            np.testing.assert_array_equal(d,original)
            for key,x in f.items():
                restored=x.astype(np.float16).astype(np.float32)
                error=float(np.max(abs(x-restored)))if x.size else 0.;assert error<=1e-3
                assert np.isfinite(x).all()and(not len(x)or np.max(abs(np.linalg.norm(x,axis=1)-1))<1e-4)
                values[key][outlo:outhi]=x.astype(np.float16);quantization[key]=max(quantization[key],error)
            packets.update(np.asarray([frame,len(d)],dtype='<i8').tobytes()+np.asarray(d,dtype='<f4').tobytes())
            images.update(json.dumps(image,separators=(',',':')).encode()+b'\n')
            offsets.append(outhi);detections.append(d.copy())
            if len(detections)%16==0:
                guard_storage(work);atomic_json(base/'progress'/('GPU_'+physical+'.json'),dict(sequence=sequence,actual_frame=frame,selected_frames_complete=len(detections),selected_frames=64))
        assert offsets[-1]==n
        values.update(detections=np.concatenate(detections),offsets=np.asarray(offsets,np.int64),frame_ids=np.asarray(selected,np.int64))
        out=base/'private_bank'/(sequence+'.npz');out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
        temporary=out.with_suffix('.tmp.npz');np.savez_compressed(temporary,**values);os.replace(temporary,out)
        manifests[sequence]=dict(frames=64,selected_frame_ids=selected,proposals=n,path=str(out),sha256=sha256(out),bytes=out.stat().st_size,
            original_bank_sha256=q['bank_sha256'],original_selected_detector_packet_sha256=packets.hexdigest(),
            delivered_image_packet_sha256=images.hexdigest(),normalized_FP32_descriptors_stored_FP16=True,
            roundtrip_max_error=quantization,first_frame_repeat_FP32_exact=True,labels_or_GT_crops_used=False,
            no_future_frame_input=True,no_patch_tokens_crops_or_images_stored=True)
        frozen_json(base/'private_bank'/(sequence+'.json'),manifests[sequence]);del a,values
        print(json.dumps(dict(sequence=sequence,proposals=n,status='ACTUAL_DINO_SPARSE_FRAMES_COMPLETE')),flush=True)
    assert tensor_digest(model)==before
    registered();guard_storage(work)
    frozen_json(base/('GPU_'+physical+'.json'),dict(status='COMPLETE_ACTUAL_FROZEN_DINO_TRAIN_SPARSE_FRAMESES',UTC=utc(),
        physical_gpu=physical,GPU_uuid=str(uuid),actual_identity=identity(os.getpid()),sequences=manifests,
        model_all_weights_and_buffers_unchanged=True,model_tensor_digest=before,
        source_hashes=reg['source_hashes'],vendor_source_sha256=reg['source_vendor']['aggregate_sha256'],
        checkpoint_sha256=reg['checkpoint_sha256'],model_parameters=sum(p.numel()for p in model.parameters()),
        peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),GT_labels_read=False,Dev_or_Test_rows=0,
        new_NN_training=False,new_training_seeds=0,tracking_accuracy=False,goal_met=False,supported_innovations=0))


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,train=locations();assert not(base/'state.json').exists()
    state=dict(status='STARTING_TWO_CARD_DINO_TRAIN_SPARSE_FRAMES_EXTRACTION',controller_pid=os.getpid(),phases={},seed=0,goal_met=False)
    queue=Queue(work,base,state,plan['data_root'],registered);queue.save();cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
    try:
        queue.jobs('DINO_FROZEN_TRAIN_SPARSE_FRAMESES',[dict(id=g,physical_gpu=g,command=cmd+['gpu'])for g in ['4','5']],True,
            lambda j:read(base/('GPU_'+j['id']+'.json'))['model_all_weights_and_buffers_unchanged'])
        reports=[read(base/('GPU_'+g+'.json'))for g in ['4','5']]
        assert sum(v['proposals']for r in reports for v in r['sequences'].values())==6752
        state.update(status='COMPLETE_DINO_384_TRAIN_FRAMES_REQUIRES_CPU_FEATURE_ANALYSIS');queue.save()
    except Exception as error:
        state.update(status='FAILED_PRESERVE_DINO_EXTRACTION',error=repr(error));queue.save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run','gpu']);a=p.parse_args()
    {'register':register,'run':run,'gpu':gpu}[a.mode]()
