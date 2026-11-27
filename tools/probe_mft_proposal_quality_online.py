#!/usr/bin/env python3
"""Two-card actual-Train-image quality bridge and whole-IR fallback engineering."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.run_mft_proposal_quality import registered as evaluation_registered,fitted,locations as parent_locations,utc
from tools.mft_proposal_quality_online import ProposalQualityOnline
from tools.mft_proposal_quality_evaluation import policies,SharedConfirmation,make_tracker
from tools.run_mft_archive_reasoning import fitted as archive_fitted
from tools.fit_mft_candidate_risk import threshold_values
from tools.run_mft_tracklet_permanence_r2 import gpu_models
from tools.fusion_memory_observation import FixedSequenceImages
from tools.mft_candidate_risk_data import load_arrays
from tools.native_pyramid_experiment import context,read,frozen_json
from tools.native_pyramid_stream_tracks import mot_line
from tools.probe_mft_native_memory_priority import tensor_digest
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_queue import Queue
from tools.run_mft_native_memory_priority_v3 import identity

SPEC=ROOT/'configs/perceptual_proposal_quality_online_probe_20261127.json'


def locations():
    plan,protocol,work,parent,train=parent_locations();return plan,protocol,work,work/'proposal_quality_online_probe_20261127',train


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    main=evaluation_registered();_,fit,views,heads,thresholds=fitted();plan,protocol,work,base,train=locations()
    sources={**main['source_hashes']}
    for n in ['tools/mft_proposal_quality_online.py','tools/probe_mft_proposal_quality_online.py',str(SPEC.relative_to(ROOT))]:sources[n]=sha256(ROOT/n)
    q=read(train/'fit_registered.json')['qualified_Train']['BT-003']
    v=dict(status='REGISTERED_ACTUAL_TRAIN_QUALITY_BRIDGE_AND_WHOLE_IR_ENGINEERING',UTC=utc(),source_hashes=sources,
        specification_sha256=sha256(SPEC),model_sha256=fit['model_sha256'],qualified_Train_BT003=q,
        cases=policies(),physical_GPUs=['4','5'],maximum_cards=2,seed=0,
        neural_modes=['normal','unavailable','IR_image_spatial_roll'],Train_frames_per_mode=12,normal_reset_prefix=6,
        labels_read=False,Dev_or_Test_frames=0,new_accuracy=False,new_training_seeds=0,new_NN_fits=0,
        whole_IR_controls_affect_both_visual_detector_and_quality=True,primary_baseline_unchanged=True,
        learned_parameters_not_for_Git=True,goal_met=False,supported_innovations=0)
    assert not(base/'registered.json').exists();frozen_json(base/'registered.json',v)
    print(json.dumps(dict(status=v['status'],sources=len(sources))),flush=True)


def registered():
    evaluation_registered();plan,protocol,work,base,train=locations();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    assert sha256(SPEC)==r['specification_sha256']
    return r


def gpu():
    import torch
    reg=registered();_,fit,views,heads,thresholds=fitted();plan,protocol,work,base,train=locations()
    _,_,_,calibration,_,_=context();_,_,archive_models,archive_thresholds=archive_fitted()
    models=gpu_models();physical,uuid=models[-2:];assert physical in ['4','5']
    teacher,module,visual,encoders,_,_=models
    def digests():return {k:tensor_digest(v)for k,v in dict(teacher=teacher,module=module,**encoders).items()}
    before=digests();q=reg['qualified_Train_BT003'];bank=load_arrays(q['bank_path'],q['frames'],q['bank_sha256'])
    speed=threshold_values(work)['Train_unknown_speed_upper'];results={};normal_prefix=None
    def execute(mode,count):
        engine=ProposalQualityOnline(models,calibration,views,heads,thresholds,speed,archive_models,archive_thresholds,policies(),mode)
        images=FixedSequenceImages(plan,'train','BT-003',frames=count);shared=SharedConfirmation()
        controls={c['id']:make_tracker(c,views,heads,thresholds,speed,archive_models,archive_thresholds,shared)for c in policies()}
        rows={c['id']:[]for c in policies()};sources=hashlib.sha256();parity=dict(detections=0.,RGB_postBN_FP16=0.,IR_postBN_FP16=0.)
        for frame in range(1,count+1):
            rgb,ir,image=images.read_frame(frame,engine.needs_actual_IR_image);sources.update(json.dumps(image,separators=(',',':')).encode()+b'\n')
            out,packets,info=engine.step(rgb,ir,'BT-003',frame);e=engine.last_evidence
            d,r,i=e['detections'],e['rgb'],e['ir']
            for case in policies():
                name=case['id'];actual=''.join(map(mot_line,out[name]));expected=''.join(map(mot_line,controls[name].update('BT-003',frame,d,r,i)))
                assert actual==expected;rows[name].append(actual)
            if mode=='normal':
                lo,hi=map(int,bank['offsets'][frame-1:frame+1]);expected=bank['detections'][lo:hi]
                assert d.shape==expected.shape
                for key,a,b,tolerance in [('detections',d,expected,1e-4),('RGB_postBN_FP16',r,bank['rgb'][lo:hi].astype(np.float32),1e-3),
                    ('IR_postBN_FP16',i,bank['ir'][lo:hi].astype(np.float32),1e-3)]:
                    error=float(np.max(abs(a-b)))if a.size else 0.;assert error<=tolerance,(key,frame,error);parity[key]=max(parity[key],error)
            if mode=='unavailable':
                assert i is None and all(info['visual']['full_head_exact_to_fresh_RGB'].values())
        engine.close();assert digests()==before
        return rows,dict(frames=count,image_delivery_sha256=sources.hexdigest(),online_vs_fresh_FP32_replay_all16_MOT_exact=True,
            existing_Train_bank_parity_max_abs_error=parity if mode=='normal'else None,
            unavailable_visual_head_exact_to_same_source_fresh_RGB=True if mode=='unavailable'else None,
            all_model_weights_and_BN_unchanged=True)
    for mode in reg['neural_modes']:
        rows,details=execute(mode,12);results[mode]=details
        if mode=='normal':normal_prefix={k:v[:6]for k,v in rows.items()}
        guard_storage(work)
    rows,details=execute('normal',6);assert normal_prefix==rows
    assert sha256(fit['model_file'])==fit['model_sha256'];registered()
    frozen_json(base/('GPU_'+physical+'.json'),dict(status='COMPLETE_REAL_GPU_QUALITY_IMAGE_BRIDGE_AND_WHOLE_IR_ENGINEERING',UTC=utc(),
        physical_gpu=physical,GPU_uuid=uuid,actual_identity=identity(os.getpid()),results=results,
        normal_reset_prefix6_all16_MOT_exact=True,model_sha256=fit['model_sha256'],source_versions=reg['source_hashes'],
        GT_labels_parsed_or_used=False,oracle_crops=False,Dev_or_Test_frames=0,new_accuracy=False,
        descriptor_image_or_FPN_disk_dump=False,new_NN_fits=0,new_training_seeds=0,
        whole_IR_mode_scope='unavailable/spatialrollaffectsvisualdetectorandquality;notnewbaselineorHOTAresult',goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_REAL_GPU_QUALITY_IMAGE_BRIDGE_AND_WHOLE_IR_ENGINEERING',physical_gpu=physical,results=results)),flush=True)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,train=locations();assert not(base/'state.json').exists()
    state=dict(status='STARTING_TWO_CARD_ACTUAL_IMAGE_QUALITY_ENGINEERING',controller_pid=os.getpid(),phases={},seed=0,goal_met=False)
    queue=Queue(work,base,state,plan['data_root'],registered);queue.save();cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
    try:
        queue.jobs('QUALITY_REAL_IMAGE_GPU_ENGINEERING',[dict(id=g,physical_gpu=g,command=cmd+['gpu'])for g in ['4','5']],True,
            lambda j:read(base/('GPU_'+j['id']+'.json'))['normal_reset_prefix6_all16_MOT_exact'])
        state.update(status='COMPLETE_TWO_CARD_ACTUAL_IMAGE_QUALITY_ENGINEERING');queue.save()
    except Exception as error:
        state.update(status='FAILED_PRESERVE_REAL_IMAGE_QUALITY_ENGINEERING',error=repr(error));queue.save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run','gpu']);a=p.parse_args()
    {'register':register,'run':run,'gpu':gpu}[a.mode]()
