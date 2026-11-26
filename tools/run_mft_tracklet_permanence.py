#!/usr/bin/env python3
"""Frozen native-IR proposals, compact dual-identity cache, causal tracklet study."""
import argparse
import copy
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
from tools.fusion_memory_evaluation import run_official_tracks,official_csv47,validate_mot
from tools.fusion_memory_observation import FixedSequenceImages,LENGTHS
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.mft_tracklet_permanence import TrackletPermanence,cases
from tools.native_pyramid_experiment import context,frozen_json,gpu_settings,read
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.probe_mft_native_memory_priority import tensor_digest
from tools.run_mft_fisher_memory import load_models as load_visual,registered as previous_registered
from tools.run_mft_native_memory_priority_v3 import PriorityQueue,identity
from tools.run_pseudo_ir_comparison import sha256

SPEC=ROOT/'configs/perceptual_tracklet_permanence_20261127.json'
PREFIX='TRACKLET_20261127_R1'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    values=context()
    return (*values,values[4]/'tracklet_permanence_20261127')


def ready(work):
    b=work/'fisher_memory_20261127';execution=read(b/'controller.execution.json');state=read(b/'state.json')
    assert execution['actual_child_exit_code']==0 and not identity(execution['identity']['PID'])
    assert state['status']=='COMPLETE_FISHER_NO_POSITIVE_DEV_GAIN_NO_NEW_TEST'
    assert all(j['status']=='COMPLETE' and type(j.get('exit_code'))is int and j['exit_code']==0 and not identity(j['pid'])for rows in state['phases'].values()for j in rows.values())
    for n in ['independent_Dev_audit_r1.execution.json','independent_Test_audit_r1.execution.json']:
        e=read(b/n);assert e['actual_child_exit_code']==0 and not identity(e['identity']['PID'])


def anchor(work,split,sequence):
    folder='dev_normal_06'if split=='dev'else'test_original_baseline_priority_20261126'
    path=work/'native_pyramid_streams'/folder/'sequence_manifests'/(sequence+'.json')
    value=read(path);assert value['frames']==LENGTHS[split][sequence] and value['GT_read']is False
    return path,value


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,study,protocol,calibration,work,checkpoint,base=locations();ready(work)
    fit=read(base/'Train_thresholds.json');execution=read(base/'Train_threshold_fit_r1.execution.json')
    assert execution['actual_child_exit_code']==0 and execution['source_unchanged'] and not identity(execution['identity']['PID'])
    assert fit['specification_sha256']==sha256(SPEC) and fit['Dev_rows']==fit['Test_rows']==0
    sources=dict(previous_registered()['source_hashes'])
    for n in ['tools/mft_tracklet_permanence.py','tools/fit_mft_tracklet_thresholds.py','tools/run_mft_tracklet_permanence.py','tests/test_tracklet_permanence.py',str(SPEC.relative_to(ROOT))]:sources[n]=sha256(ROOT/n)
    for name,digest in sources.items():assert sha256(ROOT/name)==digest
    artifacts={str(p):sha256(p)for p in [base/'Train_thresholds.json',base/'Train_threshold_fit_r1.execution.json',base/'CPU_tracklet_regression_r1.json',work/'fisher_memory_20261127/controller.execution.json',work/'fisher_memory_20261127/complete_Dev.json']}
    for cue in ['rgb','ir']:
        meta=fit['sources'][cue];assert sha256(meta['checkpoint'])==meta['checkpoint_sha256']
        artifacts[meta['checkpoint']]=meta['checkpoint_sha256']
        assert sha256(meta['path'])==meta['sha256']
        artifacts[meta['path']]=meta['sha256']
        feature_meta=Path(meta['path']).parent/'complete.json'
        assert sha256(feature_meta)==meta['metadata_sha256']
        artifacts[str(feature_meta)]=meta['metadata_sha256']
    expected=0;anchors={}
    for split in ['dev','test']:
        anchors[split]={}
        for seq in LENGTHS[split]:
            p,v=anchor(work,split,seq);anchors[split][seq]=dict(path=str(p),sha256=sha256(p))
            expected+=v['streams']['NP_temporal_ir_e06']['detections']*(5+1024)*4+(v['frames']*2+1)*8
    storage=guard_storage(work)
    used=sum(p.stat().st_size for p in work.rglob('*')if p.is_file())
    assert used+int(expected*1.2)<64*2**30,'Conservative compact Dev/Test bank must fit64GiB'
    value=dict(status='REGISTERED_NATIVE_TRACKLET_PERMANENCE_WITH_COMPACT_CAUSAL_CACHE',UTC=utc(),
        specification_sha256=sha256(SPEC),source_hashes=sources,artifact_hashes=artifacts,
        normal_cases=cases(),image_anchors=anchors,expected_Dev_and_Test_uncompressed_bytes=expected,
        original_negative_Fisher_Dev_preserved=True,physical_gpus=['4','5'],fixed_seed=0,
        supported_innovations=0,goal_met=False)
    assert not(base/'pipeline_registered.json').exists();frozen_json(base/'pipeline_registered.json',value)
    print(json.dumps(dict(status=value['status'],sources=len(sources),compact_payload_GiB=expected/2**30)),flush=True)


def registered():
    plan,study,protocol,calibration,work,checkpoint,base=locations();previous_registered()
    value=read(base/'pipeline_registered.json');assert value['specification_sha256']==sha256(SPEC)
    for n,h in value['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in value['artifact_hashes'].items():assert sha256(n)==h,n
    return value


def gpu_models():
    # Keep ownership explicit: all four frozen models share just this worker's card.
    import torch
    from tools.extract_perceptual_mft_features import AppearanceEncoder
    plan,study,protocol,calibration,work,checkpoint,base=locations();ready(work)
    teacher,module,visual,_=load_visual(study,checkpoint,work)
    encoders={};fit=read(base/'Train_thresholds.json')
    for cue in ['rgb','ir']:
        info=fit['sources'][cue];assert sha256(info['checkpoint'])==info['checkpoint_sha256']
        saved=torch.load(info['checkpoint'],map_location='cpu',weights_only=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0);model=AppearanceEncoder()
        assert set(saved['model'])==set(model.state_dict())
        for key,value in saved['model'].items():assert value.dtype==model.state_dict()[key].dtype
        model.load_state_dict(saved['model'],strict=True);encoders[cue]=model.eval().requires_grad_(False)
    physical,uuid=gpu_settings()
    teacher=teacher.cuda().half().eval().requires_grad_(False);module=module.cuda().float().eval().requires_grad_(False)
    encoders={k:v.cuda().float().eval().requires_grad_(False)for k,v in encoders.items()}
    return teacher,module,visual,encoders,physical,str(uuid)


def encode(encoders,rgb,ir,boxes):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from yolox.tracker.native_cue import native_crop,native_input
    result={}
    with torch.inference_mode():
        for cue,image in [('rgb',rgb),('ir',ir)]:
            values=[]
            for lo in range(0,len(boxes),32):
                pairs=[native_crop(image,box)for box in boxes[lo:lo+32]]
                pixels=torch.from_numpy(np.stack([p[0]for p in pairs]));rectangles=torch.from_numpy(np.stack([p[1]for p in pairs]))
                model=encoders[cue];x=native_input(pixels,rectangles,'cuda')
                raw=model.projection(model.backbone(x));values.append(F.normalize(model.neck(raw),dim=-1).cpu().numpy())
            result[cue]=np.concatenate(values).astype(np.float32)if values else np.empty((0,512),np.float32)
            assert np.isfinite(result[cue]).all() and np.allclose(np.linalg.norm(result[cue],axis=1),1,atol=1e-5)
    return result


def collect(split,sequence,teacher,module,visual,encoders,frames=None):
    import numpy as np
    plan,study,protocol,calibration,work,checkpoint,base=locations()
    images=FixedSequenceImages(plan,split,sequence,frames=frames)
    detector=OnlineNativeDetector(teacher,[visual['model']],{visual['id']:module},(800,1440),calibration)
    image_hash,packet_hash=hashlib.sha256(),hashlib.sha256();parts={k:[]for k in ['detections','rgb','ir']};offsets=[0]
    started=time.monotonic()
    for frame in range(1,images.frames+1):
        rgb,ir,row=images.read_frame(frame,True);image_hash.update(json.dumps(row,separators=(',',':')).encode()+b'\n')
        outputs,_=detector.step(rgb,ir,sequence,frame);d=outputs[visual['id']];embedding=encode(encoders,rgb,ir,d[:,:4])
        packet_hash.update(np.asarray([frame,len(d)],dtype='<i8').tobytes());packet_hash.update(np.asarray(d,dtype='<f4').tobytes())
        parts['detections'].append(d.copy());parts['rgb'].append(embedding['rgb']);parts['ir'].append(embedding['ir']);offsets.append(offsets[-1]+len(d))
        if frame%100==0 or frame==images.frames:
            guard_storage(work)
            path=base/'cache'/split/(sequence+'.progress.json');path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(frame=frame,frames=images.frames,seconds=time.monotonic()-started,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],GT_read=False))+'\n')
    arrays={k:np.concatenate(v,axis=0)for k,v in parts.items()}
    arrays.update(offsets=np.asarray(offsets,np.int64),frame_ids=np.arange(1,images.frames+1,dtype=np.int64))
    return arrays,dict(frames=images.frames,sequence=sequence,split=split,image_delivery_sha256=image_hash.hexdigest(),detection_packet_sha256=packet_hash.hexdigest(),seconds=time.monotonic()-started,GT_read=False,future_frame_input=False,full_frame_feature_cache=False)


def replay_arrays(arrays,normal_cases,thresholds,sequence,limit=None):
    trackers={c['id']:TrackletPermanence(c,thresholds)for c in normal_cases};rows={c['id']:[]for c in normal_cases}
    for i,frame in enumerate(arrays['frame_ids'][:limit]):
        lo,hi=map(int,arrays['offsets'][i:i+2]);d=arrays['detections'][lo:hi]
        for c in normal_cases:rows[c['id']].extend(trackers[c['id']].update(sequence,int(frame),d,arrays['rgb'][lo:hi],arrays['ir'][lo:hi]))
    return rows,trackers


def engineering():
    import numpy as np
    import torch
    from torch.nn import functional as F
    from yolox.tracker.native_cue import native_input
    reg=registered();plan,study,protocol,calibration,work,checkpoint,base=locations()
    teacher,module,visual,encoders,physical,uuid=gpu_models();before={k:tensor_digest(v)for k,v in dict(teacher=teacher,module=module,**encoders).items()}
    parity={}
    with np.load(work/'native_crops/BT-003/crops.npz')as crops:
        row=int(crops['indices'][0]);rect=torch.from_numpy(crops['rectangles'][:1].copy())
        for cue in ['rgb','ir']:
            model=encoders[cue]
            with torch.inference_mode():
                x=native_input(torch.from_numpy(crops[cue][:1].copy()),rect,'cuda');raw=model.projection(model.backbone(x));actual=F.normalize(model.neck(raw),dim=-1).cpu().numpy()[0]
            with np.load(work/'native_train_features'/(cue+'_s0')/'features.npz')as data:expected=data['post_neck'][row].copy()
            # Original bank used batch64; this independent probe uses batch1.
            np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-4);parity[cue]=float(np.max(np.abs(actual-expected)))
    arrays,source=collect('train','BT-003',teacher,module,visual,encoders,frames=12)
    thresholds=read(base/'Train_thresholds.json')['thresholds'];rows,trackers=replay_arrays(arrays,reg['normal_cases'],thresholds,'BT-003')
    original=IndependentByteTrack();expected=[]
    for i,frame in enumerate(arrays['frame_ids']):
        lo,hi=map(int,arrays['offsets'][i:i+2]);expected.extend(original.update(arrays['detections'][lo:hi],int(frame)))
    assert ''.join(map(mot_line,expected))==''.join(map(mot_line,rows['TP_native_zero']))
    corrupt={k:v.copy()for k,v in arrays.items()};start=int(arrays['offsets'][6]);corrupt['rgb'][start:]*=-1;corrupt['ir'][start:]*=-1
    first,_=replay_arrays(arrays,reg['normal_cases'],thresholds,'BT-003',6);second,_=replay_arrays(corrupt,reg['normal_cases'],thresholds,'BT-003',6)
    assert {k:''.join(map(mot_line,v))for k,v in first.items()}=={k:''.join(map(mot_line,v))for k,v in second.items()}
    for k,v in dict(teacher=teacher,module=module,**encoders).items():assert tensor_digest(v)==before[k]
    registered();frozen_json(base/('gpu_engineering_'+physical+'.json'),dict(status='COMPLETE_REAL_GPU_TRACKLET_IDENTITY_CACHE_ENGINEERING',UTC=utc(),physical_gpu=physical,GPU_uuid=uuid,Train_frames=12,actual_native_Train_crop_postneck_max_abs_error=parity,zero_MOT_exact=True,future_suffix_corruption_prefix_invariant=True,all_weights_and_BN_unchanged=True,GT_or_Test_accuracy_read=False,Train_crop_cache_read=True,source=source,pipeline_registration_sha256=sha256(base/'pipeline_registered.json')))
    print(json.dumps(dict(status='COMPLETE_REAL_GPU_TRACKLET_IDENTITY_CACHE_ENGINEERING',physical_gpu=physical)),flush=True)


def require_engineering(base):
    jobs=read(base/'state.json')['phases'][PREFIX+'_GPU_ENGINEERING'];assert set(jobs)=={'4','5'}
    for gpu,job in jobs.items():
        assert job['status']=='COMPLETE' and job['exit_code']==0 and not identity(job['pid'])
        p=read(base/('gpu_engineering_'+gpu+'.json'));assert p['zero_MOT_exact'] and p['future_suffix_corruption_prefix_invariant']


def split_cases(split):
    reg=registered();plan,study,protocol,calibration,work,checkpoint,base=locations()
    if split=='dev':return reg['normal_cases']
    freeze=read(base/'Test_selection_frozen.json');assert freeze['Test_read_for_selection']is False and freeze['complete_Dev_sha256']==sha256(base/'complete_Dev.json')
    return freeze['cases']


def produce(split,sequence):
    import numpy as np
    reg=registered();plan,study,protocol,calibration,work,checkpoint,base=locations();require_engineering(base);split_cases(split)
    teacher,module,visual,encoders,physical,uuid=gpu_models();folder=base/'cache'/split;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(sequence+'.npz');meta_path=folder/(sequence+'.json');assert not path.exists() and not meta_path.exists()
    arrays,info=collect(split,sequence,teacher,module,visual,encoders)
    p,expected=anchor(work,split,sequence)
    assert info['frames']==LENGTHS[split][sequence] and info['image_delivery_sha256']==expected['original_image_bytes_delivery_sha256']
    assert info['detection_packet_sha256']==expected['streams']['NP_temporal_ir_e06']['detection_packet_sha256']
    temporary=folder/(sequence+'.tmp.npz');np.savez_compressed(temporary,**arrays);os.replace(temporary,path)
    registered();guard_storage(work);frozen_json(meta_path,dict(status='COMPLETE_CAUSAL_NATIVE_IR_PROPOSAL_DUAL_IDENTITY_BANK',**info,path=str(path),sha256=sha256(path),bytes=path.stat().st_size,source_anchor_sha256=sha256(p),physical_gpu=physical,GPU_uuid=uuid,actual_identity=identity(os.getpid()),pipeline_registration_sha256=sha256(base/'pipeline_registered.json'),source_versions=reg['source_hashes'],dtype='FP32 descriptors and proposals; no FP16 conversion',no_image_or_FPN_copy=True))
    print(json.dumps(dict(status='COMPLETE_CAUSAL_NATIVE_IR_PROPOSAL_DUAL_IDENTITY_BANK',split=split,sequence=sequence,bytes=path.stat().st_size)),flush=True)


def replay(split,sequence):
    import numpy as np
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,study,protocol,calibration,work,checkpoint,base=locations();normal=split_cases(split)
    meta=read(base/'cache'/split/(sequence+'.json'));assert sha256(meta['path'])==meta['sha256'] and meta['source_versions']==reg['source_hashes']
    with np.load(meta['path'])as bank:arrays={k:bank[k].copy()for k in bank.files}
    count=LENGTHS[split][sequence];np.testing.assert_array_equal(arrays['frame_ids'],np.arange(1,count+1));assert len(arrays['offsets'])==count+1 and arrays['offsets'][-1]==len(arrays['detections'])
    output=base/'streams'/split;writer=FusionMemoryStreamWriter(output,sequence,[c['id']for c in normal]);thresholds=read(base/'Train_thresholds.json')['thresholds']
    trackers={c['id']:TrackletPermanence(c,thresholds)for c in normal};geometry={c['id']:hashlib.sha256()for c in normal}
    try:
        for i,frame in enumerate(arrays['frame_ids']):
            frame=int(frame);lo,hi=map(int,arrays['offsets'][i:i+2]);d=arrays['detections'][lo:hi];rows={}
            for c in normal:
                rows[c['id']]=trackers[c['id']].update(sequence,frame,d,arrays['rgb'][lo:hi],arrays['ir'][lo:hi])
                for row in rows[c['id']]:
                    cells=mot_line(row).split(',');geometry[c['id']].update(','.join([cells[0],*cells[2:]]).encode())
            writer.append(frame,rows,{c['id']:d for c in normal})
        streams=writer.close(count)
    finally:writer.abort()
    p,original=anchor(work,split,sequence);parent=original['streams']['NP_temporal_ir_e06']
    for c in normal:
        assert streams[c['id']]['detection_packet_sha256']==parent['detection_packet_sha256']
        if c['id']=='TP_native_zero':assert streams[c['id']]['track_sha256']==parent['track_sha256']
        if c['buffer']==25:assert geometry[c['id']].hexdigest()==geometry['TP_native_zero'].hexdigest()
    registered();frozen_json(output/'sequence_manifests'/(sequence+'.json'),dict(status='COMPLETE_TRACKLET_CAUSAL_BANK_REPLAY',split=split,sequence=sequence,frames=count,cases=normal,streams=streams,memory_statistics={k:t.statistics for k,t in trackers.items()},nonidentity_output_rows_sha256={k:h.hexdigest()for k,h in geometry.items()},zero_MOT_exact_to_native_parent=True,same_buffer_all_output_geometry_scores_order_exact=True,bank_sha256=meta['sha256'],source_image_delivery_sha256=meta['image_delivery_sha256'],source_versions=reg['source_hashes'],only_current_frame_values_supplied_to_tracker=True,GT_read=False,future_frame_input=False,previous_outputs_rewritten=False,compact_feature_cache_read=True,encoder_and_image_cost_excluded_from_replay=True))


def evaluate(split,case_id):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,study,protocol,calibration,work,checkpoint,base=locations();normal=split_cases(split);case=next(c for c in normal if c['id']==case_id)
    coverage={}
    for seq,n in LENGTHS[split].items():
        source=read(base/'streams'/split/'sequence_manifests'/(seq+'.json'));assert source['frames']==n and source['source_versions']==reg['source_hashes'] and source['GT_read']is False
        assert source['zero_MOT_exact_to_native_parent'] and source['same_buffer_all_output_geometry_scores_order_exact']
        stream=source['streams'][case_id];path=Path(stream['track_path']);assert sha256(path)==stream['track_sha256'];validate_mot(path,n,stream['track_rows']);coverage[seq]=stream
    output=base/'evaluation'/split/case_id;assert not output.exists();output.mkdir(parents=True)
    values=run_official_tracks(base/'streams'/split/case_id/'track_results',output,plan,split)
    registered();frozen_json(output/'metrics.json',dict(status='COMPLETE_OFFICIAL_TRACKLET_PERMANENCE_METRICS',case=case,frames=sum(LENGTHS[split].values()),coverage=coverage,source_versions=reg['source_hashes'],**values,goal_met=False,supported_innovations=0))


def run_phase(queue,split):
    plan,study,protocol,calibration,work,checkpoint,base=locations();normal=split_cases(split);cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
    queue.jobs(PREFIX+'_'+split.upper()+'_NN',[dict(id=s,command=cmd+['produce','--split',split,'--sequence',s])for s in sorted(LENGTHS[split],key=lambda s:-LENGTHS[split][s])],True,lambda j:read(base/'cache'/split/(j['id']+'.json'))['frames']==LENGTHS[split][j['id']])
    queue.jobs(PREFIX+'_'+split.upper()+'_REPLAY',[dict(id=s,command=cmd+['replay','--split',split,'--sequence',s])for s in LENGTHS[split]],False,lambda j:read(base/'streams'/split/'sequence_manifests'/(j['id']+'.json'))['zero_MOT_exact_to_native_parent'])
    queue.jobs(PREFIX+'_'+split.upper()+'_OFFICIAL',[dict(id=c['id'],command=cmd+['evaluate','--split',split,'--case',c['id']])for c in normal],False,lambda j:read(base/'evaluation'/split/j['id']/'metrics.json')['frames']==sum(LENGTHS[split].values()))
    return [read(base/'evaluation'/split/c['id']/'metrics.json')for c in normal]


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,study,protocol,calibration,work,checkpoint,base=locations();ready(work)
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_TRACKLET_PERMANENCE_ENGINEERING',controller_pid=os.getpid(),phases={},data_root=plan['data_root'],physical_gpus=['4','5'],maximum_cards=2,fixed_seed=0,supported_innovations=0,goal_met=False);queue=PriorityQueue(work,state,base);queue.save()
        try:
            cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
            queue.jobs(PREFIX+'_GPU_ENGINEERING',[dict(id=g,physical_gpu=g,command=cmd+['engineering'])for g in ['4','5']],True,lambda j:read(base/('gpu_engineering_'+j['id']+'.json'))['zero_MOT_exact'])
            values=run_phase(queue,'dev');zero=values[0];expected=read(work/'fisher_memory_20261127/complete_Dev.json')['rows'][0]
            assert zero['combined47']==expected['combined47'] and zero['per_sequence47']==expected['per_sequence47']
            chosen=min(values,key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),reg['normal_cases'].index(r['case'])));gain=chosen['combined47']['HOTA']-zero['combined47']['HOTA']
            frozen_json(base/'complete_Dev.json',dict(status='COMPLETE_ALL5_TRACKLET_PERMANENCE_DEV',UTC=utc(),rows=values,chosen_case=chosen['case'],Dev_gain_pp=gain,Test_read_for_selection=False,zero_all47_exact=True,goal_met=False,supported_innovations=0))
            if gain<=0 or chosen['case']==zero['case']:
                state.update(status='COMPLETE_TRACKLET_NO_POSITIVE_DEV_GAIN_NO_NEW_TEST',Dev_gain_pp=gain);queue.save();return
            frozen_json(base/'Test_selection_frozen.json',dict(status='FROZEN_TRACKLET_PERMANENCE_FROM_COMPLETE_DEV_BEFORE_TEST',UTC=utc(),cases=[zero['case'],chosen['case']],chosen_case=chosen['case'],complete_Dev_sha256=sha256(base/'complete_Dev.json'),Test_read_for_selection=False,historical_Test_previously_exposed=True))
            test=run_phase(queue,'test');selected=next(r for r in test if r['case']==chosen['case'])
            from tools.export_perceptual_metrics47 import official_schema
            folder=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB';schema,_=official_schema(plan['evaluation']['TrackEval_root']);csv=folder/'trackeval_output/pedestrian_detailed.csv'
            original_csv=Path(plan['frozen_detection_sources']['rgb_test'])/'trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv)==sha256(original_csv)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            old_parent=work/'native_pyramid_evaluation/test/NP_temporal_ir_e06';parent=official_csv47(old_parent/'trackeval_output/pedestrian_detailed.csv',schema,'test',read(old_parent/'metrics.json')['trackeval'])
            zero_test=next(r for r in test if r['case']['id']=='TP_native_zero')
            assert zero_test['combined47']==parent['combined47'] and zero_test['per_sequence47']==parent['per_sequence47']
            baseline=official_csv47(csv,schema,'test',read(folder/'metrics.json')['trackeval']);delta=selected['combined47']['HOTA']-baseline['combined47']['HOTA']
            frozen_json(base/'full_Test_results.json',dict(status='COMPLETE_DEV_FROZEN_TRACKLET_PERMANENCE_FULL7_TEST',UTC=utc(),rows=test,original_RGB_baseline=baseline,primary_case=chosen['case'],HOTA_gain_over_original_RGB_pp=delta,target_plus5_met=delta>=5,target_plus10_met=delta>=10,Test_used_for_selection=False,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0))
            state.update(status='COMPLETE_TRACKLET_FULL_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT',HOTA_gain_over_original_RGB_pp=delta);queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_TRACKLET_PERMANENCE_ATTEMPT',error=repr(error));queue.save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run','engineering','produce','replay','evaluate']);p.add_argument('--split',choices=['dev','test']);p.add_argument('--sequence');p.add_argument('--case');a=p.parse_args()
    if a.mode=='register':register()
    elif a.mode=='run':run()
    elif a.mode=='engineering':engineering()
    elif a.mode=='produce':produce(a.split,a.sequence)
    elif a.mode=='replay':replay(a.split,a.sequence)
    else:evaluate(a.split,a.case)


if __name__=='__main__':main()
