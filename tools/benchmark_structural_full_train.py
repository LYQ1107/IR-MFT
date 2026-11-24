#!/usr/bin/env python3
"""Measure each frozen formal full-Train model/policy against its exact Dev prefix."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from tools.benchmark_structural_factorial import inputs
from tools.extract_structural_full_train import context, load_bundle
from tools.extract_structural_relative import detections
from tools.indexed_postprocess import indexed_postprocess
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.structural_relative_inference import decoded_relative_response
from yolox.models.frozen_head_repair import head_predictions
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.priority_observations import priority_union
from yolox.utils import postprocess


def predict(model, adapter, arrays, ratios, available, cue, policy):
    features = {name:model.backbone(value) for name,value in arrays['inputs'].items()}
    teacher = model.head(features['rgb'])
    if adapter is None:
        return [detections(v,r) for v,r in zip(postprocess(teacher,1,.1,.7),ratios)]
    teacher_raw = head_predictions(model.head,features['rgb'])[0]
    scores=[]
    offset=0
    for value in features['rgb']:
        count=value.shape[-2]*value.shape[-1]
        probability=teacher[:,offset:offset+count,4]*teacher[:,offset:offset+count,5]
        scores.append(probability.reshape(len(ratios),1,*value.shape[-2:]))
        offset+=count
    view='ir' if cue=='ir' else 'rgb'
    descriptors,validity=arrays['descriptors'],arrays['validity']
    common=(features['rgb'],features[cue],scores,descriptors['rgb'],descriptors[view],
            validity['rgb'],validity[view],torch.tensor(available,device='cuda'))
    current=head_predictions(model.head,adapter(*common))[0]
    null=head_predictions(model.head,adapter(*common,null=True))[0]
    output=decoded_relative_response(current,null,teacher_raw,teacher,centered=adapter.centered,bound=2.)
    assert torch.equal(output[:,:,:4],teacher[:,:,:4])
    values=[detections(v,r) for v,r in zip(indexed_postprocess(output,1,.1,.7)[0],ratios)]
    if policy=='priority':
        original=[detections(v,r) for v,r in zip(postprocess(teacher.clone(),1,.1,.7),ratios)]
        values=[priority_union(a,[(b,1.)])[0] if ok else a.copy()
                for a,b,ok in zip(original,values,available)]
    return values


def measure(model,adapter,plan,config,calibration,cue,policy,key,image_cache):
    work=Path(plan['work_root'])
    records=[]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for sequence in ['BT-005','SN-013']:
        source=work/'full_train_observations/structural_relative_predictions/dev'/sequence/'prefix_check_32'/(key+'.npz')
        with np.load(source)as z:
            reference={name:z[name] for name in ['frame_ids','offsets','boxes','scores']}
        tracker=BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),frame_rate=25)
        total=preparation=compute=tracking=0.
        count=checked=output_rows=0
        for first in range(1,33,4):
            frames=list(range(first,first+4))
            torch.cuda.synchronize()
            start=time.perf_counter()
            arrays,ratios,available=inputs(plan,config,calibration,sequence,frames,cue,image_cache)
            torch.cuda.synchronize()
            ready=time.perf_counter()
            values=predict(model,adapter,arrays,ratios,available,cue,policy)
            torch.cuda.synchronize()
            predicted=time.perf_counter()
            for value in values:
                tracks=tracker.update(value,(1,1),(1,1))
                output_rows+=sum(int(t.tlwh[2]*t.tlwh[3]>100) for t in tracks)
            torch.cuda.synchronize()
            finished=time.perf_counter()
            if first>8:
                total+=finished-start
                preparation+=ready-start
                compute+=predicted-ready
                tracking+=finished-predicted
                count+=4
            for frame,value in zip(frames,values):
                lo,hi=reference['offsets'][frame-1:frame+1]
                np.testing.assert_array_equal(value,np.c_[reference['boxes'][lo:hi],reference['scores'][lo:hi]])
                checked+=1
            del arrays,values
        assert count==24 and checked==32
        records.append(dict(sequence=sequence,timed_frames=count,warmup_frames=8,seconds=total,
            tracking_output_rows_including_warmup=output_rows,
            input_preparation_transfer_seconds=preparation,NN_NMS_policy_transfer_seconds=compute,
            tracking_seconds=tracking,source32_exact=True,source_packet_sha256=sha256(source)))
    return dict(per_sequence=records,timed_frames=48,
        FPS=48/sum(r['seconds'] for r in records),
        NN_NMS_policy_transfer_FPS=48/sum(r['NN_NMS_policy_transfer_seconds'] for r in records),
        peak_own_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_own_reserved_GPU_GiB=torch.cuda.max_memory_reserved()/2**30)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--memory-inputs-and-components',action='store_true')
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4'
    plan,study,study_path,config,config_path,policy_path=context()
    assert config['architecture']['semantic_logit_bound']==2.
    work=Path(plan['work_root'])
    assert json.loads((work/'structural_full_train_dev_state.json').read_text())['status']=='COMPLETE'
    assert not (work/'structural_full_train_test_state.json').exists(), 'Runtime must precede Test GPU work'
    spec_path=ROOT/'configs/perceptual_structural_fulltrain_test_20261123.json'
    specification=json.loads(spec_path.read_text())
    model_map={row['id']:row for row in specification['models']}
    cases=[]
    for row in specification['cases']:
        case=json.loads((ROOT/'configs/perceptual_structural_fulltrain_test_cases_20261123'/(row['id']+'.json')).read_text())
        if case.get('intervention') or (case.get('baseline') and not case.get('fresh_RGB')):
            continue
        cases.append((row['id'],case))
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.45,0)
    calibration_path=work/'correlation_cache/calibration.json'
    calibration=json.loads(calibration_path.read_text())
    assert calibration['config_sha256']==config['parent_plan_sha256']
    image_cache=None
    preload_seconds=0.
    if args.memory_inputs_and_components:
        started=time.perf_counter()
        image_cache={}
        for sequence in ['BT-005','SN-013']:
            for frame in range(1,33):
                rgb=cv2.imread(str(Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'))
                ir=cv2.imread(str(Path(plan['generated_ir_root'])/'dev'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'),-1)
                assert rgb is not None and ir is not None and ir.ndim==2 and ir.dtype==np.uint8 and ir.shape==rgb.shape[:2]
                image_cache[sequence,frame,'rgb']=rgb
                image_cache[sequence,frame,'ir']=ir
        preload_seconds=time.perf_counter()-started
    records=[]
    with torch.inference_mode():
        for identifier,case in cases:
            name=case['model_id']
            model,loaded,information,_=load_bundle(dict(models=[model_map[name]]),plan,study,study_path,config,config_path)
            adapter=loaded[name]
            cue='baseline' if case.get('baseline') else case['cue']
            policy=case.get('policy','raw')
            key=case['proposal_key']
            result=measure(model,adapter,plan,config,calibration,cue,policy,key,image_cache)
            result.update(id=identifier,cue=cue,policy=policy,model_id=name,
                checkpoint_sha256=information[name].get('checkpoint_sha256'),
                original_parameters=sum(p.numel() for p in model.parameters()),
                adapter_parameters=0 if adapter is None else sum(p.numel() for p in adapter.parameters()))
            result['total_parameters']=result['original_parameters']+result['adapter_parameters']
            records.append(result)
            del model,loaded,information,adapter
            gc.collect()
            torch.cuda.empty_cache()
    suffix='_MEMORY_COMPONENTS' if args.memory_inputs_and_components else ''
    stem=ROOT/('docs/fusion_research/PERCEPTUAL_MFT_20261123_FULL_TRAIN_SINGLE_METHOD_RUNTIME'+suffix)
    assert not stem.with_suffix('.json').exists()
    report=dict(status='COMPLETE_FROZEN_FULL_TRAIN_METHOD_RUNTIME',records=records,
        source_sha256=sha256(__file__),input_helper_sha256=sha256(ROOT/'tools/benchmark_structural_factorial.py'),
        study_sha256=sha256(study_path),Test_specification_sha256=sha256(spec_path),
        memory_inputs=args.memory_inputs_and_components,preload_seconds_excluded=preload_seconds,
        unique_Dev_frames=64,method_frame_comparisons=64*len(records),timed_method_frames=48*len(records),
        physical_gpu='4',GT_read=False,
        scope='Independent single-model B4 throughput on2Devvideos,8warmup+24timed frames/video. '
              'Every64frame proposal/policy output exactly matches frozen prefixes. '
              'Includes inputpreparation,transfers,CNN,fusion,NMS,selected raw/priority policy and ByteTrack. '
              'Memory-input condition excludes image reads/preload; both exclude IRgeneration/model loading/track-file serialization. '
              'No gate observer. Shared-host load and filecache uncontrolled; not B1latency or a real-time claim.',
        goal_met=False,supported_innovations=0)
    dump(stem.with_suffix('.json'),report)
    lines=['# Frozen full-Train methods: single-method runtime','',report['scope'],'',
        '| Case | FPS | NN/NMS/policy/transfer FPS | Parameters | Peak own allocated GiB |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in records:
        lines.append(f"| {row['id']} | {row['FPS']:.3f} | {row['NN_NMS_policy_transfer_FPS']:.3f} | {row['total_parameters']} | {row['peak_own_allocated_GPU_GiB']:.3f} |")
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],methods=len(records),memory_inputs=args.memory_inputs_and_components)))


if __name__=='__main__':
    main()
