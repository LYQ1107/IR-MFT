#!/usr/bin/env python3
"""Individual frozen-cell real-image prefix cost with literal track checks."""
import gc
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from tools.evaluate_fusion_factorial import FRESH
from tools.evaluate_native_pyramid import read_image
from tools.fusion_factorial_workflow import SCIENCE,context,load_modules,read,require_gpu
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line
from tools.native_pyramid_training_loop import state_sha256
from tools.run_pseudo_ir_comparison import dump,sha256


def track_prefix(path,through_frame):
    if not isinstance(through_frame,int) or through_frame<1:
        raise ValueError('A positive causal prefix length is required')
    rows=[];previous=0
    with Path(path).open('rb') as stream:
        for line in stream:
            frame=int(line.split(b',',1)[0])
            if frame<1 or frame<previous:raise ValueError('Source tracks are not in causal frame order')
            if frame>through_frame:break
            rows.append(line);previous=frame
    return b''.join(rows)


def main():
    plan,study,protocol,calibration,work,checkpoint,base=context()
    _,_,_,physical,uuid=require_gpu(work)
    freeze_path=base/'Test_selection_frozen.json';freeze=read(freeze_path)
    assert freeze['status']=='FROZEN_FOUR_FACTORIAL_CELLS_FROM_COMPLETE_DEV'
    assert freeze['complete_all_four_Dev'] and not freeze['Test_accuracy_used_for_selection']
    coverage_path=base/'Test_coverage_plan.json';coverage=read(coverage_path)
    assert coverage['Test_selection_sha256']==sha256(freeze_path)
    output=base/'individual_runtime.json';assert not output.exists()
    from tools.native_pyramid_experiment import load_teacher
    teacher=load_teacher(checkpoint,training=False)
    original_state=state_sha256(teacher.state_dict());rows=[]
    conditions=[dict(FRESH)]+freeze['selected_unique_models']
    for case in conditions:
        modules,info=load_modules([case],study,'cuda')
        engine=OnlineNativeDetector(teacher,[case],modules,study['inference_input_hw'],calibration)
        initial_module=None if modules[case['id']] is None else state_sha256(modules[case['id']].state_dict())
        original=coverage['sources'][case['id']]
        manifest=read(Path(original['metrics_json']).parent/'inference_manifest.json')
        assert manifest['split']=='test' and manifest['frames']==15985
        intervals,sequences=[],[]
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        for sequence in protocol['historical_test']:
            # The new full-Test stream's image digest was checked against the
            # parent stream. This binds timed prefixes even for reused cells.
            anchor=read(base/'streams/test/sequence_manifests'/(sequence+'.json'))
            assert anchor['actual_RGB_IR_bytes_digest_exact_parent']
            expected=anchor['first32_image_content'];assert len(expected)==32
            reference=manifest['sources'][sequence]
            assert sha256(reference['manifest_path'])==reference['manifest_sha256']
            producer=read(reference['manifest_path'])
            stream=producer['streams'][original['source_case_id']]
            assert sha256(stream['track_path'])==stream['track_sha256']
            tracker=IndependentByteTrack();actual=[]
            for frame in range(1,33):
                torch.cuda.synchronize();started=time.perf_counter()
                rgb,rgb_sha=read_image(Path(plan['data_root'])/'test'/sequence/'visible'/f'{frame:06d}.jpg')
                ir,ir_sha=None,None
                if engine.needs_actual_IR_image:
                    ir,ir_sha=read_image(Path(plan['generated_ir_root'])/'test'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png',scalar=True)
                packets,_=engine.step(rgb,ir,sequence,frame)
                tracked=tracker.update(packets[case['id']],frame)
                torch.cuda.synchronize();seconds=time.perf_counter()-started
                if frame>8:intervals.append(seconds)
                actual.extend(mot_line(row).encode() for row in tracked)
                assert expected[frame-1][0]==frame and rgb_sha==expected[frame-1][1]
                if engine.needs_actual_IR_image:assert ir_sha==expected[frame-1][2]
            assert b''.join(actual)==track_prefix(stream['track_path'],32)
            sequences.append(dict(sequence=sequence,prefix_tracks_literal_exact=True,
                current_prefix_RGB_and_consumed_IR_bytes_exact=True))
        assert len(intervals)==168 and state_sha256(teacher.state_dict())==original_state
        if initial_module is not None:assert state_sha256(modules[case['id']].state_dict())==initial_module
        rows.append(dict(id=case['id'],sequences=sequences,measured_frames=168,input_frames=224,
            warmup_frames_each_sequence=8,seconds=sum(intervals),FPS=168/sum(intervals),
            mean_latency_ms=1000*sum(intervals)/168,median_latency_ms=1000*sorted(intervals)[84],
            peak_own_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_own_reserved_GPU_GiB=torch.cuda.max_memory_reserved()/2**30,
            RGB_parameters=sum(p.numel() for p in teacher.parameters()),
            auxiliary_parameter_inventory=0 if initial_module is None else sum(p.numel() for p in modules[case['id']].parameters()),
            physical_gpu=physical,GPU_uuid=uuid,independent_input_batch=1,
            includes_current_image_IO_byte_hash_preprocessing_RGB_native_fusion_required_flow_NMS_ByteTrack=True,
            includes_weight_loading=False,includes_offline_IR_generation=False,
            gate_instrumentation_enabled=False,valid_for_frozen_accuracy_prefix=True))
        del engine,modules,info;gc.collect();torch.cuda.empty_cache()
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30;assert peak<=8
    dump(output,dict(status='COMPLETE_FROZEN_FACTORIAL_INDIVIDUAL_PREFIX_RUNTIME',rows=rows,
        science_sha256=sha256(SCIENCE),Test_selection_sha256=sha256(freeze_path),
        Test_coverage_plan_sha256=sha256(coverage_path),peak_cumulative_process_RSS_GiB=peak,
        scope='All frozen cells plus fresh RGB, separately on first32frames of all7Test videos,8warmup/24measured. Literal causal track prefixes and actual consumed image bytes checked. Not full15985frame FPS, offline generation cost, repeated-seed variance or per-case hostRSS.',
        GT_read=False,Test_accuracy_used_for_selection=False,goal_met=False,supported_innovations=0))


if __name__=='__main__':main()
