#!/usr/bin/env python3
"""Actual Train/B4 CPU runtime-component check; not historical-GPU equivalence."""
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

from tools.benchmark_task_utility import main as formal_runtime
from tools.fusion_memory_model_loading import load_source_rgb_cpu
from tools.fusion_memory_observation import write_json
from tools.native_pyramid_experiment import read
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line
from tools.neck72_runtime import require_host_memory,require_storage
from tools.probe_task_utility_observation import previous_sources_exact
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_design import CANONICAL
from tools.task_utility_runtime import CanonicalRGBBatch,RuntimeImages,aggregate_runtime,measure_prefix,packet_digest,literal_prefix
from tools.task_utility_study import STUDY,context,predecessor_ready
from tools.task_utility_training_loop import state_sha256
from yolox.data.data_augment import preproc
from yolox.tracker.foveated_regions import exact_half_preproc
from yolox.utils import postprocess

SPEC=ROOT/'configs/perceptual_task_utility_runtime_components_20261125.json'


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    started=time.monotonic();spec=read(SPEC);source_spec_sha=sha256(SPEC);sources=spec['source_hashes']
    assert all(sha256(ROOT/p)==s for p,s in sources.items())
    unit=read(spec['pytest_receipt']);assert sha256(spec['pytest_receipt'])==spec['pytest_receipt_sha256']
    assert unit['exit_code']==0 and unit['all_sources_unchanged'] and sha256(unit['log'])==unit['log_sha256']
    assert '17 passed' in Path(unit['log']).read_text() and all(sources[p]==s for p,s in unit['source_hashes'].items())
    plan,study,protocol,calibration,work,checkpoint,formal_base=context();assert sha256(STUDY)==spec['study_sha256']
    require_host_memory(8*2**30,maximum_workers=1);require_storage(work);before=previous_sources_exact()
    argv=list(sys.argv)
    try:
        sys.argv=['benchmark_task_utility.py','--specification',str(formal_base/'specifications/test_frozen.json'),
                  '--config',str(formal_base/'cases/test/TU_source_RGB_canonical.json')]
        try:formal_runtime()
        except FileNotFoundError as error:
            assert Path(error.filename)==formal_base/'pipeline_registered.json'
            preflight=dict(complete_pipeline_registered=False,stopped_before_any_tensor_load_or_CUDA=True,error=repr(error))
        else:raise ValueError('This component probe expects the actual incomplete pipeline to stop before CUDA')
    finally:sys.argv=argv
    assert not torch.cuda.is_initialized();preflight['actual_predecessor_ready']=predecessor_ready(work)
    base=Path(spec['output_directory']);base.mkdir(exist_ok=False);proc=Path('/proc/self')
    identity=dict(pid=os.getpid(),start_ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
        command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
    state=dict(status='RUNNING_TASK_UTILITY_RUNTIME_TRAIN_B4_CPU_COMPONENT',identity=identity,
        actual_completed_NN_batches=0,actual_NN_frame_visits=0,NN_calls=[],formal_GPU_eligibility=False)
    phase={'name':'independent_legacy_reference'}
    def save():
        temporary=base/'state.tmp.json';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(base/'state.json')
    def resources():
        require_storage(work)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('Own CPU component exceeded8GiB')
    save();handles=[]
    try:
        prototype=read(ROOT/'configs/perceptual_task_utility_prototype_20261125.json')
        content=prototype['completed_IR_content_manifest'];assert sha256(content['path'])==content['sha256']
        expected={(r['frame'],r['modality']):r['sha256'] for r in read(content['path']) if r['sequence']=='BT-003' and r['frame']<=8}
        assert len(expected)==16
        teacher=load_source_rgb_cpu(checkpoint,study['source_RGB_sha256']);frozen=state_sha256(teacher.state_dict())
        pending=[]
        def before_forward(module,inputs):
            value=inputs[0]
            assert value.shape==(4,3,*study['inference_input_hw']) and value.dtype==torch.float32
            pending.append(dict(phase=phase['name'],input_shape=list(value.shape),input_dtype=str(value.dtype),
                per_sample_input_sha256=[tensor_sha(sample) for sample in value]))
        def after_forward(module,inputs,output):
            row=pending.pop();assert output.shape==(4,23625,6) and bool(torch.isfinite(output).all())
            row.update(raw_output_shape=list(output.shape),raw_output_sha256=tensor_sha(output))
            state['NN_calls'].append(row);state['actual_completed_NN_batches']+=1;state['actual_NN_frame_visits']+=4
            save();print(json.dumps(dict(phase=row['phase'],completed_NN_batches=state['actual_completed_NN_batches'],
                NN_frame_visits=state['actual_NN_frame_visits'])),flush=True)
        handles=[teacher.register_forward_pre_hook(before_forward),teacher.register_forward_hook(after_forward)]
        reader=RuntimeImages(plan,'train','BT-003',8);tracker=IndependentByteTrack();packets=hashlib.sha256();tracks=hashlib.sha256()
        source_rows=[];mot=[];reference_counts=[]
        with torch.no_grad():
            for first in [1,5]:
                originals=[reader.read_frame(f,False) for f in range(first,first+4)];prepared=[];ratios=[]
                for offset,(rgb,ir,row) in enumerate(originals):
                    frame=first+offset;assert row[2]==expected[frame,'rgb'] and ir is None and row[3] is None
                    legacy,ratio=preproc(rgb,study['inference_input_hw'],(.485,.456,.406),(.229,.224,.225))
                    lookup,other=exact_half_preproc(rgb,study['inference_input_hw'],(.485,.456,.406),(.229,.224,.225))
                    np.testing.assert_array_equal(legacy.astype(np.float16),lookup);assert ratio==other
                    prepared.append(legacy.astype(np.float16).astype(np.float32));ratios.append(ratio)
                    source_rows.append([row[0],frame,row[2],expected[frame,'ir']])
                decoded=teacher(torch.from_numpy(np.stack(prepared)));outputs=postprocess(decoded,1,.1,.7)
                for frame,output,ratio in zip(range(first,first+4),outputs,ratios):
                    if output is None:values=np.empty((0,5),np.float32)
                    else:
                        a=output.float().cpu().numpy();values=np.column_stack([a[:,:4]/ratio,a[:,4]*a[:,5]]).astype(np.float32)
                    packet_digest(packets,frame,values);reference_counts.append(len(values))
                    lines=''.join(mot_line(r) for r in tracker.update(values,frame));tracks.update(lines.encode());mot.append(lines)
                resources()
        (base/'independent_legacy_CPU_B4_Train_MOT.txt').write_text(''.join(mot))
        reference=dict(id=CANONICAL['id'],sequence='BT-003',prefix_frames=8,source_rows=source_rows,
            packet_sha256=packets.hexdigest(),literal_MOT_sha256=tracks.hexdigest())
        made=[]
        def factory():
            phase['name']='verification' if not made else 'timing'
            engine=CanonicalRGBBatch(teacher,dict(CANONICAL),study['inference_input_hw']);made.append(engine);return engine
        measured=measure_prefix(factory,lambda:RuntimeImages(plan,'train','BT-003',8),dict(CANONICAL),'BT-003',reference,
            frames=8,warmup=4,resources=resources)
        cost=aggregate_runtime([measured]);assert cost['measured_frames']==4 and cost['NN_batch_size']==4
        assert state['actual_completed_NN_batches']==6 and state['actual_NN_frame_visits']==24
        calls=state['NN_calls']
        for offset in [0,1]:
            assert len({calls[i]['raw_output_sha256'] for i in [offset,offset+2,offset+4]})==1
            assert calls[offset]['per_sample_input_sha256']==calls[offset+2]['per_sample_input_sha256']==calls[offset+4]['per_sample_input_sha256']
        assert state_sha256(teacher.state_dict())==frozen and all(p.grad is None for p in teacher.parameters())
        assert not pending and not torch.cuda.is_initialized()
        for h in handles:h.remove()
        handles=[];assert not teacher._forward_pre_hooks and not teacher._forward_hooks
        assert previous_sources_exact()==before and sha256(SPEC)==source_spec_sha and all(sha256(ROOT/p)==s for p,s in sources.items())
        prior=work/'task_utility_observation_CPU_engineering_20261125/Train_RGB_prefix/TU_source_RGB_fresh_B1/track_results/BT-003.txt'
        old_B1_digest=literal_prefix(prior,8)
        cost.pop('FPS');resources()
        result=dict(status='COMPLETE_TASK_UTILITY_ACTUAL_TRAIN_B4_CPU_RUNTIME_COMPONENT_NOT_GPU_COST',identity=identity,
            specification_sha256=source_spec_sha,study_sha256=sha256(STUDY),source_hashes=sources,
            pytest_receipt=spec['pytest_receipt'],pytest_receipt_sha256=spec['pytest_receipt_sha256'],actual_unique_Train_frames=8,
            Train_domains=1,actual_NN_batches=6,actual_NN_frame_visits=24,actual_optimizer_updates=0,
            actual_model_precision='CPU_FP32_frozen_original_FP16_source_weights',actual_input_precision='Originalfloat64preproc->FP32->FP16->CPUFP32',
            independent_legacy_preprocessing_and_lookup_inputs_bit_exact=True,independent_raw_B4outputs_bit_exact_in_all_three_passes=True,
            independent_NMS_packets_and_literal_MOT_exact=True,source_rows=source_rows,NN_calls=calls,
            independent_reference_packet_sha256=packets.hexdigest(),independent_reference_MOT_sha256=tracks.hexdigest(),
            reference_detections_per_frame=reference_counts,prefix_measurement=measured,CPU_engineering_wall_measurement=cost,
            instrumented_NN_input_output_hash_hooks_inside_CPU_wall_measurement=True,formal_method_FPS_reported=False,
            prior_B1_CPU_8frame_MOT_sha256=old_B1_digest,current_B4_equals_prior_B1_literal_MOT=old_B1_digest==tracks.hexdigest(),
            IR_bytes_read_or_decoded=0,original_RGB_weights_BN_gradients_unchanged=True,hooks_removed=True,
            actual_formal_runtime_entry_preflight=preflight,old_protected_sources_exact=before,
            actual_device='cpu',new_GPU_work=False,formal_GPU_eligibility=False,full_workflow_registered=False,
            historical_GPU_B4_packet_equivalence_established=False,TaskUtility_fitted_snapshot_loaded=False,
            new_Dev_Test_accuracy=False,new_image_FPN_disk_cache=False,saved_weights=False,
            prior_CPU_numerical_anomaly_cause_still_unknown=True,goal_met=False,supported_innovations=0,
            seconds=time.monotonic()-started,peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
            scope='ActualRGBTrainBT003frames1-8only. Independentoriginalfloat64preproc/B4CPUYOLOXforward versusnewB4routeverification/timing,6realB4NNcalls24visits. '
                'CPUFP32andhashinstrumentation,notoriginalFP16GPU/B4packetqualification ormethodFPS. NoIRbranch,TaskUtilitysnapshot,fullworkflow,accuracyorGPUeligibility;oldCPUanomalycauseunknown.')
        write_json(base/'complete.json',result,exclusive=True);state.update(status=result['status'],result_sha256=sha256(base/'complete.json'));save()
        print(json.dumps(dict(status=result['status'],seconds=result['seconds'],peak_process_RSS_GiB=result['peak_process_RSS_GiB'])),flush=True)
    except Exception as error:
        state.update(status='FAILED_ACTUAL_TASK_UTILITY_RUNTIME_CPU_COMPONENT_PRESERVE',error=repr(error));save();raise
    finally:
        for h in handles:h.remove()


if __name__=='__main__':main()
