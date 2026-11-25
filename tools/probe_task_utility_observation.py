#!/usr/bin/env python3
"""Actual source-RGB CPU observation/replay checks; no TaskUtility accuracy claim."""
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
import torch

from tools.fusion_memory_evaluation import official_csv47,read,sha
from tools.fusion_memory_model_loading import load_source_rgb_cpu
from tools.fusion_memory_observation import FixedSequenceImages,LENGTHS,write_json
from tools.fusion_memory_pipeline import current_source_hashes
from tools.fusion_memory_workflow import FusionMemoryWorkflow
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.neck72_runtime import require_host_memory,require_storage
from tools.observe_task_utility import run_official_tracks
from tools.task_utility_design import CANONICAL,FRESH
from tools.task_utility_observation import observe_sequence,replay_canonical_sequence
from tools.task_utility_study import STUDY,context,predecessor_ready,registered
from tools.task_utility_training_loop import state_sha256

SPEC=ROOT/'configs/perceptual_task_utility_observation_components_20261125.json'


def previous_sources_exact():
    source=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_ALL_SIX_FULL_TRAIN_SOURCE_AUDIT.json'
    audit=read(source);rows=[];cache={}
    def digest(path):
        path=Path(path)
        if path not in cache:cache[path]=sha(path)
        return cache[path]
    for item in audit['existing_registrations']:
        path=Path(item['registration']);assert digest(path)==item['registration_sha256']
        value=read(path)
        assert all(digest(ROOT/p)==s for p,s in value['source_hashes'].items())
        assert all(digest(ROOT/p)==s for p,s in value.get('artifact_hashes',{}).items())
        rows.append(dict(item,verified_here=True))
    old=FusionMemoryWorkflow().registered();assert old['source_hashes']==current_source_hashes()
    return dict(existing_registrations=rows,F4_source_count=len(old['source_hashes']),
        F4_registration_sha256=digest(FusionMemoryWorkflow().base/'pipeline_registered.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    started=time.monotonic();spec=read(SPEC);spec_sha=sha(SPEC);sources=spec['source_hashes']
    assert all(sha(ROOT/p)==s for p,s in sources.items())
    unit=read(spec['pytest_receipt']);assert sha(spec['pytest_receipt'])==spec['pytest_receipt_sha256']
    assert unit['exit_code']==0 and unit['all_sources_unchanged'] and sha(unit['log'])==unit['log_sha256']
    assert '11 passed' in Path(unit['log']).read_text() and all(sources[p]==s for p,s in unit['source_hashes'].items())
    plan,study,protocol,calibration,work,checkpoint,formal_base=context();assert sha(STUDY)==spec['study_sha256']
    require_host_memory(8*2**30,maximum_workers=1);require_storage(work);before=previous_sources_exact()
    try:registered(work)
    except (FileNotFoundError,ValueError) as error:eligibility=dict(complete_pipeline_registered=False,error=repr(error))
    else:raise ValueError('This independent component proof expects the incomplete original workflow')
    eligibility['predecessor_ready']=predecessor_ready(work)
    base=Path(spec['output_directory']);base.mkdir(exist_ok=False)
    proc=Path('/proc/self');identity=dict(pid=os.getpid(),start_ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
        command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
    state=dict(status='RUNNING_TASK_UTILITY_SOURCE_RGB_OBSERVATION_CPU_COMPONENT',identity=identity,
        phases={},formal_GPU_eligibility=False,new_GPU_work=False)
    def save():
        temporary=base/'state.tmp.json';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(base/'state.json')
    def resources():
        require_storage(work)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('Own component exceeded8GiB')
    def phase(name,value):
        state['phases'][name]=value;resources();save();print(json.dumps(dict(phase=name,seconds=time.monotonic()-started)),flush=True)
    save()
    try:
        prior_path=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_WORKER_CPU_ENGINEERING.json'
        prior=read(prior_path);assert sha(prior_path)==spec['prior_real_IR_worker_sha256']
        records={(r['sequence'],r['frame'],r['modality']):r['sha256'] for r in prior['image_content_records']}
        image_rows=[['BT-003',f,records['BT-003',f,'rgb'],records['BT-003',f,'ir']] for f in range(1,13)]
        expected=hashlib.sha256(b''.join(json.dumps(r,separators=(',',':')).encode()+b'\n' for r in image_rows)).hexdigest()
        teacher=load_source_rgb_cpu(checkpoint,study['source_RGB_sha256']);original=state_sha256(teacher.state_dict())
        engine=OnlineNativeDetector(teacher,[dict(FRESH)],{FRESH['id']:None},study['inference_input_hw'],calibration)
        images=FixedSequenceImages(plan,'train','BT-003',frames=12)
        def matched(streams):
            assert streams[FRESH['id']]['track_sha256']==prior['actual_record']['inference']['MOT_prefix_sha256']['direct']
        observed=observe_sequence(engine,images,[dict(FRESH)],base/'Train_RGB_prefix',expected,
            dict(actual_device='cpu',specification_sha256=spec_sha,prior_real_IR_worker_sha256=sha(prior_path),component_only=True),
            resource_check=resources,verify_streams=matched)
        assert observed['frames']==12 and observed['prefix32_source_rows']==image_rows
        assert not observed['formal_Dev_Test_sequence_coverage'] and observed['actual_IR_frames_decoded']==0
        assert observed['actual_IR_byte_frames_hashed']==12 and state_sha256(teacher.state_dict())==original
        assert all(p.grad is None for p in teacher.parameters())
        phase('actual_current_RGB_prefix',dict(frames=12,GT_read=False,previous_direct_MOT_exact=True,receipt=observed))
        replayed={};replay_started=time.monotonic()
        for sequence in LENGTHS['dev']:
            replayed[sequence]=replay_canonical_sequence(plan,'dev',sequence,base/'canonical_Dev',
                dict(actual_device='cpu',specification_sha256=spec_sha,component_only=True),resource_check=resources)
            old_track=Path(plan['frozen_detection_sources']['rgb_dev'])/'track_results'/(sequence+'.txt')
            assert replayed[sequence]['streams'][CANONICAL['id']]['track_sha256']==sha(old_track)
        replay_seconds=time.monotonic()-replay_started
        phase('complete_canonical_Dev_replay',dict(frames=4834,seconds=replay_seconds,GT_read=False,original_MOT_exact=True))
        evaluation_started=time.monotonic();output=base/'canonical_Dev_evaluation'
        values=run_official_tracks(base/'canonical_Dev'/CANONICAL['id']/'track_results',output,plan,'dev')
        old=Path(plan['frozen_detection_sources']['rgb_dev']);original_metrics=read(old/'metrics.json')['trackeval']
        old_values=official_csv47(old/'trackeval_output/pedestrian_detailed.csv',values['official_schema'],'dev',original_metrics)
        assert values['metrics39']==original_metrics
        assert all(values[k]==old_values[k] for k in ['combined47','per_sequence47','selection_metrics','official_CSV_sha256'])
        official_seconds=time.monotonic()-evaluation_started
        write_json(output/'component_metrics.json',dict(status='BASELINE_CPU_REPRODUCTION_NOT_TASK_UTILITY_GAIN',**values),exclusive=True)
        phase('complete_canonical_Dev_official47',dict(frames=4834,seconds=official_seconds,
            all39_headline_and47_sequence_combined_and276_column_CSV_exact=True,HOTA=values['selection_metrics']['HOTA']))
        assert previous_sources_exact()==before and sha(SPEC)==spec_sha and all(sha(ROOT/p)==s for p,s in sources.items())
        assert not torch.cuda.is_initialized();resources()
        value=dict(status='COMPLETE_TASK_UTILITY_RGB_OBSERVATION_AND_CANONICAL_DEV_CPU_COMPONENT',identity=identity,
            specification_sha256=spec_sha,study_sha256=sha(STUDY),source_hashes=sources,
            pytest_receipt=spec['pytest_receipt'],pytest_receipt_sha256=spec['pytest_receipt_sha256'],
            current_RGB_Train_prefix=observed,canonical_Dev_replay=replayed,baseline_official_reproduction=values,
            actual_Train_RGB_B1_frames=12,complete_canonical_Dev_frames=4834,canonical_replay_seconds=replay_seconds,
            separate_official_evaluation_seconds=official_seconds,all_original_sources_unchanged=before,
            original_RGB_weights_BN_gradients_unchanged=True,actual_device='cpu',IR_images_decoded=0,
            IR_bytes_hashed_only_for_prefix_provenance=12,actual_TaskUtility_fitted_checkpoint_loaded=False,
            eligibility_preflight=eligibility,formal_GPU_eligibility=False,new_GPU_work=False,
            new_TaskUtility_Dev_Test_accuracy=False,baseline_reproduction_only=True,full_workflow_registered=False,
            saved_weights=False,new_image_FPN_disk_cache=False,goal_met=False,supported_innovations=0,
            seconds=time.monotonic()-started,peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
            scope='Actual source-RGB12frame current-image NN/writer and entire4834Dev canonical packet replay with original official47 metrics. No learned auxiliary snapshot,formal CUDA qualification,new candidate accuracy,IR efficacy,originality or independent full-systemFPS. Prior CPU numerical anomaly cause remains unknown.')
        write_json(base/'complete.json',value,exclusive=True);state.update(status=value['status'],result_sha256=sha(base/'complete.json'));save()
        print(json.dumps(dict(status=value['status'],seconds=value['seconds'],peak_process_RSS_GiB=value['peak_process_RSS_GiB'])),flush=True)
    except Exception as error:
        state.update(status='FAILED_TASK_UTILITY_SOURCE_RGB_OBSERVATION_CPU_COMPONENT_PRESERVE',error=repr(error));save();raise


if __name__=='__main__':main()
