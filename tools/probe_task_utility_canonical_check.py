#!/usr/bin/env python3
"""Audit preserved R1 partial outputs and independently reproduce all Dev metrics."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from tools.fusion_memory_evaluation import official_csv47,read,sha,validate_mot
from tools.fusion_memory_observation import LENGTHS,write_json
from tools.neck72_runtime import require_host_memory,require_storage
from tools.neck72_runtime_study import exited
from tools.observe_task_utility import run_official_tracks
from tools.probe_task_utility_observation import previous_sources_exact
from tools.task_utility_canonical_check import compare_original_tracks
from tools.task_utility_design import CANONICAL,FRESH
from tools.task_utility_study import STUDY,context

SPEC=ROOT/'configs/perceptual_task_utility_canonical_check_20261125.json'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    started=time.monotonic();spec=read(SPEC);spec_sha=sha(SPEC);sources=spec['source_hashes']
    assert all(sha(ROOT/p)==s for p,s in sources.items())
    unit=read(spec['pytest_receipt']);assert sha(spec['pytest_receipt'])==spec['pytest_receipt_sha256']
    assert unit['exit_code']==0 and unit['all_sources_unchanged'] and sha(unit['log'])==unit['log_sha256']
    assert '6 passed' in Path(unit['log']).read_text() and all(sources[p]==s for p,s in unit['source_hashes'].items())
    plan,study,protocol,calibration,work,checkpoint,formal_base=context();require_storage(work)
    require_host_memory(8*2**30,maximum_workers=1);before=previous_sources_exact()
    for p,s in spec['preserved_R1_artifact_hashes'].items():assert sha(p)==s
    failed=read(spec['preserved_R1_execution']);state=read(spec['preserved_R1_state'])
    assert failed['exit_code']==1 and sha(failed['log'])==failed['log_sha256'] and not failed['RSS_limit_exceeded']
    assert exited(failed['pid'],identity=failed['identity']) and state['identity']==failed['identity']
    assert state['status']=='FAILED_TASK_UTILITY_SOURCE_RGB_OBSERVATION_CPU_COMPONENT_PRESERVE'
    assert 'AssertionError' in state['error'] and list(state['phases'])==['actual_current_RGB_prefix']
    base=Path(spec['output_directory']);base.mkdir(exist_ok=False);proc=Path('/proc/self')
    identity=dict(pid=os.getpid(),start_ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
        command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
    observations=state['phases']['actual_current_RGB_prefix']['receipt'];prefix_stream=observations['streams'][FRESH['id']]
    assert observations['frames']==12 and observations['split']=='train' and not observations['formal_Dev_Test_sequence_coverage']
    assert observations['actual_IR_byte_frames_hashed']==12 and observations['actual_IR_frames_decoded']==0
    prefix_manifest=Path(spec['preserved_R1_directory'])/'Train_RGB_prefix/sequence_manifests/BT-003.json'
    assert read(prefix_manifest)==observations and sha(prefix_stream['track_path'])==prefix_stream['track_sha256']
    validate_mot(prefix_stream['track_path'],12,prefix_stream['track_rows'])
    comparisons={};replay_receipts={};replay_files={}
    directory=Path(spec['preserved_R1_directory'])/'canonical_Dev'
    for sequence,frames in LENGTHS['dev'].items():
        path=directory/'sequence_manifests'/(sequence+'.canonical.json');value=read(path)
        assert value['status']=='COMPLETE_TASK_UTILITY_CANONICAL_REPLAY_WITHOUT_GT' and value['frames']==frames
        assert value['split']=='dev' and value['sequence']==sequence and not value['GT_read']
        assert not value['future_frame_input'] and not value['previous_outputs_rewritten']
        assert Path(value['source_packet'])==Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(sequence+'.npz')
        assert sha(value['source_packet'])==value['source_packet_sha256']
        stream=value['streams'][CANONICAL['id']];track=directory/CANONICAL['id']/'track_results'/(sequence+'.txt')
        assert stream['frames']==frames and Path(stream['track_path'])==track and sha(track)==stream['track_sha256']
        validate_mot(track,frames,stream['track_rows'])
        old=Path(plan['frozen_detection_sources']['rgb_dev'])/'track_results'/(sequence+'.txt')
        comparisons[sequence]=compare_original_tracks(old,track);replay_receipts[sequence]=value
        for p in [path,track,Path(value['source_packet'])]:replay_files[str(p)]=sha(p)
    official_started=time.monotonic();output=base/'official_Dev'
    values=run_official_tracks(directory/CANONICAL['id']/'track_results',output,plan,'dev')
    old=Path(plan['frozen_detection_sources']['rgb_dev']);metrics=read(old/'metrics.json')['trackeval']
    previous=official_csv47(old/'trackeval_output/pedestrian_detailed.csv',values['official_schema'],'dev',metrics)
    assert values['metrics39']==metrics
    assert all(values[k]==previous[k] for k in ['combined47','per_sequence47','selection_metrics','official_CSV_sha256'])
    official_seconds=time.monotonic()-official_started
    assert all(sha(p)==s for p,s in replay_files.items()) and previous_sources_exact()==before
    assert sha(SPEC)==spec_sha and all(sha(ROOT/p)==s for p,s in sources.items())
    assert not torch.cuda.is_initialized();require_storage(work)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30;assert peak<=8
    result=dict(status='COMPLETE_TASK_UTILITY_CANONICAL_CPU_CHECK_WITH_R1_FAILURE_PRESERVED',identity=identity,
        specification_sha256=spec_sha,study_sha256=sha(STUDY),source_hashes=sources,
        actual_new_NN_inference_frames=0,actual_new_optimizer_updates=0,reused_R1_Train_RGB_prefix_frames=12,
        actual_new_official_Dev_evaluated_frames=4834,all39_headlines_47_persequence_combined_and276_CSV_exact=True,
        original_vs_replayed_tracks=comparisons,baseline_official_reproduction=values,
        reused_R1_canonical_replay_receipts=replay_receipts,reused_R1_files_unchanged=replay_files,
        R1_execution=failed,R1_state_sha256=sha(spec['preserved_R1_state']),R1_failure_preserved=True,
        reused_Train_RGB_prefix_receipt=observations,original_source_audit=before,actual_device='cpu',
        separate_official_evaluation_seconds=official_seconds,seconds=time.monotonic()-started,peak_process_RSS_GiB=peak,
        no_original_or_replayed_track_ID_rewritten=True,formal_GPU_eligibility=False,new_GPU_work=False,
        actual_TaskUtility_fitted_checkpoint_loaded=False,new_TaskUtility_Dev_Test_accuracy=False,
        baseline_reproduction_only=True,full_workflow_registered=False,goal_met=False,supported_innovations=0,
        old_CPU_numerical_anomaly_cause_still_unknown=True,
        scope='R1literal-file assertion failure explained by original global versus per-video IDs only; all non-ID fields and fixed video-wide identity bijection exact. Independent official47/whole276column CSV exact. No repeated NN/replay,no changed outputs or metrics tolerance,no new method accuracy or complete workflow qualification.')
    write_json(base/'complete.json',result,exclusive=True)
    print(json.dumps(dict(status=result['status'],HOTA=values['selection_metrics']['HOTA'],seconds=result['seconds'],
        peak_process_RSS_GiB=peak,separate_official_evaluation_seconds=official_seconds)),flush=True)


if __name__=='__main__':main()
