#!/usr/bin/env python3
"""Register/run the complete12-arm TaskUtility study after actual F4 release."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.native_pyramid_experiment import frozen_json,read
from tools.neck72_runtime import require_host_memory,require_storage
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.task_utility_controller import execute
from tools.task_utility_pipeline import current_source_hashes,register,task
from tools.task_utility_queue import Queue,process_identity
from tools.task_utility_study import STUDY,REGISTRY,context,predecessor_ready,registered


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('The waiting controller must remain CPU only')
    plan,study,protocol,calibration,work,checkpoint,base=context();registration=registered(work)
    if registration['source_hashes']!=current_source_hashes():raise ValueError('Complete pipeline source closure changed')
    base.mkdir(parents=True,exist_ok=True);lock=(base/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (base/'state.json').exists():raise ValueError('Inspect previous state and actual children before any recovery;do not restart blindly')
    identity=process_identity(os.getpid());frozen_json(base/'controller_identity.json',identity)
    state=dict(status='WAITING_FOR_COMPLETE_F4_EXPORT_AND_ACTUAL_EXITS',controller_pid=os.getpid(),controller_identity=identity,
        study_sha256=sha256(STUDY),pipeline_registration_sha256=sha256(base/REGISTRY),phases={},worker_python=sys.executable,
        fixed_seed=0,physical_gpus=['4','5'],maximum_GPU_workers=2,maximum_CPU_workers=2,
        goal_met=False,supported_innovations=0)
    def guard():
        value=registered(work)
        if value!=registration or value['source_hashes']!=current_source_hashes():raise ValueError('Original complete workflow changed')
    queue=Queue(work,base,state,plan['data_root'],guard);queue.save()
    try:
        while not predecessor_ready(work):guard();time.sleep(10)
        while True:
            guard()
            try:
                require_storage(work,growth_bytes=int(1.5*2**30));require_host_memory(8*2**30,maximum_workers=2);break
            except RuntimeError as error:
                state.update(status='WAITING_FOR_TASK_UTILITY_RESOURCE_RESERVE',resource_wait_reason=str(error));queue.save();time.sleep(10)
        execute(queue,plan,study,protocol,registration)
        report=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_FORMAL_RESULTS.json'
        def verify_export(entry):
            value=read(report)
            if (value['status']!='COMPLETE_TASK_UTILITY_FORMAL_EXPORT_REQUIRES_GLOBAL_SCIENTIFIC_ASSESSMENT' or
                    value['pipeline_registration_sha256']!=sha256(base/REGISTRY) or value['study_sha256']!=sha256(STUDY) or
                    value['complete_Test_objects']!=26 or value['all_original_choices_preserved'] is not True):
                raise ValueError('Complete original formal result export required')
        queue.jobs('FINAL_EXPORT',[task('scientific_export','export_task_utility.py')],False,verify_export)
        row=state['phases']['FINAL_EXPORT']['scientific_export']
        dump(base/'final_export_execution.json',dict(exit_code=0,identity={k:row[k] for k in ['pid','start_ticks','command_sha256']},
            report=str(report),report_sha256=sha256(report),log=row['log'],log_sha256=sha256(row['log']),
            source_sha256=sha256(ROOT/'tools/export_task_utility.py'),actual_CPU_export=True))
        state['status']='COMPLETE_TASK_UTILITY_STUDY_REQUIRES_GLOBAL_SCIENTIFIC_ASSESSMENT';queue.save()
    except Exception as error:
        state.update(status='FAILED_PRESERVE_TASK_UTILITY_STUDY',error=repr(error));queue.save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['register','run'])
    parser.add_argument('--CPU-proof',type=Path);args=parser.parse_args()
    if args.action=='register':register(args.CPU_proof)
    else:run()
