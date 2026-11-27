#!/usr/bin/env python3
"""Continue an already frozen Test after replacing only the failed Train audit.

The original controller and auditor retain their real failure exits. No model,
normal Dev output, case ranking or existing selection file is changed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_mft_identity_ledger as original
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.task_utility_queue import Queue
from tools.audit_mft_candidate_risk import require_closed_jobs


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=original.registered();plan,protocol,work,base,train=original.locations()
    parent=read(base/'run_r1.execution.json');state=read(base/'state.json')
    assert parent['actual_child_exit_code']==1 and parent['source_unchanged']and identity(parent['identity']['PID'])is None
    assert state['status']=='FAILED_PRESERVE_LEDGER_EVALUATION_ATTEMPT'
    assert not any('TEST'in p for p in state['phases'])and not(base/'full_Test_results.json').exists()
    bad=read(train/'audit_r1.execution.json');assert bad['actual_child_exit_code']==1 and bad['source_unchanged']
    good=read(train/'audit_r2.execution.json');audit=read(train/'independent_fit_audit.json')
    assert good['actual_child_exit_code']==0 and good['source_unchanged']and identity(good['identity']['PID'])is None
    assert audit['all6_Train_replayed']and audit['model_sha256']==read(train/'fit_complete.json')['model_sha256']
    dev=read(base/'complete_Dev.json');freeze=read(base/'Test_selection_frozen.json');control=read(base/'complete_Dev_interventions.json')
    chosen=min(dev['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
    assert chosen['case']==freeze['chosen_case']==dev['chosen_case']and dev['gain_over_parent_best_Dev_pp']>1e-8
    assert not freeze['Test_read_for_selection']and freeze['complete_Dev_sha256']==sha256(base/'complete_Dev.json')
    assert control['missing_IR_MOT_and_all47_exact_RGB']and control['no_reselection']
    require_closed_jobs(state,list(state['phases']),sum(len(p)for p in state['phases'].values()))
    continuation=base/'Test_continuation_r2';continuation.mkdir(exist_ok=True)
    paths=[base/'run_r1.execution.json',base/'state.json',base/'complete_Dev.json',base/'Test_selection_frozen.json',
        base/'complete_Dev_interventions.json',train/'audit_r1.execution.json',train/'audit_r2.execution.json',train/'independent_fit_audit.json']
    sources={**reg['source_hashes'],'tools/continue_mft_identity_ledger_Test.py':sha256(Path(__file__)),
        'tools/audit_mft_identity_ledger_r2.py':sha256(ROOT/'tools/audit_mft_identity_ledger_r2.py')}
    registration=dict(status='REGISTERED_UNCHANGED_DEV_FROZEN_TEST_AFTER_R2_TRAIN_AUDIT',UTC=original.utc(),
        source_hashes=sources,artifact_hashes={str(p):sha256(p)for p in paths},original_failed_exits_preserved=True,
        no_new_selection_or_fit=True,selected_case=freeze['chosen_case'],Test_used_for_selection=False)
    frozen_json(continuation/'registered.json',registration)
    def guard():
        original.registered()
        for n,h in sources.items():assert sha256(ROOT/n)==h,n
        for n,h in registration['artifact_hashes'].items():assert sha256(n)==h,n
    with(continuation/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(continuation/'state.json').exists()
        own=dict(status='STARTING_UNCHANGED_DEV_FROZEN_LEDGER_TEST_CONTINUATION',controller_pid=os.getpid(),phases={},seed=0)
        queue=Queue(work,continuation,own,plan['data_root'],guard);queue.save()
        try:
            test=original.phase(queue,'test');case=freeze['chosen_case'];selected=next(r for r in test if r['case']==case)
            csv=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            schema,_=original.official_schema(plan['evaluation']['TrackEval_root'])
            baseline=original.official_csv47(csv,schema,'test',read(csv.parent.parent/'metrics.json')['trackeval'])
            delta=selected['combined47']['HOTA']-baseline['combined47']['HOTA'];guard()
            frozen_json(base/'full_Test_results.json',dict(status='COMPLETE_DEV_FROZEN_LEDGER_ALL7_TEST',UTC=original.utc(),rows=test,
                selected_case=case,original_RGB_baseline=baseline,HOTA_gain_over_original_RGB_pp=delta,target_plus5_met=delta>=5,
                target_plus10_met=delta>=10,Test_used_for_selection=False,historical_Test_previously_exposed=True,
                continuation_registration_sha256=sha256(continuation/'registered.json'),original_failed_auditor_and_controller_exits_preserved=True,
                goal_met=False,supported_innovations=0))
            own.update(status='COMPLETE_LEDGER_FULL_TEST_REQUIRES_ASSESSMENT',HOTA_gain_over_original_RGB_pp=delta);queue.save()
        except Exception as error:
            own.update(status='FAILED_PRESERVE_LEDGER_TEST_CONTINUATION',error=repr(error));queue.save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run']);p.parse_args();run()
