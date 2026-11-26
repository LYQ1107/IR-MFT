#!/usr/bin/env python3
"""Recheck a finished archive audit after its supervisor session ended143.

Keep the original missing exit record unchanged. This separate read-only
verification recomputes every official table and emits its own tool completion.
"""
import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_mft_recoverability_studies import registered,context,audit_split,require_closed_jobs
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';started=time.monotonic()
    reg=registered('archive');plan,protocol,work,risk,base,prefix,driver=context('archive')
    original_execution=base/'independent_study_audit_r1.execution.json';original_hash=sha256(original_execution)
    execution=read(original_execution);tool=read(base/'independent_study_audit_r1.actual_tool_reply.json')
    assert tool['exit_code']==143 and execution['actual_child_exit_code']is None
    assert identity(execution['identity']['PID'])is None
    for dependency in reg['dependencies']:
        e=read(dependency['execution']);assert e['identity']==dependency['identity']
        assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    own=read(base/'controller.execution.json');assert own['actual_child_exit_code']==0 and own['source_unchanged']
    state=read(base/'state.json');jobs=[]
    for phase in state['phases']:jobs.extend(require_closed_jobs(state,[phase],len(state['phases'][phase])))
    assert len(jobs)==30
    receipt=read(base/'independent_study_audit_complete.json');report=read(receipt['report'])
    assert sha256(receipt['report'])==receipt['report_sha256']and receipt['official_numeric_fields_reparsed']==6627
    main_reg,fit,models,thresholds,ev=driver.ready()
    parts=[audit_split('archive','dev',read(base/'complete_Dev.json'),main_reg,fit,ev,state),
        audit_split('archive','dev',read(base/'complete_Dev_interventions.json'),main_reg,fit,ev,state,'intervention'),
        audit_split('archive','test',read(base/'full_Test_results.json'),main_reg,fit,ev,state)]
    assert len(parts)==len(report['parts'])==3
    for fresh,old in zip(parts,report['parts']):
        assert (fresh['split'],fresh['stage'])==(old['split'],old['stage'])
        assert fresh['official_numeric_fields_reparsed']==old['official_numeric_fields_reparsed']
        for family in ['rows','references']:
            assert len(fresh[family])==len(old[family])
            for a,b in zip(fresh[family],old[family]):
                for field in ['combined47','per_sequence47','selection_metrics','official_CSV_sha256','original_columns']:
                    assert a[field]==b[field],(fresh['split'],family,field)
                assert a.get('case',a.get('id'))==b.get('case',b.get('id'))
                if family=='rows':assert a['coverage']==b['coverage']
    assert sum(p['official_numeric_fields_reparsed']for p in parts)==6627
    assert sha256(fit['model_file'])==fit['model_sha256']and sha256(original_execution)==original_hash
    registered('archive')
    value=dict(status='COMPLETE_SEPARATE_READ_ONLY_ARCHIVE_AUDIT_RECHECK',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha256(Path(__file__)),original_session_actual_tool_exit=143,
        original_audit_child_exit_not_recorded=True,original_execution_record_unchanged=True,
        original_execution_sha256=original_hash,original_report_sha256=receipt['report_sha256'],
        all6627_fields_and_all_MOT_hashes_rechecked=True,official_numeric_fields_reparsed=6627,
        original_study_jobs_actual_exit_zero=len(jobs),original_study_controller_actual_exit=0,
        model_sha256=fit['model_sha256'],model_bytes_unchanged=True,changed_selection=False,
        new_optimizer_steps=0,new_model_fits=0,new_GPU_work=False,seconds=time.monotonic()-started,
        selected_Test_HOTA_gain_pp=parts[-1]['HOTA_gain_over_original_RGB_pp'],
        final_tool_completion_is_separate_record=True,goal_met=False,supported_innovations=0)
    assert not(base/'independent_study_closure_verification.json').exists()
    frozen_json(base/'independent_study_closure_verification.json',value)
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
