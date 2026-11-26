#!/usr/bin/env python3
"""Recover only CPU aggregation after all11 original Test jobs exited zero.

The earlier controller's copied baseline constant differed from its unchanged
original cached CSV by seven millionths of a HOTA point. Preserve that actual
failure; derive the baseline from the exact original CSV. No inference, fitting,
metric recomputation, checkpoint selection or prior-state rewriting is allowed.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_evaluation import official_csv47
from tools.fusion_memory_observation import LENGTHS
from tools.native_pyramid_experiment import context, frozen_json, read, registry
from tools.run_pseudo_ir_comparison import sha256

ORIGINAL_CSV_SHA = 'c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
LEGACY_CONSTANT = 29.5289636096556
PHASE = 'test_original_baseline_priority_20261126'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint = context()
    registration = registry(work)
    original = work / 'original_baseline_priority_20261126'
    output = work / 'original_baseline_priority_20261126_cpu_recovery_r1'
    assert not output.exists()
    state = read(original / 'state.json')
    execution = read(original / 'controller.execution.json')
    assert state['status'] == 'FAILED_PRESERVE_BASELINE_PRIORITY_ATTEMPT'
    assert execution['actual_child_exit_code'] == 1 and execution['source_unchanged'] is True
    assert execution['source_sha256'] == sha256(ROOT / 'tools/run_mft_original_baseline_priority.py') == 'f5579fe7dca72d734d2d030a86216dd9584f672732252c77ec40ef2720fc91bb'
    assert sha256(execution['log']) == execution['log_sha256']
    log = Path(execution['log']).read_text()
    assert 'assert abs(baseline - BASELINE) < 1e-9' in log and log.rstrip().endswith('AssertionError')
    expected = {PHASE + '_NN': set(LENGTHS['test']), PHASE + '_CANONICAL': {'source'},
        PHASE + '_SOURCE_OFFICIAL': {'NP_source_fresh_RGB', 'NP_source_canonical_RGB'},
        PHASE + '_OFFICIAL': {'NP_temporal_ir_e06'}}
    assert set(state['phases']) == set(expected)
    for phase, ids in expected.items():
        jobs = state['phases'][phase]
        assert set(jobs) == ids
        assert all(job['status'] == 'COMPLETE' and type(job.get('exit_code')) is int and job['exit_code'] == 0 for job in jobs.values())
    freeze_path = work / 'native_pyramid_Test_selection_frozen.json'
    assert sha256(freeze_path) == '65dad7840e5c8df49f15c68ce16e293581a4c7d432f395a5e9a3ccdaaa5cd8db'
    freeze = read(freeze_path)
    assert freeze['Test_read_for_selection'] is False
    assert freeze['roles']['primary_IR'] == 'NP_temporal_ir_e06'
    assert datetime.datetime.fromisoformat(freeze['UTC']) < datetime.datetime.fromisoformat(execution['UTC_started'])
    assert not (original / 'full_Test_results.json').exists()
    schema, families = official_schema(plan['evaluation']['TrackEval_root'])
    rows = []
    for case in freeze['cases']:
        assert sha256(case['config']) == case['config_sha256']
        folder = work / 'native_pyramid_evaluation/test' / case['id']
        data = read(folder / 'metrics.json')
        assert data['status'] == 'COMPLETE'
        manifest = data['manifest']
        assert manifest['config_sha256'] == case['config_sha256']
        assert manifest['config'] == read(case['config'])
        assert manifest['frames'] == 15985 and manifest['sequences'] == list(LENGTHS['test'])
        assert manifest['source_versions'] == registration['source_hashes']
        assert manifest['GT_inference_input'] is False and manifest['future_frame_input'] is False
        table = official_csv47(folder / 'trackeval_output/pedestrian_detailed.csv', schema, 'test', data['trackeval'])
        rows.append(dict(id=case['id'], **table))
    by_id = {row['id']:row for row in rows}
    canonical = work / 'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
    original_csv = Path(plan['frozen_detection_sources']['rgb_test']) / 'trackeval_output/pedestrian_detailed.csv'
    assert sha256(canonical) == sha256(original_csv) == ORIGINAL_CSV_SHA
    baseline = by_id['NP_source_canonical_RGB']['combined47']['HOTA']
    assert round(baseline, 3) == 29.529 and abs(baseline - LEGACY_CONSTANT) < .0001
    assert baseline != LEGACY_CONSTANT
    primary = by_id[freeze['roles']['primary_IR']]['combined47']['HOTA']
    delta = primary - baseline
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = dict(status='COMPLETE_PRIMARY_AND_ORIGINAL_BASELINE_FULL7_TEST', UTC=now,
        study_sha256=freeze['study_sha256'], registration_sha256=sha256(original / 'registered.json'),
        freeze_sha256=sha256(freeze_path), rows=rows, official_fields=schema, official_family_field_counts=families,
        videos=7, frames=15985, primary_IR=freeze['roles']['primary_IR'], original_RGB_baseline_HOTA=baseline,
        primary_IR_HOTA=primary, HOTA_gain_pp=delta, required_gain_pp=5, stretch_gain_pp=10,
        target_plus5_met=delta >= 5, target_plus10_met=delta >= 10, whole_research_goal_completed=False,
        Test_used_for_selection=False, historical_Test_previously_exposed=True,
        prior_scope_removals_and_extra_RGB_deferred_by_user=True, recovered_CPU_aggregation=True,
        original_controller_actual_exit_code=1, original_all11_worker_actual_exit_codes_zero=True,
        original_RGB_CSV_sha256=ORIGINAL_CSV_SHA, previous_incorrect_constant=LEGACY_CONSTANT,
        baseline_constant_difference_pp=baseline - LEGACY_CONSTANT, original_baseline_method_or_CSV_changed=False)
    proof = dict(status='COMPLETE_CPU_AGGREGATION_RECOVERY_OF_ALL11_ORIGINAL_EXIT0_JOBS', UTC=now,
        source_sha256=sha256(Path(__file__)), original_controller_actual_exit_code=1,
        original_controller_execution_sha256=sha256(original / 'controller.execution.json'),
        original_controller_state_sha256=sha256(original / 'state.json'), original_error_log_sha256=sha256(execution['log']),
        original_completed_worker_count=sum(len(jobs) for jobs in state['phases'].values()),
        every_original_inference_and_official_worker_actual_exit_zero=True,
        failure_limited_to_post_metrics_copied_constant_assertion=True, previous_constant=LEGACY_CONSTANT,
        source_original_RGB_CSV=str(original_csv), source_original_RGB_CSV_sha256=ORIGINAL_CSV_SHA,
        source_corrected_baseline_HOTA=baseline, difference_pp=baseline - LEGACY_CONSTANT,
        primary_IR_checkpoint_or_parameters_changed=False, original_Test_freeze_sha256=sha256(freeze_path),
        original_controller_failure_state_or_history_rewritten=False, new_NN_or_TrackEval_or_Train_work=False,
        fixed_seed=0, supported_innovations=0, goal_met=False)
    output.mkdir()
    for source, name in [(original / 'state.json', 'original_failure_state.json'),
        (original / 'controller.execution.json', 'original_controller_failure_execution.json'),
        (Path(execution['log']), 'original_controller_failure_stdout.txt')]:
        (output / name).write_bytes(source.read_bytes())
    frozen_json(output / 'full_Test_results.json', report)
    proof['recovered_report_sha256'] = sha256(output / 'full_Test_results.json')
    frozen_json(output / 'recovery_proof.json', proof)
    print(json.dumps(dict(status=proof['status'], original_RGB_baseline_HOTA=baseline,
        primary_IR_HOTA=primary, HOTA_gain_pp=delta, all_original_workers_actual_zero=11,
        new_NN_or_evaluation_work=False, original_failure_preserved=True)), flush=True)


if __name__ == '__main__':
    main()
