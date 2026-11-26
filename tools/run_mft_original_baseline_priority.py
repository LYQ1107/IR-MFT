#!/usr/bin/env python3
"""Run the previously Dev-frozen primary against the original RGB baseline.

The human changed scope on 2026-11-26: defer extra controls/removals and first
measure the seven-video primary Test result. Original sources and all earlier
negative results stay intact. Test never selects the checkpoint or parameters.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.launch_native_pyramid_study import Queue, run_trial
from tools.native_pyramid_design import CANONICAL, FRESH, register_trial, snapshot_row
from tools.native_pyramid_experiment import STUDY, context, frozen_json, read, registry
from tools.run_pseudo_ir_comparison import dump, sha256

PHASE = 'test_original_baseline_priority_20261126'
BASELINE = 29.5289636096556
DEV_SHA = '874da6913aa24f7840b3cf3c80ba2360e15e37e553d3ecfd4cb4a075e032219e'
NORMAL_SELECTION_SHA = 'fd3b82e44e499f6e8d4488a5155a2ef55da75b4b4f15e6573307e1c2255857d9'
USER_INSTRUCTION = '先不用做这样的实验，只需要做我们的实验比一般的最原始的不使用红外模态的数据的跟踪baseline高至少5HOTA，或者进一步可以高10HOTA就行'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_inputs():
    plan, study, protocol, calibration, work, checkpoint = context()
    registry(work)
    dev_path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261126_NATIVE_STAGE9_FULL_DEV.json'
    assert sha256(dev_path) == DEV_SHA
    dev = read(dev_path)
    assert dev['stage'] == 9 and dev['cases'] == 92 and dev['frames'] == 4834
    assert dev['Test_results_read'] is False
    selection_path = work / 'native_pyramid_normal_selection_frozen.json'
    assert sha256(selection_path) == NORMAL_SELECTION_SHA
    selected = read(selection_path)
    assert selected['Test_read_for_selection'] is False and selected['complete_epochs_each'] == 9
    primary = selected['selection']['primary']
    assert primary['id'] == 'NP_temporal_ir_e06' and primary['cue'] == 'ir' and primary['epoch'] == 6
    assert sha256(primary['checkpoint']) == primary['checkpoint_sha256']
    assert protocol['historical_test'] and len(protocol['historical_test']) == 7
    return plan, study, protocol, work, selected, dev_path


def prepare():
    plan, study, protocol, work, selected, dev_path = check_inputs()
    directory = work / 'original_baseline_priority_20261126'
    cancellation = directory / 'original_scope_cancellation.json'
    cancelled = read(cancellation)
    assert cancelled['status'] == 'ORIGINAL_REMOVAL_SCOPE_SUPERSEDED_BY_EXPLICIT_USER'
    assert cancelled['original_controller_and_two_workers_no_longer_running'] is True
    assert cancelled['user_instruction'] == USER_INSTRUCTION
    original = read(work / 'native_pyramid_study_state.json')
    assert original['status'] == 'SUPERSEDED_BY_USER_BASELINE_PRIORITY_20261126'
    assert not (work / 'native_pyramid_Test_selection_frozen.json').exists()
    primary = selected['selection']['primary']
    models = [dict(FRESH), snapshot_row(work, primary['fit_id'], primary['epoch'])]
    spec, cases = register_trial(work, PHASE, 'test', models, True)
    complete_dev = directory / 'complete_normal_Dev_new_user_scope.json'
    frozen_json(complete_dev, dict(status='COMPLETE_NORMAL_DEV_WITH_REMOVALS_DEFERRED_BY_USER',
        UTC=utc(), study_sha256=sha256(STUDY), source_report=str(dev_path), source_report_sha256=DEV_SHA,
        original_normal_selection_sha256=NORMAL_SELECTION_SHA, primary=primary,
        user_instruction=USER_INSTRUCTION, required_extra_RGB_or_removal_trials=False,
        earlier_original_scope_completed=False, Test_read_for_selection=False))
    freeze = dict(status='FROZEN_FROM_COMPLETE_NATIVE_DEV_BEFORE_TEST', UTC=utc(),
        study_sha256=sha256(STUDY), complete_Dev_before_Test=True,
        complete_Dev_file=str(complete_dev), complete_Dev_sha256=sha256(complete_dev),
        specification_sha256=sha256(spec), models=models, cases=cases,
        roles=dict(primary_IR=primary['id'], source_RGB_fresh=FRESH['id'], source_RGB_canonical=CANONICAL['id']),
        complete_epochs_each=9, original_normal_selection_sha256=NORMAL_SELECTION_SHA,
        cancellation_receipt_sha256=sha256(cancellation), user_instruction=USER_INSTRUCTION,
        scope='Original RGB baseline and the previously Dev-frozen primary; additional RGB/removal studies deferred.',
        original_RGB_baseline_HOTA=BASELINE, required_HOTA=BASELINE+5, stretch_HOTA=BASELINE+10,
        seed=0, physical_gpus=['4', '5'], maximum_cards=2,
        historical_Test_previously_exposed=True, Test_read_for_selection=False,
        new_seed_or_Test_based_retuning=False, goal_met=False)
    frozen_json(work / 'native_pyramid_Test_selection_frozen.json', freeze)
    frozen_json(directory / 'registered.json', dict(status='REGISTERED_ORIGINAL_BASELINE_PRIORITY_FROZEN_TEST',
        UTC=utc(), specification=str(spec), specification_sha256=sha256(spec),
        freeze_sha256=sha256(work / 'native_pyramid_Test_selection_frozen.json'),
        source_sha256=sha256(Path(__file__)), user_instruction=USER_INSTRUCTION,
        cases=cases, model_count=len(models), sequences=protocol['historical_test'],
        frames=15985, original_sources_modified=False, supported_innovations=0, goal_met=False))
    print(json.dumps(dict(status='PREPARED_FROZEN_PRIMARY_AND_ORIGINAL_BASELINE', specification=str(spec),
                         cases=[case['id'] for case in cases], Test_results_read=False)), flush=True)


class PriorityQueue(Queue):
    def __init__(self, work, state, path):
        super().__init__(work, state)
        self.path = path

    def save(self):
        self.state['UTC'] = utc()
        super().save()


def run():
    plan, study, protocol, work, selected, dev_path = check_inputs()
    directory = work / 'original_baseline_priority_20261126'
    registration = read(directory / 'registered.json')
    assert registration['source_sha256'] == sha256(Path(__file__))
    freeze = read(work / 'native_pyramid_Test_selection_frozen.json')
    assert sha256(work / 'native_pyramid_Test_selection_frozen.json') == registration['freeze_sha256']
    assert sha256(registration['specification']) == registration['specification_sha256']
    path = directory / 'state.json'
    with (directory / 'state.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not path.exists(), 'Preserve any original attempt before considering a resume.'
        state = dict(status='STARTING_USER_ORIGINAL_BASELINE_PRIORITY', controller_pid=os.getpid(),
            data_root=plan['data_root'], registration_sha256=sha256(directory / 'registered.json'),
            phases={}, physical_gpus=['4', '5'], maximum_cards=2, fixed_seed=0,
            additional_seed_work=False, user_instruction=USER_INSTRUCTION,
            original_RGB_baseline_HOTA=BASELINE, required_HOTA=BASELINE+5,
            stretch_HOTA=BASELINE+10, goal_met=False)
        queue = PriorityQueue(work, state, path)
        queue.save()
        try:
            rows, folder, spec = run_trial(queue, plan, protocol, PHASE, 'test', freeze['models'], True)
            from tools.export_perceptual_metrics47 import official_schema
            from tools.fusion_memory_evaluation import official_csv47
            schema, family_counts = official_schema(plan['evaluation']['TrackEval_root'])
            tables = []
            for row in rows:
                table = official_csv47(Path(row['source_metrics']).parent / 'trackeval_output/pedestrian_detailed.csv',
                                       schema, 'test', headline=row['metrics'])
                tables.append(dict(id=row['id'], **table))
            by_id = {table['id']: table for table in tables}
            primary_id = freeze['roles']['primary_IR']
            baseline = by_id[CANONICAL['id']]['combined47']['HOTA']
            assert abs(baseline - BASELINE) < 1e-9
            gain = by_id[primary_id]['combined47']['HOTA'] - baseline
            report = dict(status='COMPLETE_PRIMARY_AND_ORIGINAL_BASELINE_FULL7_TEST', UTC=utc(),
                study_sha256=sha256(STUDY), registration_sha256=sha256(directory / 'registered.json'),
                freeze_sha256=registration['freeze_sha256'], rows=tables, official_fields=schema,
                official_family_field_counts=family_counts,
                videos=7, frames=15985, primary_IR=primary_id, original_RGB_baseline_HOTA=baseline,
                primary_IR_HOTA=by_id[primary_id]['combined47']['HOTA'], HOTA_gain_pp=gain,
                required_gain_pp=5, stretch_gain_pp=10, target_plus5_met=gain >= 5,
                target_plus10_met=gain >= 10, whole_research_goal_completed=False,
                Test_used_for_selection=False, historical_Test_previously_exposed=True,
                prior_scope_removals_and_extra_RGB_deferred_by_user=True)
            frozen_json(directory / 'full_Test_results.json', report)
            state.update(status='COMPLETE_PRIMARY_ORIGINAL_BASELINE_TEST_REQUIRES_ASSESSMENT',
                         report_sha256=sha256(directory / 'full_Test_results.json'), HOTA_gain_pp=gain)
            queue.save()
            print(json.dumps(dict(status=state['status'], HOTA_gain_pp=gain, target_plus5_met=gain >= 5)), flush=True)
        except Exception as error:
            state.update(status='FAILED_PRESERVE_BASELINE_PRIORITY_ATTEMPT', error=repr(error))
            queue.save()
            raise


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only controller required.'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'prepare', 'run'])
    args = parser.parse_args()
    if args.mode == 'check':
        check_inputs()
        print(json.dumps(dict(status='VALIDATED_ORIGINAL_DEV_FROZEN_PRIMARY', GPU_work=False,
                              Test_results_read=False, checkpoint='NP_temporal_ir_e06')))
    elif args.mode == 'prepare':
        prepare()
    else:
        run()


if __name__ == '__main__':
    main()
