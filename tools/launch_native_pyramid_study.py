#!/usr/bin/env python3
"""Matched full-Train/Dev/conditional-removal/frozen-Test native study queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.native_pyramid_design import CANONICAL, FRESH, register_trial, snapshot_row, stage_models, test_models
from tools.native_pyramid_experiment import FIT_CONFIGS, STUDY, context, frozen_json, make_arm, predecessor_ready, read, registry
from tools.native_pyramid_study_rules import best_dev, budget_decision, select_models
from tools.run_pseudo_ir_comparison import dump, sha256


def metric_path(work, split, key):
    return Path(work) / 'native_pyramid_evaluation' / split / key / 'metrics.json'


def metric_row(work, split, case):
    path = metric_path(work, split, case['id'])
    data = read(path)
    assert data['status'] == 'COMPLETE' and data['manifest']['config_sha256'] == case['config_sha256']
    row = dict(read(case['config']), split=split, metrics=data['trackeval'], source_metrics_sha256=sha256(path),
        source_metrics=str(path), sequences=data['manifest']['sequences'], frames=data['manifest']['frames'])
    return row


def verify_probe(work, fit_id):
    data = read(Path(work) / 'native_pyramid_engineering' / fit_id / 'proof.json')
    assert data['status'] == 'COMPLETE_REAL_GPU_NATIVE_TRAIN_AND_B1_INFERENCE_ENGINEERING'
    assert data['metadata']['fit_config_sha256'] == sha256(FIT_CONFIGS / (fit_id+'.json'))
    assert data['updates'] == 2 and data['delivered_Train_frames'] == 8
    assert data['frozen_RGB_weights_and_BN_unchanged_before_head_precision_conversion']
    assert all(all(row[k] for k in ['direct_B1_source_exact', 'zero_full_head_exact',
        'trained_unavailable_full_head_exact']) for row in data['source_and_missing_identity'])


def verify_fit(work, fit_id, stage):
    path = Path(work) / 'native_pyramid_fits' / fit_id / 'progress.json'
    data = read(path)
    assert data['completed_full_Train_epochs'] >= stage
    assert data['metadata']['fit_config_sha256'] == sha256(FIT_CONFIGS / (fit_id+'.json'))
    assert all(h['delivery']['complete_all_Train_frames'] and h['frozen_RGB_exact_unchanged']
               and h['delivery']['frames'] == 27251 and h['delivery']['clips'] == 6814
               for h in data['history'][:stage])
    assert len(data['history'][:stage]) == stage
    for epoch in range(1, stage+1):
        assert (path.parent / f'epoch_{epoch:02d}.pt').exists()


class Queue:
    def __init__(self, work, state):
        self.work, self.state, self.active = Path(work), state, {}
        self.path = self.work / 'native_pyramid_study_state.json'

    def save(self):
        dump(self.path, self.state)

    def jobs(self, phase, tasks, gpu, check):
        self.state['status'], self.state['phase'] = phase, phase
        entries = self.state.setdefault('phases', {}).setdefault(phase, {})
        for task in tasks:
            assert task['id'] not in entries, 'Resume requires inspecting the preserved queue state first'
            entries[task['id']] = dict(task, status='PENDING')
        self.save()
        try:
            while True:
                registry(self.work)
                for slot, (key, process) in list(self.active.items()):
                    code = process.poll()
                    if code is None: continue
                    del self.active[slot]
                    entry = entries[key]
                    entry['exit_code'] = code
                    if code:
                        entry['status'] = 'FAILED'
                        raise RuntimeError('Preserve native child failure: '+entry['log'])
                    if check(entry) is False:
                        raise RuntimeError('Native job completed but its required evidence failed: '+key)
                    entry['status'] = 'COMPLETE'
                slots = ['4', '5'] if gpu else ['CPU0', 'CPU1']
                for slot in slots:
                    if slot in self.active: continue
                    pending = [(key, value) for key, value in entries.items() if value['status'] == 'PENDING'
                        and (not gpu or value.get('physical_gpu', slot) == slot)]
                    if not pending: continue
                    key, entry = pending[0]
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=slot if gpu else '',
                        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8', MFT26IR_ROOT=self.state['data_root'])
                    log = self.work / 'native_pyramid_logs' / (phase+'_'+key+'.log')
                    log.parent.mkdir(exist_ok=True)
                    with log.open('xb', buffering=0) as handle:
                        process = subprocess.Popen(entry['command'], cwd=ROOT, env=environment,
                            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    entry.update(status='RUNNING', pid=process.pid, log=str(log), slot=slot)
                    self.active[slot] = key, process
                self.save()
                if not self.active:
                    assert all(e['status'] == 'COMPLETE' for e in entries.values())
                    return
                time.sleep(5)
        except Exception:
            for key, process in self.active.values():
                if process.poll() is None: process.terminate()
            self.save()
            raise


def worker(mode, fit_id, stage=None):
    command = [sys.executable, '-u', 'tools/train_native_pyramid.py', mode, '--fit-id', fit_id]
    if stage is not None: command += ['--through-epoch', str(stage)]
    return command


def run_trial(queue, plan, protocol, phase, split, models, include_source_cases, anchor=None):
    from tools.evaluate_native_pyramid import lengths
    work = queue.work
    specification_path, cases = register_trial(work, phase, split, models, include_source_cases)
    if split == 'test':
        # The immutable selector writes this before producing any Test frame.
        freeze = read(work / 'native_pyramid_Test_selection_frozen.json')
        assert freeze['specification_sha256'] == sha256(specification_path)
    specification = read(specification_path)
    folder = Path(specification['stream_directory'])
    seq_lengths = lengths(plan, protocol, split)
    tasks = [dict(id=seq, command=[sys.executable, '-u', 'tools/evaluate_native_pyramid.py', 'produce',
        '--split', split, '--sequence', seq, '--specification', str(specification_path)])
        for seq in sorted(seq_lengths, key=lambda s: -seq_lengths[s])]
    def check_producer(entry):
        data = read(folder / 'sequence_manifests' / (entry['id']+'.json'))
        assert data['status'] == 'COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT'
        assert data['specification_sha256'] == sha256(specification_path)
        assert data['frames'] == seq_lengths[entry['id']] and not data['GT_read']
        assert set(data['streams']) == {r['id'] for r in models}
        for row in data['streams'].values(): assert sha256(row['track_path']) == row['track_sha256']
        if anchor is not None:
            old = read(Path(anchor) / 'sequence_manifests' / (entry['id']+'.json'))
            for key in ['track_sha256', 'detection_packet_sha256']:
                assert data['streams'][FRESH['id']][key] == old['streams'][FRESH['id']][key]
    queue.jobs(phase+'_NN', tasks, True, check_producer)
    if include_source_cases:
        command = [sys.executable, '-u', 'tools/evaluate_native_pyramid.py', 'canonical',
                   '--split', split, '--specification', str(specification_path)]
        def check_canonical(entry):
            for seq in seq_lengths:
                data = read(folder / 'sequence_manifests' / (seq+'.canonical.json'))
                assert data['status'] == 'COMPLETE_NATIVE_CANONICAL_SOURCE_WITHOUT_GT'
                assert data['frames'] == seq_lengths[seq]
        queue.jobs(phase+'_CANONICAL', [dict(id='source', command=command)], False, check_canonical)
    tasks = [dict(id=c['id'], config=c['config'], config_sha256=c['config_sha256'], command=[sys.executable,
        '-u', 'tools/evaluate_native_pyramid.py', 'evaluate', '--split', split, '--config', c['config']]) for c in cases]
    sources = [t for t in tasks if t['id'] in [FRESH['id'], CANONICAL['id']]]
    if sources:
        queue.jobs(phase+'_SOURCE_OFFICIAL', sources, False, lambda entry: metric_row(work, split, entry))
    tasks = [t for t in tasks if t not in sources]
    queue.jobs(phase+'_OFFICIAL', tasks, False, lambda entry: metric_row(work, split, entry))
    rows = [metric_row(work, split, case) for case in cases]
    fresh = read(metric_path(work, split, FRESH['id']))
    csv_fresh = metric_path(work, split, FRESH['id']).parent / 'trackeval_output/pedestrian_detailed.csv'
    for row in rows:
        if row.get('intervention') == 'unavailable':
            data = read(row['source_metrics'])
            assert row['metrics'] == fresh['trackeval']
            assert sha256(Path(row['source_metrics']).parent / 'trackeval_output/pedestrian_detailed.csv') == sha256(csv_fresh)
            assert all(data['manifest']['coverage'][seq]['output_sha256'] ==
                fresh['manifest']['coverage'][seq]['output_sha256'] for seq in seq_lengths)
    return rows, folder, specification_path


def fit_tasks(fits, mode, stage=None):
    return [dict(id=r['id'], physical_gpu=r['physical_gpu'], command=worker(mode, r['id'], stage)) for r in fits]


def source_files():
    names = [str(p.relative_to(ROOT)) for p in ROOT.glob('tools/native_pyramid_*.py')]
    names += ['tools/launch_native_pyramid_study.py', 'tools/train_native_pyramid.py',
        'tools/evaluate_native_pyramid.py', 'tools/benchmark_native_pyramid.py', 'tools/export_native_pyramid.py',
        'tools/probe_native_pyramid_workflow.py', 'tools/probe_native_pyramid_canonical_dev.py',
        'tools/structural_train_image_reader.py', 'tools/build_correlation_cache.py',
        'tools/extract_structural_relative.py', 'tools/indexed_postprocess.py', 'tools/eval_mft26ir.py',
        'tools/structural_legacy_backfill.py', 'tools/summarize_pseudo_ir_comparison.py',
        'yolox/models/native_pyramid_fusion.py', 'yolox/models/frozen_head_repair.py',
        'yolox/utils/boxes.py', 'yolox/exp/yolox_base.py', 'exps/yolox_mft26ir_visible.py',
        'configs/perceptual_gpu_handoff_20261123.json', 'configs/perceptual_single_seed_policy_20261123.json',
        'configs/perceptual_mft_20261122.json', 'configs/cfp/protocol_v2.json']
    names += [str(p.relative_to(ROOT)) for p in (ROOT / 'yolox/models').glob('*.py')]
    names += [str(p.relative_to(ROOT)) for p in (ROOT / 'yolox/tracker').glob('*.py')]
    names += [str(p.relative_to(ROOT)) for p in FIT_CONFIGS.glob('*.json')]
    return sorted(set(names))


def register():
    from tools.native_pyramid_training_loop import guard_storage
    plan, study, protocol, calibration, work, checkpoint = context()
    assert all((FIT_CONFIGS / (r['id']+'.json')).exists() for r in study['normal_fits'])
    # Import targets without calling any CUDA function during registration.
    from tools import benchmark_native_pyramid, export_native_pyramid
    names = source_files()
    proof_path = work / 'native_pyramid_CPU_workflow_tests.json'
    proof = read(proof_path)
    assert proof['status'] == 'COMPLETE_CPU_NATIVE_WORKFLOW_TESTS'
    assert all(sha256(ROOT / p) == h for p, h in proof['tested_source_hashes'].items())
    sources = {n: sha256(ROOT / n) for n in sorted(set(names))}
    assert set(sources).issubset(proof['tested_source_hashes']), 'Freeze source versions covered by the CPU workflow run'
    canonical_proof = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_CANONICAL_DEV_ENGINEERING.json'
    canonical = read(canonical_proof)
    assert canonical['status'] == 'COMPLETE_NATIVE_STREAM_CANONICAL_ALL4834DEV_ALL39_CSV_EXACT'
    assert all(sha256(ROOT / name) == digest for name, digest in canonical['source_hashes'].items())
    value = dict(status='COMPLETE_REGISTERED_NATIVE_DEV_TEST_PIPELINE_PENDING_GPU_ENGINEERING',
        study_sha256=sha256(STUDY), source_hashes=sources,
        fit_configs={r['id']: sha256(FIT_CONFIGS / (r['id']+'.json')) for r in study['normal_fits']},
        artifact_hashes={str(p): sha256(p) for p in [STUDY, checkpoint,
            work / 'correlation_cache/calibration.json', work / study['Train_length_source'], proof_path, canonical_proof]},
        pending_actual_GPU_engineering=True, full_Train_Dev_Test_pipeline_registered=True,
        fixed_seed=0, maximum_physical_cards=2, physical_gpus=['4', '5'],
        generated_from_complete_Dev_only=True, no_formal_fitting_yet=True,
        goal_met=False, supported_innovations=0)
    guard_storage(work)
    frozen_json(work / 'native_pyramid_pipeline_registered.json', value)
    print(json.dumps(dict(status=value['status'], source_files=len(sources))), flush=True)


def run():
    plan, study, protocol, calibration, work, checkpoint = context()
    registration = registry(work)
    path = work / 'native_pyramid_study_state.json'
    lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists(), 'Preserve and diagnose an earlier queue before resuming'
    state = dict(status='WAITING_FOR_EXISTING_STRUCTURAL_AND_LEGACY_GPU_WORK', controller_pid=os.getpid(),
        study_sha256=sha256(STUDY), registry_sha256=sha256(work / 'native_pyramid_pipeline_registered.json'),
        source_hashes=registration['source_hashes'], data_root=plan['data_root'], phases={},
        fixed_seed=0, additional_seed_work=False, physical_gpus=['4', '5'], maximum_cards=2,
        goal_met=False, supported_innovations=0)
    queue = Queue(work, state)
    queue.save()
    try:
        while not predecessor_ready(work):
            registry(work)
            time.sleep(15)
        normal = study['normal_fits']
        queue.jobs('NORMAL_GPU_ENGINEERING', fit_tasks(normal, 'probe'), True,
                   lambda e: verify_probe(work, e['id']))
        for mode in study['modes']:
            initial = [read(work / 'native_pyramid_engineering' / r['id'] / 'proof.json')['metadata']['initial_native_sha256']
                       for r in normal if r['mode'] == mode]
            assert len(initial) == 3 and len(set(initial)) == 1
        all_dev, anchor, start_epoch, stage = [], None, 1, 6
        while True:
            dump(work / 'native_pyramid_stage_authorized.json', dict(through_epoch=stage,
                fit_ids=[r['id'] for r in normal], study_sha256=sha256(STUDY),
                authorization='Minimum registered6full epochs' if stage == 6 else 'Complete six-arm Dev extension',
                prior_decision_sha256=None if stage == 6 else sha256(work / f'native_pyramid_Dev_stage_{stage-3:02d}.json')))
            queue.jobs(f'NORMAL_TRAIN_{stage:02d}', fit_tasks(normal, 'fit', stage), True,
                       lambda e: verify_fit(work, e['id'], stage))
            models = stage_models(work, normal, start_epoch, stage, interventions=True)
            rows, folder, spec = run_trial(queue, plan, protocol, f'dev_normal_{stage:02d}',
                                           'dev', models, stage == 6, anchor)
            if anchor is None: anchor = folder
            all_dev.extend(rows)
            normal_rows = [r for r in all_dev if not r.get('baseline') and
                           r.get('intervention', 'normal') == 'normal']
            decision = budget_decision(normal_rows, stage, intervention_coverage_complete=True)
            frozen_json(work / f'native_pyramid_Dev_stage_{stage:02d}.json', dict(status='COMPLETE_ALL_REGISTERED_DEV_AT_STAGE',
                study_sha256=sha256(STUDY), rows=all_dev, decision=decision,
                all_required_off_full_metrics_CSV_tracks_exact=True, Test_read_for_selection=False))
            if not decision['extend_all_six']: break
            start_epoch, stage = stage+1, decision['next_stage']
        direct = max(plan['reference']['rgb_dev_HOTA'], read(metric_path(work, 'dev', FRESH['id']))['trackeval']['HOTA'])
        selection = select_models(normal_rows, stage, direct)
        frozen_json(work / 'native_pyramid_normal_selection_frozen.json', dict(selection=selection,
            complete_epochs_each=stage, study_sha256=sha256(STUDY),
            complete_Dev_stage_sha256=sha256(work / f'native_pyramid_Dev_stage_{stage:02d}.json'),
            Test_read_for_selection=False))
        choices, removal_fits = {}, []
        if selection['fit_registered_removals']:
            mode = selection['primary']['mode']
            removals = study['removals']['all_modes'] + (study['removals']['temporal_only'] if mode == 'temporal' else [])
            for index, removal in enumerate(removals):
                key = mode+'_'+removal
                config = make_arm(study, key, mode, 'ir', removal)
                frozen_json(FIT_CONFIGS / (key+'.json'), config)
                removal_fits.append(dict(id=key, mode=mode, cue='ir', removal=removal, physical_gpu=['4', '5'][index % 2]))
            frozen_json(work / 'native_pyramid_removals_registered.json', dict(study_sha256=sha256(STUDY),
                selection_sha256=sha256(work / 'native_pyramid_normal_selection_frozen.json'),
                fit_configs={r['id']: sha256(FIT_CONFIGS / (r['id']+'.json')) for r in removal_fits},
                fits=removal_fits, same_complete_Train_epochs=stage, Test_read_for_selection=False))
            queue.jobs('REMOVAL_GPU_ENGINEERING', fit_tasks(removal_fits, 'probe'), True,
                       lambda e: verify_probe(work, e['id']))
            dump(work / 'native_pyramid_stage_authorized.json', dict(through_epoch=stage,
                fit_ids=[r['id'] for r in removal_fits], study_sha256=sha256(STUDY),
                normal_selection_sha256=sha256(work / 'native_pyramid_normal_selection_frozen.json')))
            queue.jobs('REMOVAL_TRAIN', fit_tasks(removal_fits, 'fit', stage), True,
                       lambda e: verify_fit(work, e['id'], stage))
            models = stage_models(work, removal_fits, 1, stage, interventions=False)
            removal_rows, _, _ = run_trial(queue, plan, protocol, 'dev_removals', 'dev', models, False, anchor)
            all_dev.extend(removal_rows)
            for fit in removal_fits:
                values = [r for r in removal_rows if r.get('fit_id') == fit['id']]
                assert len(values) == stage and {r['epoch'] for r in values} == set(range(1, stage+1))
                choices[fit['removal']] = dict(Dev_best=best_dev(values),
                    paired_primary_epoch=next(r for r in values if r['epoch'] == selection['primary']['epoch']))
        frozen_json(work / 'native_pyramid_complete_Dev.json', dict(study_sha256=sha256(STUDY), rows=all_dev,
            selection=selection, removals=choices, executed_complete_epochs_each=stage,
            complete_normal_and_required_removal_Dev=True, Test_read_for_selection=False))
        models, roles = test_models(work, selection, choices)
        spec_path, cases = register_trial(work, 'test_frozen', 'test', models, True)
        frozen_json(work / 'native_pyramid_Test_selection_frozen.json', dict(status='FROZEN_FROM_COMPLETE_NATIVE_DEV_BEFORE_TEST',
            study_sha256=sha256(STUDY), complete_Dev_before_Test=True,
            complete_Dev_sha256=sha256(work / 'native_pyramid_complete_Dev.json'),
            specification_sha256=sha256(spec_path), models=models, cases=cases, roles=roles,
            selection=selection, removals=choices, complete_epochs_each=stage,
            historical_Test_previously_exposed=True, Test_read_for_selection=False))
        command = [sys.executable, '-u', 'tools/benchmark_native_pyramid.py', '--specification', str(spec_path)]
        def check_runtime(entry):
            result = read(work / 'native_pyramid_individual_runtime.json')
            assert result['status'] == 'COMPLETE_NATIVE_INDIVIDUAL_RUNTIME'
            assert result['specification_sha256'] == sha256(spec_path)
            assert len(result['rows']) == len(models)
        queue.jobs('INDIVIDUAL_RUNTIME', [dict(id='runtime', physical_gpu='4', command=command)], True,
                   check_runtime)
        run_trial(queue, plan, protocol, 'test_frozen', 'test', models, True)
        state.update(status='COMPLETE_NATIVE_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT',
                     frozen_Test_sha256=sha256(work / 'native_pyramid_Test_selection_frozen.json'))
        queue.save()
        subprocess.run([sys.executable, 'tools/export_native_pyramid.py'], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_NATIVE_STUDY', error=repr(error))
        queue.save()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['register', 'run'])
    args = parser.parse_args()
    register() if args.mode == 'register' else run()


if __name__ == '__main__':
    main()
