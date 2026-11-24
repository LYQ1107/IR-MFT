#!/usr/bin/env python3
"""Nine matched full-Train fits, Dev-only choices and complete Test/cost queue."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.native_pyramid_experiment import read
from tools.neck72_runtime import require_host_memory, require_storage
from tools.observe_spatial_native import CANONICAL, FRESH, lengths
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.spatial_native_design import metric_row, register_trial, removal_plan, stage_models, test_models
from tools.spatial_native_rules import VARIANTS, best_dev, budget_decision, complete_curves, complete_interventions, select_models
from tools.spatial_native_study import (REGISTRY, STUDY, PIPELINE, arm_path, context, frozen_json,
    make_arm, predecessor_ready, registered, validate_history)


def source_files():
    parent = read(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_SPATIAL_CPU_PROOF.json')
    names = set(parent['source_hashes']) | set(PIPELINE) | {str(STUDY.relative_to(ROOT)),
        'tools/probe_spatial_native_worker.py', 'tools/spatial_native_design.py'}
    names.update(str(p.relative_to(ROOT)) for pattern in ['tools/spatial_native_*.py', 'tests/test_spatial_native_*.py']
                 for p in ROOT.glob(pattern))
    return sorted(names)


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint, base = context()
    worker = read(work/'native_spatial_worker_CPU_proof.json')
    proof = read(work/'native_spatial_full_workflow_CPU_proof.json')
    assert worker['status'] == 'COMPLETE_NINE_ARM_SPATIAL_WORKER_CPU_CANARY_NOT_FULL_PIPELINE'
    assert proof['status'] == 'COMPLETE_SPATIAL_FULL_WORKFLOW_CPU_PROOF_PENDING_GPU'
    assert proof['worker_proof_sha256'] == sha256(work/'native_spatial_worker_CPU_proof.json')
    sources = {name: sha256(ROOT/name) for name in source_files()}
    assert sources == proof['source_hashes']
    assert all(sources[name] == digest for name, digest in worker['source_hashes'].items())
    arms = {}
    for fit in study['normal_fits']:
        path = arm_path(base, fit['id'])
        frozen_json(path, make_arm(study, fit['id'], fit['variant'], fit['cue']))
        arms[fit['id']] = sha256(path)
    assert arms == worker['normal_arms']
    require_storage(work, growth_bytes=int(1.5*2**30))
    value = dict(status='REGISTERED_COMPLETE_NINE_ARM_SPATIAL_PIPELINE_PENDING_GPU_ENGINEERING',
        study_sha256=sha256(STUDY), source_hashes=sources, normal_arms=arms,
        artifact_hashes={str(work/'native_spatial_worker_CPU_proof.json'): sha256(work/'native_spatial_worker_CPU_proof.json'),
            str(work/'native_spatial_full_workflow_CPU_proof.json'): sha256(work/'native_spatial_full_workflow_CPU_proof.json'),
            **{str(arm_path(base, key)): h for key, h in arms.items()}},
        complete_Train_Dev_Test_cost_pipeline=True, GPU_engineering_completed=False,
        formal_fits_started=False, fixed_seed=0, physical_gpus=['4', '5'], maximum_workers=2,
        goal_met=False, supported_innovations=0)
    frozen_json(base/REGISTRY, value)
    return value


class Queue:
    def __init__(self, work, base, state, data_root):
        self.work, self.base, self.state, self.data_root = Path(work), Path(base), state, data_root
        self.active = {}

    def save(self):
        dump(self.base/'state.json', self.state)

    def jobs(self, phase, tasks, gpu, check):
        assert phase not in self.state['phases'] and not self.active
        entries = {t['id']: dict(t, status='PENDING') for t in tasks}
        assert entries and len(entries) == len(tasks)
        self.state['phases'][phase] = entries
        self.state['status'] = phase
        self.save()
        failure = None
        while True:
            try:
                registered(self.work)
            except Exception as error:
                failure = repr(error)
            for slot, (key, process) in list(self.active.items()):
                entry = entries[key]
                poll_started = time.monotonic()
                code = process.poll()
                if code is None:
                    entry['last_live_monotonic'] = poll_started
                    try:
                        status = (Path('/proc')/str(process.pid)/'status').read_text().splitlines()
                        rss = next(int(s.split()[1])*1024 for s in status if s.startswith('VmRSS:'))
                        entry['observed_peak_RSS_bytes'] = max(entry.get('observed_peak_RSS_bytes', 0), rss)
                        if rss > 8*2**30:
                            process.terminate()
                            failure = 'Own child exceeded8GiB RAM; healthy sibling retained'
                    except (FileNotFoundError, StopIteration):
                        pass
                    continue
                del self.active[slot]
                entry['exit_code'] = code
                entry['completion_observed_UTC'] = datetime.now(timezone.utc).isoformat()
                entry['worker_wall_upper_seconds'] = time.monotonic()-entry['started_monotonic']
                entry['worker_wall_lower_seconds'] = max(0., entry.get('last_live_monotonic', entry['started_monotonic'])-entry['started_monotonic'])
                if code:
                    entry['status'] = 'FAILED'
                    failure = 'Preserve failed child: '+entry['log']
                else:
                    try:
                        assert check(entry) is not False
                        entry['status'] = 'COMPLETE'
                    except Exception as error:
                        entry.update(status='FAILED_EVIDENCE', error=repr(error))
                        failure = repr(error)
            if failure is None:
                for slot in (['4', '5'] if gpu else ['CPU0', 'CPU1']):
                    if slot in self.active:
                        continue
                    pending = [(k, v) for k, v in entries.items() if v['status'] == 'PENDING'
                        and (not gpu or v.get('physical_gpu', slot) == slot)]
                    if not pending:
                        continue
                    try:
                        require_storage(self.work, growth_bytes=int(.5*2**30))
                        require_host_memory(8*2**30, maximum_workers=2 if not self.active and len(pending)>1 else 1)
                    except RuntimeError as error:
                        self.state['resource_wait_reason'] = str(error)
                        continue
                    key, entry = pending[0]
                    log = self.base/'logs'/(phase+'_'+key+'.log')
                    log.parent.mkdir(parents=True, exist_ok=True)
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=slot if gpu else '',
                        MFT26IR_ROOT=self.data_root, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
                        MKL_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    launch_started, launch_UTC = time.monotonic(), datetime.now(timezone.utc).isoformat()
                    with log.open('xb', buffering=0) as stream:
                        process = subprocess.Popen(entry['command'], cwd=ROOT, env=environment,
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    proc = Path('/proc')/str(process.pid)
                    entry.update(status='RUNNING', pid=process.pid, slot=slot, log=str(log),
                        started_monotonic=launch_started, started_UTC=launch_UTC,
                        start_ticks=int((proc/'stat').read_text().rsplit(')', 1)[1].split()[19]),
                        command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
                    self.active[slot] = key, process
                    self.state.pop('resource_wait_reason', None)
            if failure:
                self.state['failure'] = failure
            self.save()
            if failure and not self.active:
                raise RuntimeError(failure)
            if not self.active and all(e['status'] == 'COMPLETE' for e in entries.values()):
                return
            time.sleep(5)


def task(identifier, script, *args, physical=None):
    value = dict(id=identifier, command=[sys.executable, '-u', 'tools/'+script, *map(str, args)])
    if physical:
        value['physical_gpu'] = physical
    return value


def verify_probe(base, fit_id):
    value = read(Path(base)/'engineering'/fit_id/'proof.json')
    assert value['status'] == 'COMPLETE_SPATIAL_GPU_TRAIN_B1_ENGINEERING' and value['GPU_work']
    assert value['metadata']['study_sha256'] == sha256(STUDY)
    assert value['metadata']['fit_config_sha256'] == sha256(arm_path(base, fit_id))
    assert value['metadata']['registry_sha256'] == sha256(Path(base)/REGISTRY)
    row = value['record']
    assert row['actual_updates'] == 3 and row['delivered_Train_frames'] == 12
    assert row['inference_B1_frames'] == 12 and len(row['source_zero_missing_checks']) == 12
    assert all(all(r[k] for k in ['fresh_direct_packets_exact', 'zero_full_head_exact',
        'learned_missing_full_head_exact']) for r in row['source_zero_missing_checks'])
    assert row['frozen_RGB_weights_BN_unchanged_before_explicit_head_precision_conversion'] and row['inference_RGB_state_unchanged']


def verify_fit(base, fit_id, stage):
    progress = read(Path(base)/'fits'/fit_id/'progress.json')
    assert progress['completed_full_Train_epochs'] == stage and progress['registered_through_epoch'] == stage
    validate_history(dict(completed_epochs=stage, history=progress['history'], metadata=progress['metadata']),
        stage, read(arm_path(base, fit_id)))
    assert all((Path(base)/'fits'/fit_id/f'epoch_{e:02d}.pt').exists() for e in range(1, stage+1))


def run_trial(queue, plan, protocol, phase, split, models, include_sources):
    base = queue.base
    specification, cases = register_trial(base, phase, split, models, include_sources)
    directory = Path(read(specification)['stream_directory'])
    expected = lengths(plan, protocol, split)
    def check_producer(entry):
        value = read(directory/'sequence_manifests'/(entry['id']+'.json'))
        assert value['status'] == 'COMPLETE_SPATIAL_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT'
        assert value['specification_sha256'] == sha256(specification)
        assert value['frames'] == expected[entry['id']] and value['parent_fresh_packets_and_tracks_exact']
        assert set(value['streams']) == {m['id'] for m in models}
        for stream in value['streams'].values():
            assert sha256(stream['track_path']) == stream['track_sha256']
        for name, gates in value['gate_statistics'].items():
            assert gates['frames'] == expected[entry['id']] and len(gates['levels']) == 3
            assert gates['available_frames']+gates['missing_frames'] == gates['frames']
    tasks = [task(s, 'observe_spatial_native.py', 'produce', '--split', split, '--sequence', s,
        '--specification', specification) for s in sorted(expected, key=lambda s: -expected[s])]
    queue.jobs(phase+'_NN', tasks, True, check_producer)
    if include_sources:
        def check_canonical(entry):
            for sequence, count in expected.items():
                value = read(directory/'sequence_manifests'/(sequence+'.canonical.json'))
                assert value['status'] == 'COMPLETE_SPATIAL_CANONICAL_REPLAY_WITHOUT_GT'
                assert value['frames'] == count and value['specification_sha256'] == sha256(specification)
        queue.jobs(phase+'_CANONICAL', [task('source', 'observe_spatial_native.py', 'canonical', '--split', split,
            '--specification', specification)], False, check_canonical)
    sources = [c for c in cases if c['id'] in [FRESH['id'], CANONICAL['id']]]
    ordinary = [c for c in cases if c not in sources]
    for label, group in [('SOURCE_OFFICIAL', sources), ('OFFICIAL', ordinary)]:
        if not group:
            continue
        tasks = [dict(task(c['id'], 'observe_spatial_native.py', 'evaluate', '--split', split,
            '--config', c['config']), **c) for c in group]
        queue.jobs(phase+'_'+label, tasks, False, lambda entry: metric_row(base, split, entry, protocol))
    rows = [metric_row(base, split, c, protocol) for c in cases]
    fresh = read(base/'evaluation'/split/FRESH['id']/'metrics.json')
    fresh_csv = base/'evaluation'/split/FRESH['id']/'trackeval_output/pedestrian_detailed.csv'
    for row in rows:
        if row.get('intervention') == 'unavailable':
            data = read(row['source_metrics'])
            assert row['metrics39'] == fresh['trackeval']
            assert row['official_CSV_sha256'] == sha256(fresh_csv)
            assert all(data['manifest']['coverage'][s]['track_sha256'] ==
                fresh['manifest']['coverage'][s]['track_sha256'] for s in expected)
    return rows, specification, cases


def fit_tasks(fits, mode, stage=None):
    return [task(f['id'], 'train_spatial_native.py', mode, '--fit-id', f['id'],
        *(['--through-epoch', stage] if stage is not None else []), physical=f['physical_gpu']) for f in fits]


def authorize(base, fits, stage, prior=None):
    dump(Path(base)/'stage_authorized.json', dict(study_sha256=sha256(STUDY), through_epoch=stage,
        fit_ids=[f['id'] for f in fits], arm_hashes={f['id']: sha256(arm_path(base, f['id'])) for f in fits},
        prior_complete_Dev_decision_sha256=prior, Test_accuracy_used=False))


def execute(queue, plan, study, protocol):
    base, normal = queue.base, study['normal_fits']
    queue.jobs('NORMAL_GPU_ENGINEERING', fit_tasks(normal, 'probe'), True, lambda e: verify_probe(base, e['id']))
    initials = {read(base/'engineering'/f['id']/'proof.json')['metadata']['initial_native_sha256'] for f in normal}
    cpu = read(queue.work/'native_spatial_worker_CPU_proof.json')
    assert initials == {r['initial_native_sha256'] for r in cpu['records']} and len(initials) == 1
    all_dev, stage, first, prior = [], 6, 1, None
    while True:
        for variant in VARIANTS:
            fits = [f for f in normal if f['variant'] == variant]
            authorize(base, fits, stage, prior)
            queue.jobs(f'TRAIN_{variant}_{stage:02d}', fit_tasks(fits, 'fit', stage), True,
                lambda e: verify_fit(base, e['id'], stage))
            models = stage_models(base, fits, first, stage, True)
            rows, _, _ = run_trial(queue, plan, protocol, f'dev_{variant}_{stage:02d}', 'dev',
                models, include_sources=not all_dev)
            all_dev.extend(rows)
            selected = [r for r in all_dev if r.get('variant') == variant and not r.get('removal')]
            complete_curves([r for r in selected if r['intervention'] == 'normal'], stage, variants=(variant,))
            complete_interventions([r for r in selected if r['intervention'] != 'normal'], stage, variants=(variant,))
        normals = [r for r in all_dev if not r.get('baseline') and not r.get('removal') and r['intervention'] == 'normal']
        interventions = [r for r in all_dev if not r.get('baseline') and not r.get('removal') and r['intervention'] != 'normal']
        decision = budget_decision(normals, interventions, stage)
        path = base/f'complete_Dev_stage_{stage:02d}.json'
        frozen_json(path, dict(study_sha256=sha256(STUDY), rows=all_dev, decision=decision,
            complete_all9_and_interventions=True, Test_accuracy_used=False))
        if not decision['extend_all_nine']:
            break
        prior, first, stage = sha256(path), stage+1, decision['next_stage']
    source_rows = {r['id']: r for r in all_dev if r.get('baseline')}
    selection = select_models(normals, stage, source_rows[CANONICAL['id']]['selection_metrics']['HOTA'],
        source_rows[FRESH['id']]['selection_metrics']['HOTA'])
    frozen_json(base/'normal_selection_frozen.json', dict(study_sha256=sha256(STUDY), selection=selection,
        stage=stage, complete_stage_sha256=sha256(path), Test_accuracy_used=False))
    removal = removal_plan(study, base, selection)
    frozen_json(base/'removals_registered.json', dict(study_sha256=sha256(STUDY),
        normal_selection_sha256=sha256(base/'normal_selection_frozen.json'),
        new_arms={f['id']: sha256(arm_path(base, f['id'])) for f in removal['new_fits']}, plan=removal))
    if removal['new_fits']:
        fits = removal['new_fits']
        queue.jobs('REMOVAL_GPU_ENGINEERING', fit_tasks(fits, 'probe'), True, lambda e: verify_probe(base, e['id']))
        authorize(base, fits, stage, sha256(base/'normal_selection_frozen.json'))
        queue.jobs('REMOVAL_TRAIN', fit_tasks(fits, 'fit', stage), True, lambda e: verify_fit(base, e['id'], stage))
        rows, _, _ = run_trial(queue, plan, protocol, 'dev_removals', 'dev', stage_models(base, fits, 1, stage, False), False)
        all_dev.extend(rows)
    choices = {}
    for label, condition in removal['conditions'].items():
        values = [r for r in all_dev if r.get('fit_id') == condition['fit_id'] and r.get('intervention') == 'normal']
        assert len(values) == stage and {r['epoch'] for r in values} == set(range(1, stage+1))
        choices[label] = dict(Dev_best=best_dev(values), paired_primary_epoch=next(
            r for r in values if r['epoch'] == selection['primary']['epoch']))
    frozen_json(base/'complete_Dev.json', dict(study_sha256=sha256(STUDY), rows=all_dev, selection=selection,
        removals=choices, removal_provenance=removal, complete_epochs_each=stage,
        complete_all9_normal_and_required_removal_Dev=True, Test_accuracy_used=False))
    models, roles = test_models(base, selection, choices)
    specification, cases = register_trial(base, 'test_frozen', 'test', models, True)
    frozen_json(base/'Test_selection_frozen.json', dict(status='FROZEN_SPATIAL_ALL_COMPLETE_DEV_BEFORE_TEST',
        study_sha256=sha256(STUDY), complete_Dev_before_Test=True, Test_accuracy_used_for_selection=False,
        complete_Dev_sha256=sha256(base/'complete_Dev.json'), specification_sha256=sha256(specification),
        models=models, cases=cases, roles=roles, selection=selection, removals=choices,
        global_RGB_gray_roles_also_frozen_from_Dev=True, historical_Test_already_exposed=True))
    run_trial(queue, plan, protocol, 'test_frozen', 'test', models, True)
    def check_runtime(entry):
        value = read(base/'individual_runtime.json')
        assert value['status'] == 'COMPLETE_SPATIAL_INDIVIDUAL_SOURCE_MATCHED_PREFIX_RUNTIME'
        assert value['Test_selection_sha256'] == sha256(base/'Test_selection_frozen.json')
        assert {r['id'] for r in value['rows']} == {r['id'] for r in models}
        assert all(r['source_equivalence_before_timing'] and r['measured_frames'] == 168 for r in value['rows'])
    queue.jobs('INDIVIDUAL_RUNTIME', [task('cost', 'benchmark_spatial_native.py', '--specification', specification,
        physical='4')], True, check_runtime)
    queue.state['status'] = 'COMPLETE_SPATIAL_EXECUTION_PENDING_SCIENTIFIC_EXPORT'
    queue.save()


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint, base = context()
    registration = registered(work)
    base.mkdir(parents=True, exist_ok=True)
    lock = (base/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (base/'state.json').exists(), 'Inspect prior state and actual processes before resuming'
    proc = Path('/proc')/str(os.getpid())
    identity = dict(pid=os.getpid(), start_ticks=int((proc/'stat').read_text().rsplit(')', 1)[1].split()[19]),
        command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
    frozen_json(base/'controller_identity.json', identity)
    state = dict(status='WAITING_FOR_COMPLETE_PRIOR_FACTORIAL_EXPORT_AND_ACTUAL_EXITS', controller_pid=os.getpid(),
        study_sha256=sha256(STUDY), registry_sha256=sha256(base/REGISTRY), phases={},
        fixed_seed=0, maximum_workers=2, physical_gpus=['4', '5'], goal_met=False, supported_innovations=0)
    queue = Queue(work, base, state, plan['data_root'])
    queue.save()
    try:
        while not predecessor_ready(work):
            registered(work)
            time.sleep(10)
        while True:
            try:
                require_storage(work, growth_bytes=int(1.5*2**30))
                require_host_memory(8*2**30, maximum_workers=2)
                break
            except RuntimeError as error:
                state.update(status='WAITING_FOR_SPATIAL_STUDY_RESOURCE_RESERVE', resource_wait_reason=str(error))
                queue.save()
                registered(work)
                time.sleep(10)
        execute(queue, plan, study, protocol)
        subprocess.run([sys.executable, 'tools/export_spatial_native.py'], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
        state['status'] = 'COMPLETE_SPATIAL_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT'
        queue.save()
    except Exception as error:
        state.update(status='FAILED_PRESERVE_SPATIAL_STUDY', error=repr(error))
        queue.save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'run'])
    {'register': register, 'run': run}[parser.parse_args().action]()
