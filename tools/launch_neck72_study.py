#!/usr/bin/env python3
"""Queue the complete seed0 neck72 accuracy study after all current GPU work."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.neck72_experiment import (STUDY, REGISTRY, ALGORITHMS, context, registered, read,
    config_for, install_case, common_budget_decision, selected_cases)
from tools.neck72_runtime import (REGISTRY as WORKER_REGISTRY, gpu_predecessor_ready,
                                 require_storage, require_host_memory)
from tools.native_remaining_test_backfill import finished
from tools.run_pseudo_ir_comparison import dump, sha256


def source_files():
    _, _, work = context()
    files = set(read(work / WORKER_REGISTRY)['source_hashes'])
    files.update('tools/'+name+'.py' for name in ['neck72_experiment', 'register_neck72_study',
        'check_neck72_train_crops', 'run_neck72_replay', 'neck72_smoke_binding', 'probe_neck72_gpu',
        'launch_neck72_study', 'export_neck72_study', 'probe_neck72_study',
        'native_pyramid_content_guard', 'content_verified_train_reader',
        'export_perceptual_metrics47', 'eval_mft26ir'])
    files.update(str(p.relative_to(ROOT)) for p in (ROOT / 'yolox/tracker').glob('*.py'))
    files.update(['configs/perceptual_neck72_study_20261124.json', 'tests/test_neck72_study.py',
                  'tools/extract_perceptual_mft_features.py'])
    files.update(b['path'] for b in read(STUDY)['blueprints'].values())
    return sorted(files)


def register():
    _, _, work = context()
    path = work / REGISTRY; assert not path.exists()
    proof_path = work / 'neck72_study_CPU_proof.json'; proof = read(proof_path)
    assert proof['status'] == 'COMPLETE_CPU_NECK72_STUDY_RULES_AND_ACTUAL_PREFIX_PROOF'
    assert all(sha256(ROOT / p) == h for p, h in proof['source_hashes'].items())
    assert proof['observer_actual_old_seed0_prefix_tracks_exact'] and not proof['GPU_work']
    files = source_files(); assert set(files) == set(proof['source_hashes'])
    workers = read(work / WORKER_REGISTRY)
    artifacts = dict(workers['artifact_hashes'])
    artifacts[str(work / WORKER_REGISTRY)] = sha256(work / WORKER_REGISTRY)
    artifacts[str(proof_path)] = sha256(proof_path)
    value = dict(status='REGISTERED_NECK72_COMPLETE_ACCURACY_PIPELINE_PENDING_ACTUAL_GPU',
        study_sha256=sha256(STUDY), worker_interfaces_sha256=sha256(work / WORKER_REGISTRY),
        source_hashes={p: sha256(ROOT / p) for p in files}, artifact_hashes=artifacts,
        complete_accuracy_pipeline_implemented=True, full_system_runtime_implemented=False,
        actual_GPU_engineering_pending=True, fixed_seed=0, physical_gpus=['4', '5'],
        maximum_CPU_workers=2, formal_fits_started=False, goal_met=False, supported_innovations=0)
    require_storage(work); dump(path, value)
    print(value['status'], len(files), flush=True)


def predecessors_ready(work):
    if not gpu_predecessor_ready(work):
        return False
    path = work / 'native_remaining_test_backfill_state.json'
    report_path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_REMAINING_TEST_BACKFILL.json'
    if not path.exists() or not report_path.exists():
        return False
    state, report = read(path), read(report_path)
    if 'FAIL' in state['status']:
        raise RuntimeError('Preserve existing native50 CPU failure before neck72')
    if state['status'] != 'COMPLETE_ALL50_REMAINING_NATIVE_TEST' or report.get('status') != 'COMPLETE_ORIGINAL58_NATIVE_BEHAVIORS_SEED0_ALL7_TEST':
        return False
    specification = ROOT / 'configs/perceptual_native_remaining_test_backfill_20261124.json'
    if state.get('specification_sha256') != report.get('specification_sha256') or state.get('specification_sha256') != sha256(specification):
        raise ValueError('Expected the exact completed native50 specification/report')
    if any(len(state.get(key, {})) != 50 for key in ['dev_jobs', 'test_jobs']):
        return False
    if not finished(state['controller_pid']):
        return False
    return all(j.get('status') == 'COMPLETE' and j.get('pid') and finished(j['pid'])
               for key in ['dev_jobs', 'test_jobs'] for j in state[key].values())


class Queue:
    def __init__(self, work, state):
        self.work, self.state, self.active = work, state, {}
        self.path = work / 'neck72_study_state.json'

    def save(self):
        dump(self.path, self.state)

    def jobs(self, phase, tasks, gpu, check):
        assert tasks and phase not in self.state['phases']
        entries = {t['id']: dict(t, status='PENDING') for t in tasks}
        assert len(entries) == len(tasks)
        self.state['phases'][phase] = entries
        self.state['status'] = phase; self.save()
        slots = ['4', '5'] if gpu else ['CPU0', 'CPU1']
        while True:
            registered()
            for slot, (key, process) in list(self.active.items()):
                code = process.poll()
                if code is None:
                    continue
                del self.active[slot]; row = entries[key]; row['exit_code'] = code
                if code:
                    row['status'] = 'FAILED'; self.save()
                    raise RuntimeError('Preserve neck72 worker failure: ' + row['log'])
                try:
                    if check(row) is False:
                        raise RuntimeError('Required neck72 worker evidence is false: '+key)
                except Exception:
                    row['status'] = 'FAILED_EVIDENCE'; self.save(); raise
                row['status'] = 'COMPLETE'
            for slot in slots:
                if slot in self.active:
                    continue
                pending = [(k, j) for k, j in entries.items() if j['status'] == 'PENDING'
                           and (not gpu or j.get('physical_gpu', slot) == slot)]
                if not pending:
                    continue
                key, row = pending[0]
                reserved = sum(entries[k].get('growth_bytes', 0) for k, _ in self.active.values())
                try:
                    require_storage(self.work, reserved + row.get('growth_bytes', 0))
                    require_host_memory(8*2**30)
                except RuntimeError as error:
                    self.state['status'] = 'WAITING_RESOURCE_HEADROOM_FOR_'+phase
                    self.state['resource_wait_reason'] = str(error)
                    continue
                self.state['status'] = phase; self.state.pop('resource_wait_reason', None)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot if gpu else '',
                           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                log = self.work / ('neck72.'+phase+'.'+key+'.log')
                with log.open('xb', buffering=0) as stream:
                    process = subprocess.Popen(row['command'], cwd=ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                row.update(status='RUNNING', pid=process.pid, slot=slot, log=str(log))
                self.active[slot] = key, process
            self.save()
            if not self.active and all(j['status'] == 'COMPLETE' for j in entries.values()):
                return
            time.sleep(10)


def complete(path, status):
    value = read(path)
    assert value['status'] == status
    return value


def evaluate(queue, study, cases, split, phase):
    _, protocol, work = context()
    tasks = []
    for case in cases:
        config_path = install_case(work, case)
        tasks.append(dict(id=case['id'], config_sha256=sha256(config_path),
            command=[sys.executable, '-u', 'tools/run_neck72_replay.py', '--split', split,
                     '--config', str(config_path)], growth_bytes=64*2**20))
    def check(row):
        value = complete(work / ('neck72_'+split) / row['id'] / 'metrics.json', 'COMPLETE')
        assert len(value['trackeval']) == 39 and value['manifest']['config_sha256'] == row['config_sha256']
        assert value['manifest']['study_sha256'] == sha256(STUDY)
        assert value['manifest']['frames'] == (4834 if split == 'dev' else 15985)
    queue.jobs(phase, tasks, False, check)


def check_zero_pairs(work, study, selected, split):
    lookup = {c['id']: c for c in selected['cases']}
    groups = {}
    for case in selected['cases']:
        if case.get('intervention') == 'missing':
            groups.setdefault((case['frontend'], case['algorithm']), []).append(case['id'])
    for keys in groups.values():
        parent = work / ('neck72_'+split) / keys[0]
        for key in keys[1:]:
            folder = work / ('neck72_'+split) / key
            assert read(folder / 'metrics.json')['trackeval'] == read(parent / 'metrics.json')['trackeval']
            assert sha256(folder / 'trackeval_output/pedestrian_detailed.csv') == sha256(parent / 'trackeval_output/pedestrian_detailed.csv')
            for p in (parent / 'track_results').glob('*.txt'):
                assert sha256(folder / 'track_results' / p.name) == sha256(p)


def run():
    plan, protocol, work, study, registration = registered()
    path = work / 'neck72_study_state.json'
    lock = path.with_suffix('.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists(), 'Preserve prior controller state'
    state = dict(status='WAITING_FOR_COMPLETE_CGR_AND_NATIVE50_PROCESSES', controller_pid=os.getpid(),
        study_sha256=sha256(STUDY), phases={}, formal_fit_started=False, fixed_seed=0,
        maximum_cards=2, goal_met=False, supported_innovations=0)
    queue = Queue(work, state); queue.save()
    try:
        while not predecessors_ready(work):
            registered(); time.sleep(10)
        tasks = [dict(id=seq+'_'+cue, command=[sys.executable, '-u', 'tools/check_neck72_train_crops.py',
            '--sequence', seq, '--cue', cue]) for seq in protocol['train'] for cue in ['rgb', 'ir']]
        queue.jobs('VERIFY_ALL_CURRENT_TRAIN_CROPS', tasks, False,
            lambda row: complete(work / 'neck72_crop_verification' / row['id'].rsplit('_', 1)[0] /
                (row['id'].rsplit('_', 1)[1]+'.json'), 'COMPLETE_NECK72_ALL_CACHED_CROPS_EXACT_TO_PINNED_CURRENT_TRAIN'))
        tasks = [dict(id=seq+'_'+cue, physical_gpu='4' if cue == 'rgb' else '5',
            growth_bytes=read(work / 'native_candidate_index' / seq / 'complete.json')['observations']*512*4*(2 if cue == 'rgb' else 1)+2**20,
            command=[sys.executable, '-u', 'tools/extract_native_neck_candidate_features.py', '--sequence', seq, '--cue', cue])
            for seq in protocol['train'] for cue in ['rgb', 'ir']]
        queue.jobs('TRAIN_CANDIDATE_ENCODERS', tasks, True,
            lambda row: complete(work / 'native_neck_candidate_features' / row['id'].rsplit('_', 1)[0] /
                row['id'].rsplit('_', 1)[1] / 'complete.json', 'COMPLETE_EXACT_TRAIN_NATIVE_FEATURES'))
        for kind, worker, family, status in [
            ('CLEAN', 'build_native_neck_query_bank', 'native_neck_query_bank', 'COMPLETE_TRAIN_ONLY_NATIVE_QUERY_BANK'),
            ('MISMATCH', 'build_native_neck_mismatch_bank', 'native_neck_mismatch_query_bank', 'COMPLETE_TRAIN_ONLY_MISMATCH_QUERY_BANK')]:
            tasks = [dict(id=seq, growth_bytes=512*2**20,
                command=[sys.executable, '-u', 'tools/'+worker+'.py', '--sequence', seq]) for seq in protocol['train']]
            queue.jobs('COMPILE_'+kind+'_TRAIN_BANKS', tasks, False,
                lambda row, family=family, status=status: complete(work / family / row['id'] / 'complete.json', status))
        for split in ['dev']:
            encode_evaluation(queue, work, protocol, split)
        tasks = [dict(id='gpu'+physical, physical_gpu=physical,
            command=[sys.executable, '-u', 'tools/probe_neck72_gpu.py']) for physical in ['4', '5']]
        queue.jobs('ACTUAL_BOTH_GPU_ENGINEERING', tasks, True,
            lambda row: complete(work / 'neck72_actual_GPU_engineering' / row['id'] / 'complete.json',
                                 'COMPLETE_NECK72_GPU_INPUT_OPTIMIZER_AND_EXACT_ZERO_FALLBACK'))
        curves = {h['id']+'_'+a: [] for h in study['heads'] for a in ALGORITHMS}
        epoch = 18
        while True:
            tasks = []
            for head in study['heads']:
                command = [sys.executable, '-u', head['worker'], '--frontend', head['frontend'],
                    '--cue', head['cue'], '--seed', '0', '--epochs', str(epoch)]
                if epoch > 18:
                    command.append('--resume')
                tasks.append(dict(id=head['id'], physical_gpu='4' if head['cue'] == 'rgb' else '5',
                                  command=command, output=head['output'], growth_bytes=4*2**20))
            state['formal_fit_started'] = True
            def check_fit(row):
                value = complete(Path(row['output']) / 'complete.json', 'COMPLETE_NATIVE_ROBUST_QUERY_FIT_STAGE')
                assert value['epochs'] == epoch and value['seed'] == 0
                assert [v['epoch'] for v in value['history']] == list(range(1, epoch+1))
                assert value['fit_sequences'] == protocol['train'] and not value['Dev_fit_rows'] and not value['Test_fit_rows']
            queue.jobs('FIT_ALL8_TO_'+str(epoch), tasks, True, check_fit)
            new_epochs = [6, 12, 18] if epoch == 18 else [epoch]
            cases = [config_for(study, h, e, a) for h in study['heads'] for e in new_epochs for a in ALGORITHMS]
            evaluate(queue, study, cases, 'dev', 'NORMAL_DEV_THROUGH_'+str(epoch))
            for case in cases:
                p = work / 'neck72_dev' / case['id'] / 'metrics.json'
                curves[case['head']+'_'+case['algorithm']].append(dict(epoch=case['epoch'],
                    HOTA=read(p)['trackeval']['HOTA'], metrics_path=str(p), metrics_sha256=sha256(p)))
            decision = common_budget_decision(curves, epoch)
            dump(work / ('neck72_budget_e'+str(epoch)+'.json'), decision)
            if not decision['extend_all8_heads']:
                break
            epoch = decision['next_epochs']
        selected = selected_cases(study, curves)
        new_cases = [c for c in selected['cases'] if not (work / 'neck72_dev' / c['id'] / 'metrics.json').exists()]
        evaluate(queue, study, new_cases, 'dev', 'ALL_SELECTED_CONTROLS_AND_REMOVALS_DEV')
        check_zero_pairs(work, study, selected, 'dev')
        selected.update(status='FROZEN_NECK72_TEST_AFTER_ALL_COMPLETE_SELECTED_DEV', study_sha256=sha256(STUDY),
            common_fit_epochs=epoch, curves=curves, complete_selected_Dev_before_Test=True,
            Dev_metrics={c['id']: sha256(work / 'neck72_dev' / c['id'] / 'metrics.json') for c in selected['cases']})
        dump(work / 'neck72_Test_selection_frozen.json', selected)
        encode_evaluation(queue, work, protocol, 'test')
        evaluate(queue, study, selected['cases'], 'test', 'ALL_FROZEN_FULL7_TEST')
        check_zero_pairs(work, study, selected, 'test')
        state['status'] = 'COMPLETE_NECK72_ALL_SELECTED_TEST_PENDING_FULL_SYSTEM_COST_AUDIT'; queue.save()
        subprocess.run([sys.executable, 'tools/export_neck72_study.py'], cwd=ROOT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_NECK72_STUDY', error=repr(error)); queue.save()
        # Preserve any already-running sibling and its recordedPID. Never
        # turn a live process into a replacement/restart merely on timeout.
        raise


def encode_evaluation(queue, work, protocol, split):
    import numpy as np
    plan, _, _ = context()
    sequences = protocol['dev' if split == 'dev' else 'historical_test']
    counts = {}
    for sequence in sequences:
        counts[sequence] = 0
        for arm in ['new_diffv2ir', 'rgb']:
            path = Path(plan['frozen_detection_sources'][arm+'_'+split]) / 'detections' / (sequence+'.npz')
            with np.load(path) as arrays:
                counts[sequence] += int(arrays['offsets'][-1])
    tasks = [dict(id=seq+'_'+cue, sequence=seq, cue=cue,
        physical_gpu='4' if cue == 'rgb' else '5', growth_bytes=counts[seq]*512*4*(2 if cue == 'rgb' else 1)+2**20,
        command=[sys.executable, '-u', 'tools/extract_native_neck_eval_features.py', '--split', split,
                 '--sequence', seq, '--cue', cue, '--seed-only', '0',
                 '--output-family', 'native_neck_seed0_eval_features']) for seq in sequences for cue in ['rgb', 'ir']]
    def check(row):
        folder = work / 'native_neck_seed0_eval_features' / split / row['sequence'] / row['cue'] / 'seed_0'
        meta = complete(folder / 'complete.json', 'COMPLETE_NATIVE_CURRENT_FEATURES')
        assert meta['selected_encoder_seeds'] == [0]
        if split == 'dev':
            import numpy as np
            for arm in ['rgb', 'new']:
                old = work / 'native_neck_eval_features/dev' / row['sequence'] / row['cue'] / 'seed_0' / (arm+'.npz')
                with np.load(old) as a, np.load(folder / (arm+'.npz')) as b:
                    assert set(a.files) == set(b.files)
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
    queue.jobs('ENCODE_COMPLETE_'+split.upper(), tasks, True, check)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'run'])
    globals()[parser.parse_args().action]()
