#!/usr/bin/env python3
"""Preregister and queue all frozen CGR coverage after existing GPU research."""
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
from tools.correlation_recovery_workflow import DOCS, SPEC, WORKFLOW, context, predecessor_ready, read, registered, require_probes
from tools.native_common_test_backfill import parity
from tools.run_pseudo_ir_comparison import dump, sha256

CASE_DIR = ROOT / 'configs/perceptual_correlation_recovery_workflow_cases_20261124'


def register():
    plan, spec, protocol, parent, calibration, work = context()
    assert not WORKFLOW.exists() and not CASE_DIR.exists()
    proof_names = ['PERCEPTUAL_MFT_20261124_CGR_NEW_STREAM_FULL_DEV_PROOF.json',
        'PERCEPTUAL_MFT_20261124_CGR_REAL_PIXELS_CPU_PROOF.json',
        'PERCEPTUAL_MFT_20261124_CGR_INFERENCE_CORE_CPU_TESTS.json',
        'PERCEPTUAL_MFT_20261124_CGR_WORKFLOW_CPU_TESTS.json']
    artifacts = {str(DOCS / n): sha256(DOCS / n) for n in proof_names}
    for name in proof_names:
        proof = read(DOCS / name)
        assert proof['status'].startswith('COMPLETE')
        assert all(sha256(ROOT / path) == h for path, h in proof['source_hashes'].items())
    files = [str(p.relative_to(ROOT)) for p in (ROOT / 'tools').glob('*correlation_recovery*.py')]
    files += ['tools/correlation_gate_observer.py', 'tools/native_pyramid_stream_tracks.py',
        'tools/native_pyramid_gate_observer.py', 'tools/native_common_test_backfill.py',
        'tools/native_pyramid_training_loop.py', 'tools/native_pyramid_experiment.py',
        'tools/structural_legacy_backfill.py', 'tools/recover_correlation_formal_test.py',
        'tools/eval_mft26ir.py', 'tools/export_perceptual_official_tables.py',
        'tools/export_perceptual_metrics47.py', 'tools/run_pseudo_ir_comparison.py',
        'tools/build_correlation_cache.py', 'yolox/models/local_correlation_repair.py',
        'yolox/tracker/byte_tracker.py', 'yolox/tracker/foveated_regions.py',
        'yolox/tracker/priority_observations.py', 'configs/perceptual_mft_20261122.json',
        'configs/cfp/protocol_v2.json', 'configs/perceptual_single_seed_policy_20261123.json',
        'configs/perceptual_correlation_recoverability_20261123.json',
        'configs/perceptual_correlation_repair_20261123.json',
        'configs/perceptual_gpu_handoff_20261123.json',
        'exps/yolox_mft26ir_visible.py', 'tools/launch_native_pyramid_study.py',
        'tools/export_native_pyramid.py', 'configs/perceptual_native_pyramid_study_20261124.json',
        'tests/test_correlation_recovery_inference.py', 'tests/test_correlation_recovery_workflow.py']
    for family in ['yolox/models', 'yolox/utils', 'yolox/exp']:
        files += [str(p.relative_to(ROOT)) for p in (ROOT / family).glob('*.py')]
    CASE_DIR.mkdir()
    case_files = []
    for case in spec['cases']:
        assert case.get('source', {}).get('seed', 0) == 0
        path = CASE_DIR / (case['id']+'.json')
        dump(path, case)
        case_files.append(dict(id=case['id'], path=str(path.relative_to(ROOT)), sha256=sha256(path)))
        files.append(str(path.relative_to(ROOT)))
    canonical = dict(dev=DOCS / 'perceptual_20261122_official_tables/correlation_recoverability_pilot_dev/CGR_original_RGB_ByteTrack.csv',
        test=DOCS / 'perceptual_20261122_official_tables/CGA_formal_test/CGA_source_RGB_canonical.csv')
    artifacts.update({str(p): sha256(p) for p in canonical.values()})
    artifacts[str(work / 'correlation_cache/calibration.json')] = sha256(work / 'correlation_cache/calibration.json')
    artifacts[str(work / 'native_pyramid_pipeline_registered.json')] = sha256(work / 'native_pyramid_pipeline_registered.json')
    toolkit = Path(plan['evaluation']['TrackEval_root'])
    for name in ['hota.py', 'clear.py', 'identity.py', 'count.py', '_base_metric.py']:
        p = toolkit / 'trackeval/metrics' / name
        artifacts[str(p)] = sha256(p)
    for split, sequences in [('dev', protocol['dev']), ('test', protocol['historical_test'])]:
        for seq in sequences:
            path = Path(plan['frozen_detection_sources']['rgb_'+split]) / 'detections' / (seq+'.npz')
            artifacts[str(path)] = sha256(path)
    value = dict(status='COMPLETE_CGR_WORKFLOW_REGISTERED_PENDING_ACTUAL_GPU_GATES',
        source_specification_sha256=sha256(SPEC), cases=spec['cases'], roles=spec['roles'], case_files=case_files,
        source_hashes={n: sha256(ROOT / n) for n in sorted(set(files))}, artifact_sources=artifacts,
        canonical_CSV={k: str(v) for k, v in canonical.items()},
        original_selections_unchanged=True, fixed_seed=0, additional_seed_work=False,
        physical_gpus=['4', '5'], maximum_cards=2,
        order=['Complete all existing fullTrain/legacy/native work and exports, actual workers exited.',
            'Both4/5: actual8frame prefixes on bothDevvideos, all8original outputs plus6zero branches and missing.',
            'GPU4: all12cases on all4834Dev, original pixels/predictions/tracks exact; official metrics all12.',
            'Freeze fullDev proof without changing any original model or readout selection.',
            'GPU5: uninstrumented single-condition B4 prefix image/ByteTrack cost for all12.',
            'GPU4/5: full7Test videos, every video all12cases; all source/off checks before official metrics.',
            'All12complete Test metrics, official39/persequence/276CSV, all47derived views and source/cost export.'],
        source_numeric_policy='Reuse already registered CGA bounds: outside final4frames RGB must be exact; '
            'finalbatch matched boxes<=2px, scores<=.005, all strict/nonstrict .6/.7 memberships exact, '
            'at most1unmatched observation with score<.6. Fresh/canonical remain different controls; '
            'no assertion of weak-NMS tracking equivalence. Original8selectedDev predictions remain exact.',
        IDs='Original counters continue across Dev videos to reproduce literal tracks. Formal Test resets '
            'IDs at1 per video; no oldCGR Test tracks exist. Tracking/evidence resets every video in both. '
            'Require canonical fullTest official39/CSV exact to existing RGB to validate the protocol.',
        runtime_scope='B4 prefix throughput,168measured frames per condition; original canonical proposal NPZ '
            'and DiffV2IR offline generation excluded. CurrentRGB encoding/head is still computed for all cases. '
            'Not fullTest FPS or B1 online latency.',
        storage='64GiB work cap/40GiB independent free reserve; no image/FPN copies, no weights in Git.',
        goal_met=False, supported_innovations=0)
    dump(WORKFLOW, value)
    path = work / 'correlation_recovery_workflow_registered.json'
    assert not path.exists()
    dump(path, dict(status=value['status'], workflow_sha256=sha256(WORKFLOW),
        source_files=len(value['source_hashes']), formal_cases=12, actual_GPU_engineering_pending=True,
        fixed_seed=0, goal_met=False, supported_innovations=0))
    print(json.dumps(dict(status=value['status'], source_files=len(value['source_hashes']), cases=12)), flush=True)


class Queue:
    def __init__(self, work, state):
        self.work, self.state, self.active = Path(work), state, {}
        self.path = self.work / 'correlation_recovery_workflow_state.json'

    def save(self):
        dump(self.path, self.state)

    def jobs(self, phase, tasks, gpu, check):
        assert tasks, 'A required phase cannot have an empty task set'
        self.state['status'] = phase
        entries = self.state['phases'].setdefault(phase, {})
        for task in tasks:
            assert task['id'] not in entries
            entries[task['id']] = dict(task, status='PENDING')
        self.save()
        try:
            while True:
                registered()
                for slot, (key, process) in list(self.active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    del self.active[slot]
                    row = entries[key]
                    row['exit_code'] = code
                    if code:
                        row['status'] = 'FAILED'
                        raise RuntimeError('Preserve CGR child failure: '+row['log'])
                    try:
                        if check(row) is False:
                            raise RuntimeError('Required CGR child evidence failed: '+key)
                    except Exception:
                        row['status'] = 'FAILED_EVIDENCE'
                        raise
                    row['status'] = 'COMPLETE'
                for slot in ['4', '5'] if gpu else ['CPU0', 'CPU1']:
                    if slot in self.active:
                        continue
                    pending = [(k, v) for k, v in entries.items() if v['status'] == 'PENDING'
                               and (not gpu or v.get('physical_gpu', slot) == slot)]
                    if not pending:
                        continue
                    key, row = pending[0]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot if gpu else '',
                        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                        MFT26IR_ROOT=self.state['data_root'])
                    log = self.work / 'correlation_recovery_logs' / (phase+'_'+key+'.log')
                    log.parent.mkdir(exist_ok=True)
                    with log.open('xb', buffering=0) as stream:
                        process = subprocess.Popen(row['command'], cwd=ROOT, env=env,
                            stdout=stream, stderr=subprocess.STDOUT)
                    row.update(status='RUNNING', pid=process.pid, log=str(log), slot=slot)
                    self.active[slot] = key, process
                self.save()
                if not self.active and all(row['status'] == 'COMPLETE' for row in entries.values()):
                    return
                time.sleep(20)
        except Exception:
            self.save()
            # Preserve any still-running sibling; do not kill healthy work.
            for key, process in self.active.values():
                self.state['preserved_active_children'][key] = process.pid
            self.save()
            raise


def check_production(work, split, sequences):
    for seq in sequences:
        p = Path(work) / ('correlation_recovery_'+split) / 'sequence_manifests' / (seq+'.json')
        value = read(p)
        assert value['status'] == 'COMPLETE_CGR_CURRENT_IMAGE_STREAMS_WITHOUT_GT'
        assert value['workflow_sha256'] == sha256(WORKFLOW)
        assert not value['GT_read'] and len(value['streams']) == 12


def check_metrics(work, split, key):
    value = read(Path(work) / ('correlation_recovery_'+split) / key / 'metrics.json')
    assert value['status'] == 'COMPLETE' and len(value['trackeval']) == 39
    assert value['manifest']['workflow_sha256'] == sha256(WORKFLOW)


def evaluate(queue, workflow, split):
    tasks = [dict(id=row['id'], command=[sys.executable, '-u', 'tools/evaluate_correlation_recovery.py',
        '--split', split, '--case', row['id']]) for row in workflow['case_files']]
    queue.jobs('OFFICIAL_'+split.upper(), tasks, False,
        lambda row: check_metrics(queue.work, split, row['id']))


def exact_pair(work, split, a, b, sequences):
    base = Path(work) / ('correlation_recovery_'+split)
    left, right = read(base / a / 'metrics.json'), read(base / b / 'metrics.json')
    assert left['trackeval'] == right['trackeval']
    assert sha256(base / a / 'trackeval_output/pedestrian_detailed.csv') == sha256(base / b / 'trackeval_output/pedestrian_detailed.csv')
    assert all(sha256(base / a / 'track_results' / (seq+'.txt')) ==
               sha256(base / b / 'track_results' / (seq+'.txt')) for seq in sequences)


def run():
    plan, spec, protocol, parent, calibration, work = context()
    workflow = registered()
    path = work / 'correlation_recovery_workflow_state.json'
    lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists()
    state = dict(status='WAITING_FOR_COMPLETE_NATIVE_AND_PRIOR_GPU_WORK', controller_pid=os.getpid(),
        workflow_sha256=sha256(WORKFLOW), data_root=plan['data_root'], phases={},
        preserved_active_children={}, physical_gpus=['4', '5'], fixed_seed=0,
        additional_seed_work=False, goal_met=False, supported_innovations=0)
    queue = Queue(work, state)
    queue.save()
    try:
        while not predecessor_ready(work):
            registered()
            time.sleep(30)
        queue.jobs('ACTUAL_GPU_ENGINEERING', [dict(id='gpu'+gpu, physical_gpu=gpu,
            command=[sys.executable, '-u', 'tools/run_correlation_recovery_gpu.py', 'probe'])
            for gpu in ['4', '5']], True, lambda row: None)
        require_probes(work)
        queue.jobs('FULL_DEV_PIXELS', [dict(id='all_Dev', physical_gpu='4',
            command=[sys.executable, '-u', 'tools/run_correlation_recovery_gpu.py', 'dev'])], True,
            lambda row: check_production(work, 'dev', protocol['dev']))
        evaluate(queue, workflow, 'dev')
        proofs = {}
        for case in spec['cases']:
            if case.get('source') and case['intervention'] == 'normal':
                current = work / 'correlation_recovery_dev' / case['id']
                proofs[case['id']] = parity(current, Path(case['source']['source_Dev_metrics']).parent, protocol['dev'])
        assert len(proofs) == 8
        exact_pair(work, 'dev', spec['roles']['unavailable'], spec['roles']['fresh_B4'], protocol['dev'])
        for split in ['dev']:
            csv = work / ('correlation_recovery_'+split) / spec['roles']['canonical_B4'] / 'trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv) == sha256(workflow['canonical_CSV'][split])
        proof_path = work / 'correlation_recovery_full_Dev_proof.json'
        assert not proof_path.exists()
        dump(proof_path, dict(status='COMPLETE_ALL12_CGR_DEV_WITH_ORIGINAL8_EXACT',
            workflow_sha256=sha256(WORKFLOW), original8_source_proofs=proofs,
            unavailable_all39_tracks_CSV_exact_to_fresh=True, canonical_CSV_exact=True,
            all_cases_complete=True, full_frames_each=4834, Test_read=False))
        freeze = work / 'correlation_recovery_Test_selection_frozen.json'
        assert not freeze.exists()
        dump(freeze, dict(status='FROZEN_CGR_TEST_FROM_ORIGINAL_SELECTION_AND_COMPLETE_DEV_PROOF',
            workflow_sha256=sha256(WORKFLOW), source_specification_sha256=sha256(SPEC),
            full_Dev_report_sha256=sha256(proof_path), complete_all13roles_12cases_Dev_before_Test=True,
            original_selection_unchanged=True, cases=spec['cases'], roles=spec['roles'],
            Test_accuracy_read_for_selection=False))
        queue.jobs('INDIVIDUAL_RUNTIME', [dict(id='runtime', physical_gpu='5',
            command=[sys.executable, '-u', 'tools/benchmark_correlation_recovery.py'])], True,
            lambda row: _check_runtime(work))
        queue.jobs('FULL_TEST_PIXELS', [dict(id=seq, sequence=seq,
            command=[sys.executable, '-u', 'tools/run_correlation_recovery_gpu.py', 'test', '--sequence', seq])
            for seq in protocol['historical_test']], True,
            lambda row: check_production(work, 'test', [row['sequence']]))
        evaluate(queue, workflow, 'test')
        exact_pair(work, 'test', spec['roles']['unavailable'], spec['roles']['fresh_B4'], protocol['historical_test'])
        canonical_csv = work / 'correlation_recovery_test' / spec['roles']['canonical_B4'] / 'trackeval_output/pedestrian_detailed.csv'
        assert sha256(canonical_csv) == sha256(workflow['canonical_CSV']['test'])
        state['status'] = 'COMPLETE_CGR_BACKFILL_REQUIRES_SCIENTIFIC_ASSESSMENT'
        queue.save()
        subprocess.run([sys.executable, 'tools/export_correlation_recovery.py'], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_CGR_WORKFLOW', error=repr(error))
        queue.save()
        raise


def _check_runtime(work):
    value = read(Path(work) / 'correlation_recovery_individual_runtime.json')
    assert value['status'] == 'COMPLETE_CGR_SINGLE_CONDITION_B4_PREFIX_RUNTIME'
    assert value['workflow_sha256'] == sha256(WORKFLOW) and len(value['rows']) == 12
    assert all(row['measured_frames'] == 168 for row in value['rows'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'run'])
    action = parser.parse_args().action
    register() if action == 'register' else run()
