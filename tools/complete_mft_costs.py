#!/usr/bin/env python3
"""Complete six actual B4cost runs between the legacy and native study queues."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.mft_cost_supplement import CASES, SPEC, NATIVE_COMMAND, WaitingControllerLease, context, registered, references, route, source_hashes
from tools.native_pyramid_experiment import frozen_json, predecessor_ready, read
from tools.neck72_runtime import require_storage
from tools.neck72_runtime_study import exited
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.task_utility_queue import Queue, process_identity


def register(unit_path):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Registration is CPU only')
    plan, work, base = context()
    unit = read(unit_path)
    sources = source_hashes()
    if (unit['exit_code'] != 0 or not unit['all_sources_unchanged'] or
            any(sources[p] != s for p, s in unit['source_hashes'].items()) or
            sha256(unit['log']) != unit['log_sha256']):
        raise ValueError('Current-source CPU input/lifecycle tests must pass')
    nativepath = work / 'native_pyramid_study_state.json'
    native = read(nativepath)
    identity = process_identity(native['controller_pid'], NATIVE_COMMAND)
    WaitingControllerLease(identity, nativepath, NATIVE_COMMAND).idle()
    artifacts = {str(Path(unit_path).resolve()): sha256(unit_path), unit['log']: sha256(unit['log'])}
    for p in [ROOT / 'configs/perceptual_structural_fulltrain_test_20261123.json',
            ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json',
            work / 'correlation_cache/calibration.json']:
        artifacts[str(p)] = sha256(p)
    for rel in ['PERCEPTUAL_MFT_20261125_FULL_TRAIN_TEST47_AUDIT.json',
            'PERCEPTUAL_MFT_20261125_FULL_TRAIN_DEV47_AUDIT.json',
            'PERCEPTUAL_MFT_20261123_FULL_TRAIN_SINGLE_METHOD_RUNTIME.json',
            'PERCEPTUAL_MFT_20261123_FULL_TRAIN_SINGLE_METHOD_RUNTIME_MEMORY_COMPONENTS.json']:
        p = ROOT / 'docs/fusion_research' / rel
        artifacts[str(p)] = sha256(p)
    audit = read(ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261125_FULL_TRAIN_TEST47_AUDIT.json')
    if audit['complete_cases'] != 13 or audit['Test_primary_reselection']:
        raise ValueError('Complete original frozen accuracy audit required')
    spec = read(ROOT / 'configs/perceptual_structural_fulltrain_test_20261123.json')
    for identifier in CASES:
        configpath, case, model = route(identifier)
        artifacts[str(configpath)] = sha256(configpath)
        for p in references(plan, case).values():
            artifacts[str(p)] = sha256(p)
        if model.get('checkpoint'):
            p = Path(model['checkpoint'])
            artifacts[str(p)] = sha256(p)
    require_storage(work, growth_bytes=4 * 2**20)
    value = dict(status='REGISTERED_MISSING_THREE_ORIGINAL_B4_COSTS_PENDING_ACTUAL_GPU',
        specification_sha256=sha256(SPEC), source_hashes=sources, artifact_hashes=artifacts,
        native_waiter_identity=identity, original_Test_specification_sha256=sha256(ROOT / 'configs/perceptual_structural_fulltrain_test_20261123.json'),
        cases=list(CASES), modes=['live', 'memory'], actual_GPU_measurements=0,
        fixed_seed=0, no_new_fits=True, no_accuracy_recipe_change=True, goal_met=False, supported_innovations=0)
    frozen_json(base / 'registered.json', value)
    registered()
    return value


def validate_result(base, identifier, memory, entry):
    mode = 'memory' if memory else 'live'
    path = base / 'results' / (identifier + '_' + mode + '.json')
    value = read(path)
    configpath, _, recipe = route(identifier)
    if (value['status'] != 'COMPLETE_ACTUAL_B4_FROZEN_FULL_TRAIN_CONTROL_COST' or value['id'] != identifier or
            value['model_id'] != recipe['id'] or value['config_sha256'] != sha256(configpath) or
            value['registration_sha256'] != sha256(base / 'registered.json') or value['memory_inputs'] is not memory or
            value['specification_sha256'] != sha256(SPEC) or value['source_hashes'] != source_hashes() or
            value['timed_frames'] != 48 or value['NN_batch_size'] != 4 or value['timed_batches'] != 12 or
            value['physical_gpu'] != '4' or value['peak_own_allocated_GPU_GiB'] <= 0 or
            value['actual_device'] != 'cuda:0' or value['raw_precision'] != 'torch.float16' or
            value['peak_process_RSS_GiB'] > 8 or value['GT_read'] or value['Test_accuracy_read'] or
            value['new_fit'] or not value['source_prefix_and_independent_MOT_exact'] or
            value['decoded_RAM_cache_bytes'] > 2 * 2**30):
        raise ValueError('Actual original-source B4 GPU cost proof required')
    if (value['own_process_identity'] != {k: entry[k] for k in ['pid', 'start_ticks', 'command_sha256']} or
            not math.isfinite(value['FPS']) or value['FPS'] <= 0 or
            value['GPU_uuid'].lower().replace('gpu-', '') != read(ROOT / 'configs/perceptual_gpu_handoff_20261123.json')['future_gpu_uuids']['4'].lower().replace('gpu-', '')):
        raise ValueError('Original actual GPU worker identity,UUIDand finite cost required')
    if [r['sequence'] for r in value['per_sequence']] != ['BT-005', 'SN-013'] or any(
            r['timed_frames'] != 24 or r['warmup_frames'] != 8 or not r['source32_arrays_exact'] or not r['independent_prefix_MOT_exact']
            for r in value['per_sequence']):
        raise ValueError('All64 actual NNframes and independent source-MOT comparisons required')
    if not math.isclose(value['FPS'], 48/sum(r['seconds'] for r in value['per_sequence']), rel_tol=1e-12):
        raise ValueError('Actual raw timing samples and aggregate differ')
    if any(sha256(p) != s for p, s in value['image_hashes'].items()):
        raise ValueError('Original cost prefix image bytes changed')
    return value


def export():
    _, work, base, registration = registered()
    state = read(base / 'state.json')
    if not state.get('native_waiter_resumed') or state.get('native_waiter_deferred'):
        raise ValueError('Resume the original native waiter before exporting')
    records = []
    for memory in [False, True]:
        for identifier in CASES:
            mode = 'memory' if memory else 'live'
            phase = 'COST_' + identifier + '_' + mode
            entry = state['phases'][phase][identifier]
            if entry['status'] != 'COMPLETE' or entry['exit_code'] != 0 or not entry['argv_identity_observed']:
                raise ValueError('All original actual child executions must exit0')
            if not exited(entry['pid'], command=entry['command'], identity=entry):
                raise ValueError('Every measured worker must actually exit')
            records.append(validate_result(base, identifier, memory, entry))
    result = dict(status='COMPLETE_THREE_MISSING_ORIGINAL_COSTS_BOTH_INPUT_MODES',
        registration_sha256=sha256(base / 'registered.json'), specification_sha256=sha256(SPEC), records=records,
        actual_independent_GPU_runs=6, original_independently_measured_cases=10, now_independently_measured_cases=13,
        native_waiter_identity=registration['native_waiter_identity'], native_waiter_resumed=True,
        original_accuracy_files_unchanged=True, new_fit=False, new_accuracy=False, goal_met=False, supported_innovations=0,
        scope='Six actual independent B4runs close the three explicitly missing costs. Original ten B4cost records remain unchanged. '
            'TwoDevprefixes48timedframes,notB1latencyorfullTestFPS. Inputhashes/prefixrowcapture overhead disclosed. '
            'Native CPUwaiter deferred before it had children;resumed automatically before export,including on failure.')
    path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261125_FULL_TRAIN_MISSING_COSTS.json'
    frozen_json(path, result)
    lines = ['# Three missing original full-Train controls: actual B4costs', '', result['scope'], '',
        '| Case | Memory input | B4 FPS | NN/NMS FPS |', '| --- | --- | ---: | ---: |']
    for row in records:
        lines.append(f"| {row['id']} | {row['memory_inputs']} | {row['FPS']:.6f} | {row['NN_NMS_policy_transfer_FPS']:.6f} |")
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    return path


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('The owning scheduler must remain CPU only')
    plan, work, base, registration = registered()
    lock = (base / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (base / 'state.json').exists():
        raise ValueError('Preserve prior state;inspect actual owner/children before any recovery')
    identity = process_identity(os.getpid())
    lease = WaitingControllerLease(registration['native_waiter_identity'], work / 'native_pyramid_study_state.json', NATIVE_COMMAND)
    state = dict(status='ACQUIRING_IDLE_NATIVE_WAITER_SCHEDULING_LEASE', controller_pid=os.getpid(),
        controller_identity=identity, phases={}, registration_sha256=sha256(base / 'registered.json'),
        native_waiter_deferred=False, native_waiter_resumed=False, fixed_seed=0, physical_gpus=['4', '5'],
        actual_profile_gpu='4', goal_met=False, supported_innovations=0)
    def guard():
        if registered()[3] != registration or not lease.paused():
            raise ValueError('Original source-bound idle waiter scheduling lease changed')
    queue = Queue(work, base, state, plan['data_root'], guard)
    queue.save()
    def stop(signum, frame):
        raise KeyboardInterrupt('Own cost scheduler termination;resume original waiter')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        lease.acquire()
        state.update(status='WAITING_FOR_ACTUAL_COMPLETE_LEGACY_EXPORT_AND_EXIT', native_waiter_deferred=True)
        queue.save()
        while not predecessor_ready(work):
            guard()
            time.sleep(10)
        for memory in [False, True]:
            for identifier in CASES:
                mode = 'memory' if memory else 'live'
                command = [sys.executable, '-u', 'tools/benchmark_mft_missing_cost.py', '--case', identifier]
                if memory:
                    command.append('--memory-inputs')
                queue.jobs('COST_' + identifier + '_' + mode,
                    [dict(id=identifier, command=command, physical_gpu='4')], True,
                    lambda entry, name=identifier, mem=memory: validate_result(base, name, mem, entry))
        state['status'] = 'COSTS_MEASURED_PENDING_NATIVE_RESUME_AND_EXPORT'
    except BaseException as error:
        state.update(status='FAILED_PRESERVE_MISSING_COSTS', error=repr(error))
        raise
    finally:
        # Never leave the old native queue waiting on a cost failure.
        # On an owner signal, stop only its own cost child and verify exit
        # before allowing the native queue to acquire either GPU.
        for slot, (name, process) in list(queue.active.items()):
            if process.poll() is None:
                process.terminate()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                while True:
                    try:
                        code = process.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        state['status'] = 'WAITING_FOR_OWN_COST_EXIT_BEFORE_NATIVE_RELEASE'
                        queue.save()
            for jobs in state['phases'].values():
                if name in jobs and jobs[name].get('pid') == process.pid:
                    jobs[name].update(status='INTERRUPTED_OWN_COST_WORKER', exit_code=code)
            del queue.active[slot]
        resumed = lease.release()
        state['native_waiter_deferred'] = False
        state['native_waiter_resumed'] = resumed
        queue.save()
    path = export()
    state.update(status='COMPLETE_MISSING_COSTS_AND_NATIVE_WAITER_RESUMED', report=str(path), report_sha256=sha256(path))
    queue.save()


def recover_release():
    """Recovery after a dead scheduler only;never bypass live active GPU workers."""
    _, work, base, registration = registered()
    state = read(base / 'state.json')
    if not exited(state['controller_pid'], identity=state['controller_identity']):
        raise ValueError('The owning scheduler is still live')
    for jobs in state['phases'].values():
        for job in jobs.values():
            if job.get('pid') and not exited(job['pid'], command=job['command'], identity=job):
                raise ValueError('Preserve live cost child;no release until it actually exits')
    lease = WaitingControllerLease(registration['native_waiter_identity'], work / 'native_pyramid_study_state.json', NATIVE_COMMAND)
    lease.idle()
    fd = os.pidfd_open(registration['native_waiter_identity']['pid'])
    try:
        lease.idle()
        signal.pidfd_send_signal(fd, signal.SIGCONT)
    finally:
        os.close(fd)
    state.update(status='RECOVERY_RELEASED_NATIVE_AFTER_DEAD_COST_SCHEDULER', native_waiter_deferred=False, native_waiter_resumed=True)
    dump(base / 'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run', 'export', 'release'])
    parser.add_argument('--pytest-receipt', type=Path)
    args = parser.parse_args()
    if args.action == 'register':
        register(args.pytest_receipt)
    elif args.action == 'run':
        run()
    elif args.action == 'export':
        export()
    else:
        recover_release()
