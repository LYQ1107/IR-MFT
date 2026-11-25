#!/usr/bin/env python3
"""Resume the preserved native NAS I/O failure without changing frozen science."""
import argparse
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
BASE = WORK / 'native_pyramid_io_recovery_20261126_r1'
ORIGINAL = ROOT / 'tools/launch_native_pyramid_study.py'
ORIGINAL_SHA = '6117c69856a69ca7eb41f043cec082f2c95306edfee4f722e7287ca9a3234a5b'
CHECKPOINT_SHA = '4aa2c60d56438dd2ce6c8bf5372d1e79d87b09639c473f6c3aa57a6c92c83965'
FAILED_IMAGE = Path('/data3/liuyeqiang/nas/A100/IR-MFT/diffv2ir_final_20261104/full_generation/train/SN-015/infrared_diffv2ir/004953.png')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
    else:
        tmp = path.with_suffix('.partial.json')
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        os.replace(tmp, path)


def continuation():
    """Compile the original entire try/except, including all Dev/Test decisions."""
    if sha(ORIGINAL) != ORIGINAL_SHA:
        raise ValueError('The original controller changed')
    tree = ast.parse(ORIGINAL.read_text(), filename=str(ORIGINAL))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    blocks = [node for node in run.body if isinstance(node, ast.Try)]
    if len(blocks) != 1:
        raise ValueError('Unexpected original continuation structure')
    node = blocks[0]
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    code = compile(ast.Module(body=[node], type_ignores=[]), str(ORIGINAL), 'exec')
    return code, digest


def pending_tasks(phase, tasks, entries):
    """Reuse only actual exit0 completions; retry only the archived IR failure."""
    if set(entries) - {task['id'] for task in tasks}:
        raise ValueError('Existing phase contains an unrequested condition')
    pending, reused = [], []
    for task in tasks:
        previous = entries.get(task['id'])
        if previous is None:
            pending.append((task, False))
            continue
        if any(previous.get(key) != value for key, value in task.items()):
            raise ValueError('A preserved command, GPU or condition differs')
        if previous['status'] == 'COMPLETE' and previous.get('exit_code') == 0:
            reused.append(task['id'])
        elif (phase == 'NORMAL_TRAIN_06' and task['id'] == 'temporal_ir' and
              previous['status'] == 'FAILED' and previous.get('exit_code') == 1 and
              previous.get('pid') == 1248117):
            pending.append((task, True))
        else:
            raise ValueError('Do not duplicate running jobs or reinterpret another failure')
    return pending, reused


def engineering():
    command = ['python', 'fit', '--through-epoch', '6']
    tasks = [dict(id=name, command=command, physical_gpu='5') for name in ['temporal_rgb', 'temporal_ir']]
    entries = {task['id']: dict(task, status='COMPLETE', exit_code=0, pid=123) for task in tasks}
    saved = copy.deepcopy(entries)
    assert pending_tasks('NORMAL_TRAIN_06', tasks, entries) == ([], ['temporal_rgb', 'temporal_ir'])
    assert entries == saved
    entries['temporal_ir'].update(status='FAILED', exit_code=1, pid=1248117)
    pending, reused = pending_tasks('NORMAL_TRAIN_06', tasks, entries)
    assert pending == [(tasks[1], True)] and reused == ['temporal_rgb']
    checks = 3
    for phase, change in [('NORMAL_TRAIN_09', {}), ('NORMAL_TRAIN_06', {'status': 'RUNNING'}),
                          ('NORMAL_TRAIN_06', {'command': ['different']}),
                          ('NORMAL_TRAIN_06', {'physical_gpu': '4'}),
                          ('NORMAL_TRAIN_06', {'exit_code': 0}),
                          ('NORMAL_TRAIN_06', {'pid': 999})]:
        altered = copy.deepcopy(entries)
        altered['temporal_ir'].update(change)
        try:
            pending_tasks(phase, tasks, altered)
        except ValueError:
            checks += 1
        else:
            raise AssertionError('Unsafe reuse was accepted')
    assert pending_tasks('fresh_phase', tasks, {}) == ([(task, False) for task in tasks], [])
    _, digest = continuation()
    result = dict(status='COMPLETE_CPU_IO_RECOVERY_ENGINEERING', checks=checks + 2,
                  original_controller_sha256=ORIGINAL_SHA, original_continuation_AST_sha256=digest,
                  original_Dev_Test_body_changed=False, GPU_NN_or_fit_work=False)
    write(BASE / 'engineering.json', result, exclusive=True)
    print(json.dumps(result), flush=True)


def expected_image_digest():
    path = WORK / 'structural_full_train_fit/ir_111_s0/source_image_content_manifest.json'
    if sha(path) != 'a2d5040d06a972139d5e97ff13042553c6f0fc4a8130da53c1079553ea02461c':
        raise ValueError('Original fixed-image manifest differs')
    rows = [row for row in read(path) if row['sequence'] == 'SN-015' and row['frame'] == 4953 and row['modality'] == 'ir']
    if len(rows) != 1:
        raise ValueError('Failed actual input is not uniquely pinned')
    return rows[0]['sha256']


def prepare():
    import torch
    from tools.native_pyramid_training_loop import restore_delivered_prefix, validate_memory_at_prefix, state_sha256
    from tools.native_pyramid_train_data import OrderedTrainPixels
    archive = read(BASE / 'archive_manifest.json')
    for item in archive['items']:
        if sha(item['archive']) != item['sha256']:
            raise ValueError('An archived failure record differs')
    for pid in [3943138, 1248117, 1452564]:
        if (Path('/proc') / str(pid)).exists():
            raise ValueError('Reinspect the original process before recovery')
    latest = WORK / 'native_pyramid_fits/temporal_ir/latest.pt'
    if sha(latest) != CHECKPOINT_SHA or sha(archive['checkpoint_hard_link']) != CHECKPOINT_SHA:
        raise ValueError('Last saved optimizer state differs')
    value = torch.load(latest, map_location='cpu', weights_only=False)
    if (value['completed_epochs'] != 4 or value['epoch_clips'] != 5888 or
            value['actual_optimizer_updates'] != 33144 or value['delivered_clips_total'] != 33144 or
            value['prefix']['epoch'] != 5 or value['metadata']['seed'] != 0 or
            value['metadata']['physical_gpu'] != '5'):
        raise ValueError('Unexpected original fixed-seed saved prefix')
    tensors = list(value['module'].values()) + [tensor for row in value['optimizer']['state'].values()
              for tensor in row.values() if isinstance(tensor, torch.Tensor)]
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors if tensor.is_floating_point()):
        raise ValueError('Saved module or AdamW state is nonfinite')
    # No image reads or NN work: replay only immutable chronological addresses.
    provider = object.__new__(OrderedTrainPixels)
    study = read(ROOT / 'configs/perceptual_native_pyramid_study_20261124.json')
    provider.lengths = study['Train_delivery']['lengths']
    provider.epoch = None
    prefix = restore_delivered_prefix(provider, value['prefix'])
    validate_memory_at_prefix(value['temporal_state'], value['previous_RGB_gray'], provider.lengths, 5, 5888)
    _, digest = continuation()
    result = dict(status='REGISTERED_SOURCE_PINNED_NAS_IO_RECOVERY_WAITING_DATA',
        source_sha256=sha(Path(__file__)), original_controller_sha256=ORIGINAL_SHA,
        original_continuation_AST_sha256=digest, archived_failure_manifest_sha256=sha(BASE/'archive_manifest.json'),
        engineering_sha256=sha(BASE/'engineering.json'), checkpoint_sha256=CHECKPOINT_SHA,
        saved_module_sha256=state_sha256(value['module']), saved_prefix=value['prefix'],
        address_only_restore=prefix, saved_optimizer_updates=33144, unsaved_updates_to_replay=21,
        checkpoint_tensors_finite=True, saved_memory_sequence=value['temporal_state'].sequence,
        saved_memory_frame=value['temporal_state'].frame, failed_image=str(FAILED_IMAGE),
        failed_image_expected_sha256=expected_image_digest(), fixed_seed=0,
        GPU_NN_or_new_fit_work=False, original_Dev_Test_body_changed=False,
        original_five_six_epoch_fits_rerun=False, goal_met=False)
    if torch.cuda.is_initialized():
        raise ValueError('CPU recovery preparation initialized CUDA')
    write(BASE/'registered.json', result, exclusive=True)
    print(json.dumps(result), flush=True)


def run():
    registration = read(BASE/'registered.json')
    if sha(Path(__file__)) != registration['source_sha256']:
        raise ValueError('Recovery source changed after preparation')
    # Existing mount supervisor owns reconnection; do not remount shared storage.
    stable = 0
    while stable < 2:
        try:
            actual = sha(FAILED_IMAGE)
            if actual != registration['failed_image_expected_sha256']:
                raise ValueError('Recovered NAS image bytes differ from fixed November input')
            stable += 1
            row = dict(status='RECOVERED_FIXED_IMAGE_SHA_MATCHES', consecutive_reads=stable)
        except OSError as error:
            stable = 0
            row = dict(status='WAITING_ORIGINAL_NAS_SUPERVISOR_RECONNECTION', errno=error.errno,
                       error=repr(error), new_GPU_work=False)
        write(BASE/'recovery_status.json', dict(row, PID=os.getpid(), timestamp=time.time()))
        print(json.dumps(row), flush=True)
        if stable < 2:
            time.sleep(15)
    sys.path.insert(0, str(ROOT))
    import tools.launch_native_pyramid_study as original

    class RecoveryQueue(original.Queue):
        def jobs(self, phase, tasks, gpu, check):
            self.state['status'], self.state['phase'] = phase, phase
            entries = self.state.setdefault('phases', {}).setdefault(phase, {})
            pending, reused = pending_tasks(phase, tasks, entries)
            for key in reused:
                if check(entries[key]) is False:
                    raise ValueError('Preserved completion verification failed: '+key)
            for task, recovering in pending:
                previous = copy.deepcopy(entries.get(task['id']))
                entries[task['id']] = dict(task, status='PENDING', IO_recovery_retry=recovering)
                if recovering:
                    entries[task['id']]['preserved_failed_attempt'] = previous
            self.save()
            try:
                while True:
                    original.registry(self.work)
                    if sha(Path(__file__)) != registration['source_sha256']:
                        raise RuntimeError('Registered recovery controller changed')
                    for slot, (key, child) in list(self.active.items()):
                        code = child.poll()
                        if code is None:
                            continue
                        del self.active[slot]
                        entry = entries[key]
                        entry['exit_code'] = code
                        if code:
                            entry['status'] = 'FAILED'
                            raise RuntimeError('Preserve recovery child failure: '+entry['log'])
                        if check(entry) is False:
                            raise RuntimeError('Required original evidence failed: '+key)
                        entry['status'] = 'COMPLETE'
                    for slot in (['4', '5'] if gpu else ['CPU0', 'CPU1']):
                        if slot in self.active:
                            continue
                        todo = [(key, entry) for key, entry in entries.items() if entry['status']=='PENDING'
                                and (not gpu or entry.get('physical_gpu', slot)==slot)]
                        if not todo:
                            continue
                        key, entry = todo[0]
                        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=slot if gpu else '',
                            OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                            CUBLAS_WORKSPACE_CONFIG=':4096:8', MFT26IR_ROOT=self.state['data_root'])
                        suffix = '.recovery_io_r1.log' if entry['IO_recovery_retry'] else '.log'
                        log = self.work/'native_pyramid_logs'/(phase+'_'+key+suffix)
                        with log.open('xb', buffering=0) as handle:
                            child = subprocess.Popen(entry['command'], cwd=ROOT, env=environment,
                                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                        entry.update(status='RUNNING', pid=child.pid, log=str(log), slot=slot)
                        self.active[slot] = key, child
                    self.save()
                    if not self.active:
                        if not all(entry['status']=='COMPLETE' for entry in entries.values()):
                            raise ValueError('Unfinished original phase')
                        return
                    time.sleep(5)
            except Exception:
                for _, child in self.active.values():
                    if child.poll() is None:
                        child.terminate()
                self.save()
                raise

    with (WORK/'native_pyramid_study_state.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for pid in [3943138, 1248117]:
            if (Path('/proc')/str(pid)).exists():
                raise ValueError('Original process identity needs reinspection')
        current = WORK/'native_pyramid_study_state.json'
        original_state = BASE/'native_pyramid_study_state.json'
        if sha(current) != sha(original_state):
            raise ValueError('Current queue changed since failure preservation')
        plan, study, protocol, calibration, work, checkpoint = original.context()
        original.registry(work)
        state = read(original_state)
        if state['status'] != 'FAILED_PRESERVE_NATIVE_STUDY':
            raise ValueError('Only the preserved actual failure is recoverable')
        state.update(controller_pid=os.getpid(), status='RESUMING_PRESERVED_NAS_IO_FAILURE',
                     original_controller_pid=3943138, IO_recovery_registration_sha256=sha(BASE/'registered.json'))
        state.pop('error', None)
        queue = RecoveryQueue(work, state)
        code, digest = continuation()
        if digest != registration['original_continuation_AST_sha256']:
            raise ValueError('Original scientific continuation differs')
        queue.save()
        namespace = dict(vars(original))
        namespace.update(plan=plan, study=study, protocol=protocol, calibration=calibration,
                         work=work, checkpoint=checkpoint, state=state, queue=queue)
        exec(code, namespace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['engineering', 'prepare', 'run'])
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Recovery engineering/controller must be CPU only')
    sys.path.insert(0, str(ROOT))
    {'engineering': engineering, 'prepare': prepare, 'run': run}[args.mode]()
