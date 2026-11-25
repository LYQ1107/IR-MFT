"""Sources, frozen routes and reversible scheduling for three missing B4 costs."""
import os
from pathlib import Path
import signal
import time

import cv2
import numpy as np

from tools.native_pyramid_experiment import read
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_pipeline import source_files as original_sources
from tools.task_utility_queue import process_identity

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs/perceptual_mft_cost_supplement_20261125.json'
NATIVE_COMMAND = ['/data3/liuyeqiang/.venvs/trackocd-a100/bin/python', '-u',
    'tools/launch_native_pyramid_study.py', 'run']
CASES = ('FTT_IR_image_roll', 'FTT_IR_unavailable', 'FTT_source_RGB_canonical')
WAITING = 'WAITING_FOR_EXISTING_STRUCTURAL_AND_LEGACY_GPU_WORK'


def context():
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    work = Path(plan['work_root'])
    return plan, work, work / 'missing_full_train_costs'


def sources():
    return sorted(set(original_sources()) | {
        'tools/mft_cost_supplement.py', 'tools/benchmark_mft_missing_cost.py',
        'tools/complete_mft_costs.py', 'tools/audit_mft_full_train_test47.py',
        'tests/test_mft_cost_supplement.py', str(SPEC.relative_to(ROOT))})


def source_hashes():
    return {p: sha256(ROOT / p) for p in sources()}


def registered():
    plan, work, base = context()
    value = read(base / 'registered.json')
    if (value['status'] != 'REGISTERED_MISSING_THREE_ORIGINAL_B4_COSTS_PENDING_ACTUAL_GPU' or
            value['cases'] != list(CASES) or value['modes'] != ['live', 'memory'] or
            value['source_hashes'] != source_hashes() or value['specification_sha256'] != sha256(SPEC) or
            value['original_Test_specification_sha256'] != sha256(ROOT / 'configs/perceptual_structural_fulltrain_test_20261123.json') or
            any(sha256(p) != s for p, s in value['artifact_hashes'].items())):
        raise ValueError('Original cost supplement sources or frozen evidence changed')
    return plan, work, base, value


def route(identifier):
    if identifier not in CASES:
        raise ValueError('Only the three explicitly missing original costs are permitted')
    configpath = ROOT / 'configs/perceptual_structural_fulltrain_test_cases_20261123' / (identifier + '.json')
    config = read(configpath)
    spec = read(ROOT / 'configs/perceptual_structural_fulltrain_test_20261123.json')
    model = next(m for m in spec['models'] if m['id'] == config['model_id'])
    canonical = identifier == 'FTT_source_RGB_canonical'
    if canonical != bool(config.get('baseline')) or (not canonical and config.get('policy') != 'raw'):
        raise ValueError('Original frozen raw routes required')
    return configpath, config, model


def references(plan, case):
    _, work, _ = context()
    if case.get('baseline'):
        return {s: Path(plan['frozen_detection_sources']['rgb_dev']) / 'detections' / (s + '.npz')
            for s in ['BT-005', 'SN-013']}
    return {s: work / 'full_train_observations/structural_relative_predictions/dev' / s /
        'prefix_check_32' / (case['proposal_key'] + '.npz') for s in ['BT-005', 'SN-013']}


def image_path(plan, sequence, frame, modality):
    if modality == 'rgb':
        return Path(plan['data_root']) / 'train' / sequence / 'visible' / f'{frame:06d}.jpg'
    if modality == 'ir':
        return Path(plan['generated_ir_root']) / 'dev' / sequence / 'infrared_diffv2ir' / f'{frame:06d}.png'
    raise ValueError('Actual RGB or fixed November IR only')


class Images:
    """Use the old preprocessor unchanged; roll the actual IR before it is called."""
    def __init__(self, plan, intervention, memory, needs_ir, loader=cv2.imread):
        if intervention not in [None, 'IR_image_spatial_roll', 'unavailable']:
            raise ValueError('Original fixed intervention only')
        self.plan, self.intervention, self.memory, self.needs_ir = plan, intervention, memory, needs_ir
        self.loader, self.cache, self.decoded_bytes = loader, {}, 0
        self.accesses = dict(rgb=0, ir=0)
    def raw(self, key):
        sequence, frame, modality = key
        if modality == 'ir' and not self.needs_ir:
            raise ValueError('Canonical RGB must never read IR')
        value = self.loader(str(image_path(self.plan, sequence, frame, modality)),
            cv2.IMREAD_COLOR if modality == 'rgb' else cv2.IMREAD_UNCHANGED)
        if value is None or value.dtype != np.uint8 or value.ndim != (3 if modality == 'rgb' else 2):
            raise ValueError('Required original image is missing or malformed')
        return value
    def preload(self):
        if not self.memory:
            raise ValueError('Only the separately declared memory-input mode preloads')
        for sequence in ['BT-005', 'SN-013']:
            for frame in range(1, 33):
                for modality in ['rgb', 'ir'] if self.needs_ir else ['rgb']:
                    key = sequence, frame, modality
                    value = self.raw(key)
                    self.decoded_bytes += value.nbytes
                    if self.decoded_bytes > 2 * 2**30:
                        raise RuntimeError('Decoded prefix image cache exceeded2GiB')
                    self.cache[key] = value
    def __getitem__(self, key):
        modality = key[2]
        value = self.cache[key] if self.memory else self.raw(key)
        self.accesses[modality] += 1
        if modality == 'ir' and self.intervention == 'IR_image_spatial_roll':
            return np.roll(value, (value.shape[0] // 4, value.shape[1] // 4), (0, 1))
        return value


class WaitingControllerLease:
    """Stop only a pinned idle waiter, then resume that same process on exit."""
    def __init__(self, identity, statepath, command, *, fixture=False):
        self.identity, self.statepath, self.command = identity, Path(statepath), command
        self.fixture, self.fd, self.stopped = fixture, None, False
    def idle(self):
        identity = process_identity(self.identity['pid'], self.command)
        if identity != self.identity:
            raise ValueError('Waiting controller identity changed')
        state = read(self.statepath)
        if state['status'] != WAITING or state.get('phases') or state['controller_pid'] != identity['pid']:
            raise ValueError('Only the original idle pre-GPU waiter can be deferred')
        proc = Path('/proc') / str(identity['pid'])
        if (proc / 'task' / str(identity['pid']) / 'children').read_text().strip():
            raise ValueError('Never defer a controller with running children')
        environment = dict(v.split(b'=', 1) for v in (proc / 'environ').read_bytes().split(b'\0') if b'=' in v)
        if environment.get(b'CUDA_VISIBLE_DEVICES') != b'':
            raise ValueError('Only a CPU-only waiting controller may be deferred')
        if not self.fixture and self.command != NATIVE_COMMAND:
            raise ValueError('Only this study original native waiting controller is eligible')
        return proc
    def acquire(self):
        self.idle()
        self.fd = os.pidfd_open(self.identity['pid'])
        try:
            self.idle()
            # Mark ownership first: a Python exception after the signal must
            # still run release(). pidfd prevents signalling a reused PID.
            self.stopped = True
            signal.pidfd_send_signal(self.fd, signal.SIGSTOP)
            for _ in range(100):
                if self.paused():
                    return
                time.sleep(.01)
            raise RuntimeError('The original idle waiter did not enter stopped state')
        except BaseException:
            self.release()
            raise
    def paused(self):
        proc = self.idle()
        return self.stopped and (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] in ['T', 't']
    def release(self):
        sent = False
        try:
            if self.stopped and self.fd is not None:
                signal.pidfd_send_signal(self.fd, signal.SIGCONT)
                sent = True
        except ProcessLookupError:
            pass
        finally:
            self.stopped = False
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
        if sent:
            for _ in range(100):
                try:
                    proc = Path('/proc') / str(self.identity['pid'])
                    state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
                    if process_identity(self.identity['pid'], self.command) == self.identity and state not in ['T', 't', 'Z']:
                        return True
                except FileNotFoundError:
                    break
                time.sleep(.01)
        return False
