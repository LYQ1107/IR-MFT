import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from tools.mft_cost_supplement import Images, WaitingControllerLease, WAITING, route
from tools.task_utility_queue import process_identity


def image_fixture(intervention=None, memory=False, needs_ir=True):
    rgb = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    ir = np.arange(8 * 12, dtype=np.uint8).reshape(8, 12)
    calls = []
    def load(path, flags):
        calls.append(path)
        return (rgb if path.endswith('.jpg') else ir).copy()
    images = Images(dict(data_root='/fixture_rgb', generated_ir_root='/fixture_IR'),
        intervention, memory, needs_ir, loader=load)
    if memory:
        images.preload()
    return images, rgb, ir, calls


@pytest.mark.parametrize('memory', [False, True])
def test_image_roll_matches_actual_original_pixel_operation_and_preserves_source(memory):
    images, rgb, ir, _ = image_fixture('IR_image_spatial_roll', memory)
    actual = images['BT-005', 1, 'ir']
    np.testing.assert_array_equal(actual, np.roll(ir, (2, 3), (0, 1)))
    np.testing.assert_array_equal(images['BT-005', 1, 'rgb'], rgb)
    if memory:
        np.testing.assert_array_equal(images.cache['BT-005', 1, 'ir'], ir)


def test_canonical_preload_reads_only_RGB_and_rejects_IR():
    images, _, _, calls = image_fixture(memory=True, needs_ir=False)
    assert len(calls) == 64 and all(p.endswith('.jpg') for p in calls)
    with pytest.raises(ValueError, match='never read IR'):
        images.raw(('BT-005', 1, 'ir'))


def test_original_intervention_routes_and_unknown_route_rejected():
    assert route('FTT_source_RGB_canonical')[1]['baseline']
    assert route('FTT_IR_image_roll')[1]['intervention'] == 'IR_image_spatial_roll'
    assert route('FTT_IR_unavailable')[1]['intervention'] == 'unavailable'
    with pytest.raises(ValueError):
        route('FTT_primary_IR')


def test_roll_is_before_original_auxiliary_preprocessing_and_descriptors(monkeypatch):
    from tools.benchmark_structural_factorial import inputs
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    actual, rgb, ir, _ = image_fixture('IR_image_spatial_roll')
    manual, _, _, _ = image_fixture()
    manual.loader = lambda path, flags: rgb.copy() if path.endswith('.jpg') else np.roll(ir, (2, 3), (0, 1))
    config = dict(inference_input_hw=[32, 64])
    calibration = dict(statistics=dict(ir=dict(mean=.5, std=.25)))
    a = inputs(actual.plan, config, calibration, 'BT-005', [1, 2, 3, 4], 'ir', actual)
    b = inputs(manual.plan, config, calibration, 'BT-005', [1, 2, 3, 4], 'ir', manual)
    for group in a[0]:
        for key in a[0][group]:
            assert torch.equal(a[0][group][key], b[0][group][key])
    assert a[1:] == b[1:]


@pytest.mark.parametrize('identifier', ['FTT_IR_unavailable', 'FTT_source_RGB_canonical'])
def test_measurement_checks_actual_packet_and_independent_MOT_prefixes(tmp_path, monkeypatch, identifier):
    import tools.benchmark_mft_missing_cost as worker
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    images, _, _, _ = image_fixture(needs_ir=identifier != 'FTT_source_RGB_canonical')
    _, case, _ = route(identifier)
    values = np.array([[10., 10., 40., 40., .9], [50., 50., 80., 80., .8]], np.float32)
    reference = tmp_path / 'synthetic_packets_NOT_GPU_PROOF.npz'
    np.savez(reference, frame_ids=np.arange(1, 33), offsets=np.arange(0, 65, 2),
        boxes=np.tile(values[:, :4], (32, 1)), scores=np.tile(values[:, 4], 32))
    def predict(model, adapter, arrays, ratios, available, cue, policy):
        assert cue == ('baseline' if case.get('baseline') else 'ir')
        assert available == ([True] * 4 if case.get('baseline') else [False] * 4)
        return [values.copy() for _ in ratios]
    monkeypatch.setattr(worker, 'predict', predict)
    result = worker.measure(None, None, images.plan, dict(inference_input_hw=[32, 64]),
        dict(statistics=dict(ir=dict(mean=.5, std=.25))), case, images,
        {'BT-005': reference, 'SN-013': reference})
    assert result['timed_frames'] == 48 and result['timed_batches'] == 12
    assert all(r['source32_arrays_exact'] and r['independent_prefix_MOT_exact'] for r in result['per_sequence'])
    assert images.accesses == dict(rgb=64, ir=0 if case.get('baseline') else 64)


@pytest.fixture
def cpu_waiter(tmp_path):
    command = [sys.executable, '-u', '-c', 'import time;time.sleep(30)']
    process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), start_new_session=True)
    identity = process_identity(process.pid, command)
    statepath = tmp_path / 'fixture_state_NOT_FORMAL.json'
    statepath.write_text(json.dumps(dict(status=WAITING, controller_pid=process.pid, phases={})))
    lease = WaitingControllerLease(identity, statepath, command, fixture=True)
    try:
        yield process, lease, statepath
    finally:
        lease.release()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=3)


def test_actual_CPU_waiter_pidfd_stop_and_resume(cpu_waiter):
    process, lease, _ = cpu_waiter
    lease.acquire()
    assert lease.paused() and process.poll() is None
    assert lease.release()
    assert not lease.stopped


def test_never_stop_waiter_after_it_has_entered_work(cpu_waiter):
    process, lease, statepath = cpu_waiter
    state = json.loads(statepath.read_text())
    state['phases'] = {'STARTED_WORK': {}}
    statepath.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='idle pre-GPU'):
        lease.acquire()
    assert not lease.stopped and process.poll() is None


def test_actual_waiter_is_resumed_on_scheduler_worker_failure(cpu_waiter, tmp_path, monkeypatch):
    import tools.complete_mft_costs as scheduler
    process, lease, _ = cpu_waiter
    base = tmp_path / 'artificial_cost_scheduler'
    base.mkdir()
    registration = dict(native_waiter_identity=lease.identity)
    (base / 'registered.json').write_text(json.dumps(registration))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(scheduler, 'registered', lambda: ({'data_root': '/unused'}, tmp_path, base, registration))
    monkeypatch.setattr(scheduler, 'WaitingControllerLease', lambda *a: lease)
    monkeypatch.setattr(scheduler, 'predecessor_ready', lambda *a: True)
    monkeypatch.setattr(scheduler.signal, 'signal', lambda *a: None)
    class FailingQueue:
        def __init__(self, work, output, state, *args):
            self.state, self.active = state, {}
        def save(self):
            (base / 'state.json').write_text(json.dumps(self.state))
        def jobs(self, *args):
            assert lease.paused()
            raise RuntimeError('ARTIFICIAL_WORKER_FAILURE_NOT_GPU')
    monkeypatch.setattr(scheduler, 'Queue', FailingQueue)
    with pytest.raises(RuntimeError, match='ARTIFICIAL_WORKER_FAILURE'):
        scheduler.run()
    state = json.loads((base / 'state.json').read_text())
    assert state['native_waiter_resumed'] and not state['native_waiter_deferred']
    assert state['status'] == 'FAILED_PRESERVE_MISSING_COSTS'
    assert process.poll() is None


def test_owner_interrupt_drains_own_actual_CPU_child_before_native_resume(cpu_waiter, tmp_path, monkeypatch):
    import tools.complete_mft_costs as scheduler
    process, lease, _ = cpu_waiter
    base = tmp_path / 'artificial_interrupted_cost_scheduler'
    base.mkdir()
    registration = dict(native_waiter_identity=lease.identity)
    (base / 'registered.json').write_text(json.dumps(registration))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(scheduler, 'registered', lambda: ({'data_root': '/unused'}, tmp_path, base, registration))
    monkeypatch.setattr(scheduler, 'WaitingControllerLease', lambda *a: lease)
    monkeypatch.setattr(scheduler, 'predecessor_ready', lambda *a: True)
    monkeypatch.setattr(scheduler.signal, 'signal', lambda *a: None)
    children = []
    original_release = lease.release
    def release():
        assert children and all(p.poll() is not None for p in children)
        return original_release()
    monkeypatch.setattr(lease, 'release', release)
    class InterruptingQueue:
        def __init__(self, work, output, state, *args):
            self.state, self.active = state, {}
        def save(self):
            (base / 'state.json').write_text(json.dumps(self.state))
        def jobs(self, phase, tasks, *args):
            command = [sys.executable, '-u', '-c', 'import time;time.sleep(20)']
            child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), start_new_session=True)
            children.append(child)
            identity = process_identity(child.pid, command)
            self.state['phases'][phase] = {tasks[0]['id']: dict(command=command, **identity)}
            self.active['4'] = tasks[0]['id'], child
            raise KeyboardInterrupt('ARTIFICIAL_OWNER_SIGNAL_NOT_GPU')
    monkeypatch.setattr(scheduler, 'Queue', InterruptingQueue)
    with pytest.raises(KeyboardInterrupt, match='ARTIFICIAL_OWNER_SIGNAL'):
        scheduler.run()
    assert all(p.poll() is not None for p in children)
    state = json.loads((base / 'state.json').read_text())
    assert state['native_waiter_resumed'] and process.poll() is None
