"""Resource, provenance and unchanged-computation checks for dormant workers."""
import ast
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from tools import neck72_runtime
from tools.neck72_evidence import load_neck72_evidence
from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = 'e25f2f9896ab094611edc809f87d7a5ad7e981d3'


def test_complete_predecessor_needs_actual_exit_and_export():
    state = dict(status='COMPLETE_CGR_BACKFILL_REQUIRES_SCIENTIFIC_ASSESSMENT',
        workflow_sha256='a'*64, controller_pid=10,
        phases={'DEV': {'x': dict(status='COMPLETE', exit_code=0, pid=11)}})
    report = dict(status='COMPLETE_ALL6_TRAINED_CGR_FITS_FROZEN12_FULL7TEST_BACKFILL',
                  workflow_sha256='a'*64)
    assert neck72_runtime.predecessor_state_ready(state, report, lambda pid: True)
    assert not neck72_runtime.predecessor_state_ready(state, report, lambda pid: pid != 10)
    assert not neck72_runtime.predecessor_state_ready(state, report, lambda pid: pid != 11)
    assert not neck72_runtime.predecessor_state_ready(state, {}, lambda pid: True)
    broken = copy.deepcopy(state); broken['phases']['DEV']['x']['exit_code'] = 1
    assert not neck72_runtime.predecessor_state_ready(broken, report, lambda pid: True)
    broken['status'] = 'FAILED_PRESERVE_CGR_WORKFLOW'
    with pytest.raises(RuntimeError):
        neck72_runtime.predecessor_state_ready(broken, report, lambda pid: True)


def test_gpu_wait_and_seed_rejection_precede_cuda(monkeypatch, tmp_path):
    assert not torch.cuda.is_initialized()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    monkeypatch.setattr(neck72_runtime, 'gpu_predecessor_ready', lambda work: False)
    with pytest.raises(RuntimeError, match='Wait for complete'):
        neck72_runtime.require_gpu(tmp_path)
    for seed in [1, 2]:
        with pytest.raises(ValueError, match='only seed0'):
            neck72_runtime.require_gpu(tmp_path, seed)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    with pytest.raises(ValueError, match='physicalGPU4/5'):
        neck72_runtime.require_gpu(tmp_path)
    assert not torch.cuda.is_initialized()


def test_decoded_crop_memory_headroom_is_reserved_for_all_producers(monkeypatch):
    from tools import structural_train_image_reader
    monkeypatch.setattr(structural_train_image_reader, 'available_memory_bytes', lambda: 72*2**30)
    stats = neck72_runtime.require_host_memory(8*2**30)
    assert stats['maximum_workers'] == 4 and stats['reserve_bytes'] == 40*2**30
    monkeypatch.setattr(structural_train_image_reader, 'available_memory_bytes', lambda: 72*2**30-1)
    with pytest.raises(RuntimeError, match='retain40GiB'):
        neck72_runtime.require_host_memory(8*2**30)


def test_phase_admission_reserves_two_workers_and_child_only_its_incoming_memory(monkeypatch):
    from tools import structural_train_image_reader
    monkeypatch.setattr(structural_train_image_reader, 'available_memory_bytes', lambda: 56*2**30)
    parent = neck72_runtime.require_host_memory(8*2**30, maximum_workers=2)
    assert parent['maximum_workers'] == 2 and parent['reserve_bytes'] == 40*2**30
    # A peer has now allocated8GiB; reserving it again in the child would
    # reject a valid40GiB remainder. The child still checks its entire8GiB.
    monkeypatch.setattr(structural_train_image_reader, 'available_memory_bytes', lambda: 48*2**30)
    child = neck72_runtime.require_host_memory(8*2**30, maximum_workers=1)
    assert child['maximum_workers'] == 1 and child['reserve_bytes'] == parent['reserve_bytes']
    monkeypatch.setattr(structural_train_image_reader, 'available_memory_bytes', lambda: 48*2**30-1)
    with pytest.raises(RuntimeError, match='retain40GiB'):
        neck72_runtime.require_host_memory(8*2**30, maximum_workers=1)


def sample_bank(work, cue):
    folder = work / 'native_neck_eval_features/dev/BT-005' / cue / 'seed_0'
    folder.mkdir(parents=True)
    data = dict(frame_ids=np.array([1, 2]), offsets=np.array([0, 1, 2]),
                boxes=np.array([[2, 3, 12, 18], [4, 5, 14, 20]], np.float32))
    vector = np.random.default_rng(0).normal(size=(2, 512)).astype(np.float32)
    vector /= np.linalg.norm(vector, axis=1, keepdims=True)
    path = folder / 'new.npz'
    arrays = dict(data, native_s0=vector)
    if cue == 'rgb':
        arrays['reference_post'] = vector[:, ::-1].copy()
    np.savez_compressed(path, **arrays)
    dump(folder / 'complete.json', dict(status='COMPLETE_NATIVE_CURRENT_FEATURES', split='dev',
        sequence='BT-005', cue=cue, GT_read=False, future_frame_input=False, selected_encoder_seeds=[0],
        outputs={'new': dict(path=str(path), sha256=sha256(path))},
        checkpoints={'native_s0': dict(sha256='b'*64)}))
    return data, arrays, folder


def config(**kwargs):
    return dict(native_seed=0, native_cue='ir', native_feature_family='neck_metric72',
                detection_arm='new', **kwargs)


def test_missing_branch_only_needs_rgb_seed0_directory(tmp_path):
    data, arrays, _ = sample_bank(tmp_path, 'rgb')
    metadata = load_neck72_evidence(data, tmp_path, 'dev', 'BT-005',
                                   config(disable_native_increment=True))
    np.testing.assert_array_equal(data['native_reference'], arrays['reference_post'])
    assert 'native_auxiliary' not in data and 'ir' not in metadata
    assert not (tmp_path / 'native_neck_eval_features/dev/BT-005/ir').exists()


def test_active_neck72_requires_real_seed0_layout_and_exact_index(tmp_path):
    data, _, _ = sample_bank(tmp_path, 'rgb')
    _, ir, folder = sample_bank(tmp_path, 'ir')
    trained = tmp_path / 'native_neck_train_features/ir_s0/complete.json'
    trained.parent.mkdir(parents=True)
    dump(trained, dict(status='COMPLETE_TRAIN_ONLY_NATIVE_FEATURES', checkpoint_sha256='b'*64))
    alignment = tmp_path / 'native_neck_alignment'
    alignment.mkdir()
    rotation = np.eye(512, dtype=np.float32)[:, ::-1].copy()
    path = alignment / 'ir_s0.npz'; np.savez_compressed(path, rotation=rotation)
    dump(alignment / 'complete.json', dict(status='COMPLETE_TRAIN_ONLY_ORTHOGONAL_ALIGNMENT',
        Dev_rows=0, Test_rows=0, outputs={'ir_s0': dict(path=str(path), sha256=sha256(path))}))
    metadata = load_neck72_evidence(data, tmp_path, 'dev', 'BT-005', config())
    np.testing.assert_array_equal(data['native_auxiliary'], ir['native_s0'] @ rotation)
    assert metadata['alignment']['applied']
    load_neck72_evidence(data, tmp_path, 'dev', 'BT-005', config(disable_native_alignment=True))
    np.testing.assert_array_equal(data['native_auxiliary'], ir['native_s0'])
    changed = dict(data, boxes=data['boxes'] + 1)
    with pytest.raises(AssertionError):
        load_neck72_evidence(changed, tmp_path, 'dev', 'BT-005', config())
    path = folder / 'new.npz'; path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        load_neck72_evidence(data, tmp_path, 'dev', 'BT-005', config())


@pytest.mark.parametrize('name', ['train_native_neck_clean_query', 'train_native_neck_robust_query'])
def test_original_training_step_computation_unchanged(name):
    path = 'tools/' + name + '.py'
    original = subprocess.check_output(['git', 'show', ORIGINAL + ':' + path], cwd=ROOT, text=True)
    current = (ROOT / path).read_text()
    def training_step(source):
        return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == 'step')
    assert ast.dump(training_step(original)) == ast.dump(training_step(current))


@pytest.mark.parametrize('name', ['extract_native_neck_candidate_features', 'extract_native_neck_eval_features'])
def test_original_pixel_forward_computation_unchanged(name):
    path = 'tools/' + name + '.py'
    original = subprocess.check_output(['git', 'show', ORIGINAL + ':' + path], cwd=ROOT, text=True)
    current = (ROOT / path).read_text()
    def forward(source):
        return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.With)
                    and any(isinstance(i.context_expr, ast.Call)
                            and isinstance(i.context_expr.func, ast.Attribute)
                            and i.context_expr.func.attr == 'inference_mode' for i in n.items))
    assert ast.dump(forward(original)) == ast.dump(forward(current))
