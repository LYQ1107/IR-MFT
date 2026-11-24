#!/usr/bin/env python3
"""Actual pending F4 entry and original RGB CPU loading; no fusion transfer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from tools.fusion_memory_model_loading import DevQualifiedModels, load_source_rgb_cpu
from tools.fusion_memory_parent_io import DevParentFiles, ParentEvidencePending
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage

SPEC = ROOT/'configs/perceptual_f4_model_loading_20261125.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pytest-receipt', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.monotonic()
    spec = read(SPEC)
    assert all(sha(ROOT/name) == value for name, value in spec['frozen_dependencies'].items())
    tests = read(args.pytest_receipt)
    assert tests['exit_code'] == 0 and sha(tests['log']) == tests['log_sha256']
    assert all(sha(ROOT/name) == value for name, value in tests['source_hashes'].items())
    found = re.search(r'\b(\d+) passed\b', Path(tests['log']).read_text())
    assert found and int(found.group(1)) == 20
    work = Path(read(ROOT/'configs/perceptual_mft_20261122.json')['work_root'])
    output = work/'fusion_memory_model_loading_CPU_proof.json'
    assert not output.exists()
    require_host_memory(4*2**30, maximum_workers=1)
    require_storage(work)
    attempts = dict(forbidden_data_reads=0, parent_entry_tensor_loads=0, non_RGB_tensor_loads=0)
    def audit(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if (DevParentFiles.is_test_accuracy(path) or path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.npy', '.npz']):
            attempts['forbidden_data_reads'] += 1
            raise AssertionError('No images, FPN arrays, GT or Test accuracy may enter loading proof')
    sys.addaudithook(audit)
    def forbid_parent(*a, **kw):
        attempts['parent_entry_tensor_loads'] += 1
        raise AssertionError('Actual missing parent files must stop before loading tensors')
    with patch.object(torch, 'load', forbid_parent):
        try:
            DevQualifiedModels()
        except ParentEvidencePending as pending:
            missing = pending.paths
        else:
            raise AssertionError('This current-state proof expects incomplete original Dev parents')
    assert len(missing) == 6
    source = Path(spec['source_RGB_checkpoint'])
    assert sha(source) == spec['source_RGB_sha256']
    actual_loads, original_load = [], torch.load
    def rgb_only(stream, *args, **kwargs):
        path = Path(getattr(stream, 'name', stream)).resolve()
        if path != source.resolve():
            attempts['non_RGB_tensor_loads'] += 1
            raise AssertionError('Only the exact original RGB checkpoint may load')
        actual_loads.append(str(path))
        return original_load(stream, *args, **kwargs)
    with patch.object(torch, 'load', rgb_only):
        model = load_source_rgb_cpu(source, spec['source_RGB_sha256'])
    state = model.state_dict()
    fingerprint = state_sha256(state)
    assert all(v.device.type == 'cpu' for v in state.values())
    assert all(v.dtype == (torch.float32 if v.is_floating_point() else torch.int64) for v in state.values())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters()) and not model.training
    assert actual_loads == [str(source.resolve())] and not any(attempts.values())
    assert not torch.cuda.is_initialized()
    names = ['tools/fusion_memory_model_loading.py', 'tools/probe_f4_model_loading.py',
        'tests/test_fusion_memory_model_loading.py', str(SPEC.relative_to(ROOT))]
    sources = {name: sha(ROOT/name) for name in names}
    sources.update(spec['frozen_dependencies'])
    result = dict(status='COMPLETE_CPU_STRICT_TENSOR_COMPONENT_ACTUAL_PARENT_PENDING',
        specification_sha256=sha(SPEC), source_hashes=sources, synthetic_tests_passed=20,
        pytest_receipt=str(args.pytest_receipt), pytest_receipt_sha256=sha(args.pytest_receipt),
        pytest_log_sha256=tests['log_sha256'], actual_parent_status='PENDING_ORIGINAL_COMPLETE_PARENT_DEV_FILES',
        actual_missing_parent_markers=missing, actual_parent_eligibility=None,
        actual_learned_fusion_tensor_loads=0, actual_RGB_tensor_loads=1, source_RGB_sha256=sha(source),
        actual_CPU_RGB_state_sha256=fingerprint, actual_RGB_parameters=sum(p.numel() for p in model.parameters()),
        actual_RGB_state_tensors=len(state), actual_RGB_integer_buffers=sum(not v.is_floating_point() for v in state.values()),
        RGB_frozen_eval_and_no_grad=True, actual_images_labels_Dev_Test_read=False,
        guarded_attempt_counters=attempts, actual_learned_fusion_transferred=False, new_accuracy=False,
        new_NN_fits=0, new_GPU_work=False, persistent_new_weights_saved=False, new_image_FPN_disk_cache=False,
        synthetic_checkpoints='Temporary untrained tensor fixtures only, separate from actual proof; no research fits.',
        formal_F4_launch_ready=False, fixed_seed=0, additional_seed_work=False,
        seconds=time.monotonic()-started, peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        scope='Strict CPU helpers and pending real parent entry; no actual learned fusion model has passed tensor loading/Dev/GPU qualification.',
        goal_met=False, supported_innovations=0)
    assert result['peak_RSS_GiB'] <= 4
    require_storage(work)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), seconds=result['seconds'], peak_RSS_GiB=result['peak_RSS_GiB'],
        RGB_parameters=result['actual_RGB_parameters'], missing_parent_markers=len(missing), new_accuracy=False)), flush=True)


if __name__ == '__main__':
    main()
