#!/usr/bin/env python3
"""Compare complete module.half-equivalent state without CNN/GPU inference."""
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from tools.neck72_experiment import context
from tools.neck72_runtime import read, require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256


def load_model_bytes(path, expected_sha256):
    """Hash exactly one buffered file read, then release unneeded optimizer data.

    BytesIO is filled directly, not constructed from another full bytes copy.
    The largest original file is2.084GiB; raw bytes plus deserialized CPU
    tensors fit the explicit8GiB admission bound. No weight file is copied.
    """
    digest = hashlib.sha256(); count = 0
    buffer = io.BytesIO()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            digest.update(block); buffer.write(block); count += len(block)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        buffer.close()
        raise ValueError('Checkpoint bytes differ from the registered source: '+str(path))
    buffer.seek(0)
    try:
        value = torch.load(buffer, map_location='cpu', weights_only=False)
    finally:
        buffer.close()
    model = value['model']
    provenance = dict(file=str(path), file_sha256=actual, bytes_read=count,
        serialized_fields=sorted(value),
        saved_seed=value.get('manifest', {}).get('seed'),
        saved_completed_epochs=value.get('completed_epochs'))
    del value
    gc.collect()
    return model, provenance


def canonical_tensor(value):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
        raise ValueError('Expected a complete CPU tensor state dictionary')
    if value.is_complex():
        raise ValueError('Unexpected complex detector state')
    # nn.Module.half converts floating parameters/buffers, preserving integer
    # counters. Casting num_batches_tracked to FP16 would be a false proof.
    return value.to(dtype=torch.float16).contiguous() if value.is_floating_point() else value.contiguous()


def tensor_digest(value):
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(shape=list(value.shape), dtype=str(value.dtype)), sort_keys=True).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def compare_states(original, compact):
    missing, extra = sorted(set(original)-set(compact)), sorted(set(compact)-set(original))
    rows = []; digests = [hashlib.sha256(), hashlib.sha256()]
    for name in sorted(set(original) & set(compact)):
        left, right = canonical_tensor(original[name]), canonical_tensor(compact[name])
        finite = (not left.is_floating_point() or bool(torch.isfinite(left).all()))
        finite &= (not right.is_floating_point() or bool(torch.isfinite(right).all()))
        same = finite and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
        hashes = [tensor_digest(left), tensor_digest(right)]
        for target, value in zip(digests, hashes):
            target.update(name.encode()); target.update(value.encode())
        rows.append(dict(name=name, shape=list(left.shape), compact_shape=list(right.shape),
            original_dtype=str(original[name].dtype), compact_dtype=str(compact[name].dtype),
            canonical_dtype=str(left.dtype), elements=left.numel(), exact=bool(same),
            canonical_original_sha256=hashes[0], canonical_compact_sha256=hashes[1]))
    return dict(all_state_tensors_exact=not missing and not extra and all(row['exact'] for row in rows),
        missing_keys=missing, extra_keys=extra, compared_tensors=len(rows),
        floating_tensors=sum(original[row['name']].is_floating_point() for row in rows),
        preserved_integer_or_boolean_buffers=sum(not original[row['name']].is_floating_point() for row in rows),
        canonical_state_sha256=dict(original=digests[0].hexdigest(), compact=digests[1].hexdigest()),
        tensor_rows=rows)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    plan, protocol, work = context()
    output = work / 'neck72_frontend_serialization_proof.json'; assert not output.exists()
    require_host_memory(8*2**30, maximum_workers=1); require_storage(work)
    started = time.monotonic(); rows = []
    source_hashes = {name: sha256(ROOT / name) for name in [
        'tools/prove_neck72_frontend_serialization.py', 'tests/test_neck72_frontend_serialization.py']}
    for arm, prefix in [('new', 'new_diffv2ir'), ('rgb', 'rgb')]:
        protocols = {}
        for split in ['dev', 'test']:
            path = Path(plan['frozen_detection_sources'][prefix+'_'+split]) / 'metrics.json'
            value = read(path)['protocol']
            assert value['seed'] == 0 and value['variant'] == ('rgb' if arm == 'rgb' else 'diffv2ir')
            assert value['input_size'] == [800, 1440] and value['conf'] == .1 and value['nms'] == .7
            protocols[split] = dict(protocol=value, path=str(path), sha256=sha256(path))
        original_info, compact_info = [protocols[split]['protocol'] for split in ['dev', 'test']]
        print('READ_ORIGINAL', arm, Path(original_info['ckpt']).stat().st_size, flush=True)
        original, original_source = load_model_bytes(original_info['ckpt'], original_info['source_sha256'])
        print('READ_COMPACT', arm, Path(compact_info['ckpt']).stat().st_size, flush=True)
        compact, compact_source = load_model_bytes(compact_info['ckpt'], compact_info['source_sha256'])
        comparison = compare_states(original, compact)
        rows.append(dict(frontend=arm, source_protocols=protocols,
            original=original_source, compact=compact_source, comparison=comparison,
            canonical_half_state_covers_all_parameters_BN_and_integer_buffers=True))
        print('COMPARED', arm, comparison['all_state_tensors_exact'], comparison['compared_tensors'], flush=True)
        del original, compact
        gc.collect()
    assert not torch.cuda.is_initialized()
    assert source_hashes == {name: sha256(ROOT / name) for name in source_hashes}
    result = dict(status='COMPLETE_NECK72_FRONTEND_SERIALIZATION_COMPARISON', rows=rows,
        all_frontends_exact=all(row['comparison']['all_state_tensors_exact'] for row in rows),
        source_hashes=source_hashes, fixed_seed=0, additional_seed_work=False,
        GPU_work=False, CNN_forward=False, new_accuracy=False, new_runtime_measurement=False,
        weight_files_created=0, original_files_modified=False,
        peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        seconds=time.monotonic()-started,
        scope='Every original Dev detector floating tensor rounded exactly as nn.Module.half, '
            'and every unchanged integer/bool buffer, compared to the existing compact Dev-selected '
            'Test artifact. Uses protocol/weight metadata only, no Test accuracy or checkpoint search. '
            'A matching state permits serialization reuse, not a different model, B1/B4 numerical '
            'equivalence, fresh-image correctness, measured speed, accuracy gain or innovation.',
        goal_met=False, supported_innovations=0)
    require_storage(work); dump(output, result)
    print(result['status'], 'all_frontends_exact', result['all_frontends_exact'],
          'seconds', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
