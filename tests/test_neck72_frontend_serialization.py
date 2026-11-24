import hashlib
import io

import pytest
import torch

from tools.prove_neck72_frontend_serialization import compare_states, load_model_bytes


def test_equal_fp16_rounding_preserves_the_complete_module_half_state():
    original = dict(weight=torch.tensor([1.00005, -2.00005]),
        running_var=torch.tensor([.99999]), num_batches_tracked=torch.tensor(4097, dtype=torch.int64))
    compact = {name: value.half() if value.is_floating_point() else value.clone()
               for name, value in original.items()}
    result = compare_states(original, compact)
    assert result['all_state_tensors_exact'] and result['compared_tensors'] == 3
    assert result['preserved_integer_or_boolean_buffers'] == 1
    assert result['canonical_state_sha256']['original'] == result['canonical_state_sha256']['compact']
    compact['num_batches_tracked'] += 1
    assert not compare_states(original, compact)['all_state_tensors_exact']


def test_missing_unused_classifier_or_changed_shape_is_not_ignored():
    original = dict(weight=torch.ones(2, 3), classifier=torch.ones(5, 3))
    missing = compare_states(original, {'weight': torch.ones(2, 3)})
    assert not missing['all_state_tensors_exact'] and missing['missing_keys'] == ['classifier']
    assert not compare_states(original, dict(weight=torch.ones(3, 2), classifier=torch.ones(5, 3)))['all_state_tensors_exact']
    original['weight'][0, 0] = float('nan')
    assert not compare_states(original, original)['all_state_tensors_exact']


def test_integer_buffer_cast_to_float_is_rejected():
    original = dict(num_batches_tracked=torch.tensor(4097, dtype=torch.int64))
    compact = dict(num_batches_tracked=original['num_batches_tracked'].half())
    assert not compare_states(original, compact)['all_state_tensors_exact']


def test_checkpoint_hash_covers_the_exact_buffer_before_deserialization(tmp_path, monkeypatch):
    buffer = io.BytesIO(); torch.save(dict(model={'weight': torch.ones(2)}, optimizer={'step': 3}), buffer)
    raw = buffer.getvalue(); path = tmp_path / 'small-fixture.pt'; path.write_bytes(raw)
    model, provenance = load_model_bytes(path, hashlib.sha256(raw).hexdigest())
    assert model['weight'].shape == (2,) and provenance['bytes_read'] == len(raw)
    called = []
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: called.append(True))
    with pytest.raises(ValueError, match='Checkpoint bytes differ'):
        load_model_bytes(path, '0'*64)
    assert not called
