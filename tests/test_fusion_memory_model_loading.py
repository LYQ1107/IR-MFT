"""Untrained temporary tensor fixtures only, never qualified research models."""
import copy
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tools.fusion_memory_model_loading import (DevQualifiedModels, load_fusion_snapshot_cpu,
    load_pinned_torch, strict_cpu_state)
from tools.fusion_memory_parent_io import ParentEvidencePending
from tools.spatial_native_fusion import SpatialNativeFusion
from yolox.models.native_pyramid_fusion import NativePyramidFusion


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artificial_snapshot(kind='quality'):
    family = 'spatial' if kind == 'spatial' else 'native'
    kwargs = dict(channels=(8, 16, 32))
    if family == 'native':
        kwargs['mode'] = kind
    torch.manual_seed(0)
    module = (SpatialNativeFusion if family == 'spatial' else NativePyramidFusion)(**kwargs)
    with torch.no_grad():
        module.output[0].bias.fill_(.125)  # Fixed untrained engineering state.
    model = dict(id='ARTIFICIAL_NOT_RESEARCH', fit_id='artificial_ir', epoch=2, cue='ir', seed=0,
        removal=None, mode=kind if family == 'native' else 'quality', geometry_fixed=False)
    if family == 'spatial':
        model['variant'] = 'spatial'
    arm = dict(model, module_kwargs=kwargs)
    metadata = dict(fit_id=model['fit_id'], seed=0, cue='ir', removal=None, mode=model['mode'],
        Dev_fit_rows=0, Test_fit_rows=0, **({'variant': 'spatial'} if family == 'spatial' else {}))
    history = [dict(epoch=e, delivery={'SYNTHETIC_NOT_REAL_TRAIN': True}, mean_clip_losses={'fixture': 1.}) for e in [1, 2, 3]]
    progress = dict(metadata=metadata, history=history)
    saved = dict(metadata=copy.deepcopy(metadata), history=copy.deepcopy(history[:2]),
        completed_epochs=2, source_RGB_only_by_reference=True,
        module={k: v.clone() for k, v in module.state_dict().items()})
    return family, model, arm, progress, saved


@pytest.mark.parametrize('kind', ['quality', 'temporal', 'spatial'])
def test_actual_classes_strict_tensor_load_frozen_values_and_rng(tmp_path, kind):
    family, model, arm, progress, saved = artificial_snapshot(kind)
    path = tmp_path/'UNTRAINED_TENSOR_FIXTURE.pt'
    torch.save(saved, path)
    before = torch.get_rng_state().clone()
    value = load_fusion_snapshot_cpu(path, sha(path), family, model, arm, progress)
    assert not value.training and all(not p.requires_grad and p.grad is None for p in value.parameters())
    assert all(p.device.type == 'cpu' for p in value.parameters())
    assert all(torch.equal(v, saved['module'][k]) for k, v in value.state_dict().items())
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('fault', ['history', 'metadata', 'epoch', 'RGB_copy', 'missing_key',
    'extra_key', 'shape', 'dtype', 'nonfinite', 'wrong_cue', 'wrong_geometry'])
def test_corrupt_snapshot_cannot_pass_by_having_correct_outer_byte_hash(tmp_path, fault):
    family, model, arm, progress, saved = artificial_snapshot()
    key = next(iter(saved['module']))
    if fault == 'history':
        saved['history'][0]['mean_clip_losses']['fixture'] = 2.
    elif fault == 'metadata':
        saved['metadata']['Dev_fit_rows'] = 1
    elif fault == 'epoch':
        saved['completed_epochs'] = 3
    elif fault == 'RGB_copy':
        saved['source_RGB_only_by_reference'] = False
    elif fault == 'missing_key':
        saved['module'].pop(key)
    elif fault == 'extra_key':
        saved['module']['unexpected_RGB_encoder'] = torch.zeros(1)
    elif fault == 'shape':
        saved['module'][key] = saved['module'][key].flatten()[:1]
    elif fault == 'dtype':
        saved['module'][key] = saved['module'][key].double()
    elif fault == 'nonfinite':
        saved['module'][key].view(-1)[0] = float('nan')
    elif fault == 'wrong_cue':
        model['cue'] = 'gray'
    else:
        model['geometry_fixed'] = True
    path = tmp_path/'CORRUPT_ARTIFICIAL_STATE.pt'
    torch.save(saved, path)
    with pytest.raises(ValueError):
        load_fusion_snapshot_cpu(path, sha(path), family, model, arm, progress)


def test_checksum_rejection_happens_before_torch_load(tmp_path):
    path = tmp_path/'not_a_model.pt'
    path.write_text('artificial fixture')
    with patch.object(torch, 'load', side_effect=AssertionError('Tensor loading was attempted')) as load:
        with pytest.raises(ValueError, match='before tensor loading'):
            load_pinned_torch(path, 'f'*64)
    load.assert_not_called()


@pytest.mark.parametrize('replacement', [False, True])
def test_open_file_or_path_changed_during_load_is_rejected(tmp_path, replacement):
    path = tmp_path/'fixture.pt'
    torch.save({'artificial': torch.zeros(1)}, path)
    expected, original_load = sha(path), torch.load
    def changed(stream, *args, **kwargs):
        value = original_load(stream, *args, **kwargs)
        if replacement:
            other = tmp_path/'replacement.pt'
            other.write_bytes(path.read_bytes()+b'CHANGED')
            other.replace(path)
        else:
            with path.open('ab') as output:
                output.write(b'CHANGED')
        return value
    with patch.object(torch, 'load', changed):
        with pytest.raises(ValueError, match='changed during tensor loading'):
            load_pinned_torch(path, expected)


def test_same_checkpoint_loads_have_independent_parameters_for_intervention_routes(tmp_path):
    family, model, arm, progress, saved = artificial_snapshot()
    path = tmp_path/'untrained.pt'
    torch.save(saved, path)
    a = load_fusion_snapshot_cpu(path, sha(path), family, model, arm, progress)
    b = load_fusion_snapshot_cpu(path, sha(path), family, model, arm, progress)
    assert a is not b
    for x, y in zip(a.parameters(), b.parameters()):
        assert x.data_ptr() != y.data_ptr() and torch.equal(x, y)


def test_strict_cpu_state_preserves_integer_buffers():
    value = nn.BatchNorm2d(2)
    state = {k: v.clone() for k, v in value.state_dict().items()}
    state['num_batches_tracked'].fill_(23)
    strict_cpu_state(value, state)
    assert value.num_batches_tracked.dtype == torch.int64 and value.num_batches_tracked.item() == 23


def test_pending_parent_entry_never_attempts_tensor_loading():
    markers = [str(Path('/ARTIFICIAL_PARENT_NOT_RESEARCH')/name) for name in ['native_complete', 'spatial_complete']]
    class Pending:
        def __init__(self, root):
            pass
        def load(self):
            raise ParentEvidencePending(markers)
    with patch('tools.fusion_memory_model_loading.DevParentFiles', Pending), patch.object(
            torch, 'load', side_effect=AssertionError('No parent tensor may load')) as load:
        with pytest.raises(ParentEvidencePending):
            DevQualifiedModels()
    load.assert_not_called()
