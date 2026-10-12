import copy

import pytest
import torch
from torch.nn import functional as F

from cfp.fish_jepa import FishJEPAStudent, update_ema_target


def fixture():
    torch.manual_seed(10)
    features = torch.randn(2, 8, 512)
    frames = torch.arange(1, 9)[None].repeat(2, 1)
    valid = torch.ones(2, 8, dtype=torch.bool)
    query = torch.tensor([9, 9])
    return features, frames, valid, query


def test_teacher_is_stopped_and_ema_updates_actual_student_parameters():
    model = FishJEPAStudent()
    teacher = copy.deepcopy(model.token_encoder).requires_grad_(False).eval()
    with torch.no_grad():
        targets = teacher(torch.randn(2, 512))
    prediction, available, latent, masked = model.training_details(*fixture())
    (1 - F.cosine_similarity(latent, targets)).mean().backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert model.token_encoder.layers[0].weight.grad.norm() > 0
    before = {k: v.clone() for k, v in teacher.named_parameters()}
    with torch.no_grad():
        model.token_encoder.layers[0].weight.add_(.1)
    update_ema_target(teacher, model.token_encoder, .9)
    for name, value in teacher.named_parameters():
        assert torch.allclose(value, .9 * before[name] + .1 * dict(model.token_encoder.named_parameters())[name], atol=1e-6)


def test_masked_raw_values_cannot_leak_to_forecast_or_masked_predictions():
    model = FishJEPAStudent().eval()
    features, frames, valid, query = fixture()
    mask = torch.zeros_like(valid)
    mask[:, 2:5] = True
    changed = features.clone()
    changed[:, 2:5] = torch.randn_like(changed[:, 2:5]) * 100
    with torch.inference_mode():
        a = model.training_details(features, frames, valid, query, mask)
        b = model.training_details(changed, frames, valid, query, mask)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_current_future_times_rejected_and_inference_has_no_teacher():
    model = FishJEPAStudent()
    features, frames, valid, query = fixture()
    with pytest.raises(ValueError):
        model(features, frames, valid, torch.tensor([8, 8]))
    assert not any('teacher' in name for name, _ in model.named_modules())
    assert not any('teacher' in name for name in model.state_dict())
    with pytest.raises(ValueError):
        model.training_details(features, frames, valid, query, torch.ones_like(valid))


@pytest.mark.parametrize('architecture,latent', [('gru', 128), ('transformer', 104)])
def test_decoder_association_space_and_empty_context_are_explicit(architecture, latent):
    model = FishJEPAStudent(architecture, latent).eval()
    features, frames, valid, query = fixture()
    with torch.inference_mode():
        prototype, available = model(features, frames, valid, query)
        empty, missing = model(features, torch.zeros_like(frames), torch.zeros_like(valid), query)
    assert prototype.shape == (2, 512) and available.all()
    assert torch.allclose(prototype.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert not missing.any() and torch.equal(empty, torch.zeros_like(empty))
