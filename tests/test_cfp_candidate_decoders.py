import pytest
import torch

from cfp.candidate_id_decoder import CandidateIDDecoder
from cfp.candidate_transformer import CandidateTransformer


def inputs():
    torch.manual_seed(9)
    return (torch.randn(2, 8, 512), torch.tensor([[1, 2, 0, 0, 0, 0, 0, 0]]).repeat(2, 1),
            torch.tensor([[True, True, False, False, False, False, False, False]]).repeat(2, 1),
            torch.tensor([4, 4]), torch.randn(3, 512))


def test_id_dictionary_tracks_and_candidates_permute_without_label_leak():
    h, t, v, f, c = inputs()
    model = CandidateIDDecoder().eval()
    codes = torch.tensor([13, 79])
    with torch.no_grad():
        a = model(h, t, v, f, c, codes)
        b = model(h.flip(0), t.flip(0), v.flip(0), f.flip(0), c.flip(0), codes.flip(0))
    assert a.shape == (3, 3)
    assert torch.allclose(a[:, [1, 0, 2]].flip(0), b, atol=1e-5)
    with pytest.raises(ValueError):
        model(h, t, v, f, c, torch.tensor([1, 1]))


def test_id_decoder_ignores_hidden_past_and_requires_strict_past():
    h, t, v, f, c = inputs()
    model = CandidateIDDecoder().eval()
    a = model(h, t, v, f, c)
    h[~v] = 1000
    assert torch.allclose(a, model(h, t, v, f, c), atol=1e-6)
    t[0, 1] = 4
    with pytest.raises(ValueError):
        model(h, t, v, f, c)


def test_id_decoder_empty_history_only_predicts_newborn():
    h, t, v, f, c = inputs()
    v[:] = False; t[:] = 0
    out = CandidateIDDecoder()(h, t, v, f, c)
    assert torch.isneginf(out[:, :2]).all() and torch.isfinite(out[:, 2]).all()


def test_transformer_same_inputs_masks_and_candidate_permutation():
    h, t, v, f, c = inputs()
    c = c[None].repeat(2, 1, 1); legal = torch.ones(2, 3, dtype=torch.bool)
    evidence = torch.randn(2, 3, 21)
    model = CandidateTransformer().eval()
    a = model(h, t, v, f, c, legal, evidence)['scores']
    h[~v] = -10000; evidence[..., :9] += 100; evidence[..., 19:] += 100
    b = model(h, t, v, f, c.flip(1), legal, evidence.flip(1))['scores']
    assert torch.allclose(a[:, [2, 1, 0, 3]], b, atol=1e-6)
    legal[:] = False; v[:] = False; t[:] = 0
    out = model(h, t, v, f, c, legal, evidence)['scores']
    assert torch.isfinite(out[:, -1]).all()
    out[:, -1].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
