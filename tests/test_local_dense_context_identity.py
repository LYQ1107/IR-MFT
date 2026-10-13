import torch

from cfp.local_dense_context_identity import ContextIdentityStudent
from cfp.local_dense_jepa import DenseLocalTarget, dense_prediction_loss


def inputs():
    return dict(history=torch.randn(2, 3, 4, 24), history_fraction=torch.ones(2, 3, 4),
        history_global=torch.randn(2, 3, 24), times=torch.tensor([[1, 4, 8], [2, 5, 9]]),
        history_valid=torch.ones(2, 3, dtype=torch.bool), frame=12,
        candidates=torch.randn(3, 4, 24), candidate_fraction=torch.ones(3, 4),
        candidate_global=torch.randn(3, 24))


def test_prediction_auxiliary_does_not_replace_identity_representation_or_run_online():
    model = ContextIdentityStudent(5, 'full', dimension=24, heads=4).eval()
    x = inputs()
    original = model(**x)
    with torch.no_grad():
        model.queries.add_(100)
        model.predictor.in_proj_weight.mul_(3)
    changed = model(**x)
    torch.testing.assert_close(original['scores'], changed['scores'], rtol=0, atol=0)
    torch.testing.assert_close(original['representation'], changed['representation'], rtol=0, atol=0)
    assert not torch.allclose(original['predicted'], changed['predicted'])
    def forbidden(*args, **kwargs):
        raise RuntimeError('Prediction must not execute online')
    model.predictor.forward = forbidden
    torch.testing.assert_close(model.online(**x)['scores'], original['scores'], rtol=0, atol=0)


def test_training_auxiliary_still_updates_encoder_temporal_and_decoder_not_teacher():
    model = ContextIdentityStudent(5, 'full', dimension=24, heads=4)
    teacher = DenseLocalTarget(model)
    output = model(**inputs())
    target = teacher(torch.randn(2, 4, 24), torch.ones(2, 4))
    loss, count = dense_prediction_loss(output['predicted'], target, torch.ones(2, 4, dtype=torch.bool), output['history_available'])
    loss.backward()
    assert count==8
    for prefix in ('local_encoder.', 'temporal.', 'predictor.'):
        assert any(n.startswith(prefix) and p.grad is not None and p.grad.abs().sum()>0 for n, p in model.named_parameters())
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())


def test_no_prediction_and_local_temporal_are_declared_exact_control_aliases():
    torch.manual_seed(8)
    a = ContextIdentityStudent(5, 'no_prediction', dimension=24, heads=4)
    torch.manual_seed(8)
    b = ContextIdentityStudent(5, 'local_temporal', dimension=24, heads=4)
    assert [n for n,p in a.named_parameters() if p.requires_grad] == [n for n,p in b.named_parameters() if p.requires_grad]
    x = inputs()
    for key in ('scores', 'representation', 'predicted'):
        torch.testing.assert_close(a(**x)[key], b(**x)[key], rtol=0, atol=0)
