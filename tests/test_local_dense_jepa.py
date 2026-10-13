import torch
import pytest

from cfp.local_dense_jepa import (LocalDenseStudent, DenseLocalTarget,
                                  known_candidate_loss, dense_prediction_loss, MODES)


def inputs():
    return dict(history=torch.randn(2, 3, 4, 24), history_fraction=torch.ones(2, 3, 4),
        history_global=torch.randn(2, 3, 24), times=torch.tensor([[1, 5, 9], [2, 6, 10]]),
        history_valid=torch.ones(2, 3, dtype=torch.bool), frame=12,
        candidates=torch.randn(3, 4, 24), candidate_fraction=torch.ones(3, 4),
        candidate_global=torch.randn(3, 24))


def model(mode='full'):
    return LocalDenseStudent(5, mode, dimension=24, heads=4)


def test_prediction_has_no_current_candidate_or_teacher_target_path():
    student = model().eval()
    x = inputs()
    original = student(**x)
    changed = dict(x, candidates=x['candidates']*100, candidate_global=x['candidate_global']*100)
    torch.testing.assert_close(student(**changed)['predicted'], original['predicted'], rtol=0, atol=0)
    teacher = DenseLocalTarget(student)
    teacher(torch.randn(2, 4, 24)*1000, torch.ones(2, 4))
    torch.testing.assert_close(student(**x)['predicted'], original['predicted'], rtol=0, atol=0)
    with pytest.raises(ValueError, match='strictly earlier'):
        student(**dict(x, times=torch.tensor([[1, 5, 12], [2, 6, 10]])))


def test_invisible_history_and_candidate_regions_cannot_change_scores():
    student = model().eval()
    x = inputs()
    x['history_fraction'][:, :, 0] = 0
    x['candidate_fraction'][:, 0] = 0
    x['history_valid'][:, 0] = False
    x['times'][:, 0] = 0
    original = student(**x)
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in x.items()}
    changed['history'][:, :, 0] = 1.e5
    changed['history'][:, 0] = -1.e5
    changed['history_global'][:, 0] = 1.e5
    changed['candidates'][:, 0] = 1.e5
    torch.testing.assert_close(student(**changed)['scores'], original['scores'], rtol=0, atol=0)


def test_candidate_permutation_equivariance_and_training_target_stop():
    student = model().eval()
    x = inputs()
    original = student(**x)
    order = torch.tensor([2, 0, 1])
    changed = dict(x)
    for name in ('candidates', 'candidate_fraction', 'candidate_global'):
        changed[name] = changed[name][order]
    torch.testing.assert_close(student(**changed)['scores'], original['scores'][:, order])
    teacher = DenseLocalTarget(student, .9)
    target_input = torch.randn(2, 4, 24, requires_grad=True)
    target = teacher(target_input, torch.ones(2, 4))
    loss, tokens = dense_prediction_loss(original['predicted'], target, torch.tensor([[True, False, True, False]]).expand(2, -1), original['history_available'])
    loss.backward()
    assert tokens == 4 and target_input.grad is None
    assert student.queries.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.temporal.parameters())
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(student.local_encoder.parameters(), teacher.encoder.parameters()))


def test_teacher_updates_after_success_only_and_is_checkpointed():
    student = model()
    teacher = DenseLocalTarget(student, .75)
    teacher.train()
    assert not teacher.training and not teacher.encoder.training
    old = {n: p.clone() for n, p in teacher.encoder.named_parameters()}
    with torch.no_grad():
        for p in student.local_encoder.parameters():
            p.add_(1)
    teacher.update(student, successful=False)
    assert int(teacher.updates) == 0
    teacher.update(student)
    assert int(teacher.updates) == 1
    for name, parameter in teacher.encoder.named_parameters():
        torch.testing.assert_close(parameter, old[name]+.25)
    restored = DenseLocalTarget(student, .75)
    restored.load_state_dict(teacher.state_dict(), strict=True)
    assert int(restored.updates) == 1


def test_unknown_candidate_is_not_a_negative_and_absence_is_not_a_false_target():
    ids = torch.tensor([3, 7])
    candidates = torch.tensor([3, -1, 9])
    scores = torch.tensor([[1., 100., 0.], [2., 100., 0.]], requires_grad=True)
    loss, rows = known_candidate_loss(scores, ids, candidates, torch.ones(2, dtype=torch.bool))
    assert rows == 1
    torch.testing.assert_close(loss, torch.logsumexp(torch.tensor([1., 0.]), 0)-1)
    loss.backward()
    assert scores.grad[:, 1].abs().sum() == 0 and scores.grad[1].abs().sum() == 0


@pytest.mark.parametrize('mode', MODES)
def test_empty_or_wholly_missing_history_has_no_hallucinated_evidence(mode):
    student = model(mode).eval()
    x = inputs()
    x['history_valid'].zero_()
    x['times'].zero_()
    result = student(**x)
    assert not result['history_available'].any() and torch.count_nonzero(result['predicted']) == 0
    assert torch.isfinite(result['scores']).all() and torch.count_nonzero(result['scores']) == 0
    empty = dict(x)
    for name in ('candidates', 'candidate_fraction', 'candidate_global'):
        empty[name] = empty[name][:0]
    assert student(**empty)['scores'].shape == (2, 0)
    for name in ('history', 'history_fraction', 'history_global', 'times', 'history_valid'):
        empty[name] = empty[name][:0]
    assert student(**empty)['scores'].shape == (0, 0)
