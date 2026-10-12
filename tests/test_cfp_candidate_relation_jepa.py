import torch

from cfp.candidate_relation_jepa import CandidateRelationStudent, CandidateRelationTarget


def inputs():
    return [torch.randn(2, 8, 16), torch.arange(1, 9)[None].repeat(2, 1),
        torch.ones(2, 8, dtype=torch.bool), torch.full((2,), 10),
        torch.randn(2, 3, 16), torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 3, 21)]


def test_raw_relation_scores_candidate_equivariance_and_no_hidden_cache_leak():
    model = CandidateRelationStudent().eval()
    x = inputs(); x[2][:, 4:] = False; x[1][:, 4:] = 0
    original = model(*x)
    changed = [a.clone() for a in x]; changed[0][:, 4:] = 1000
    changed[6][..., :9] = 1000; changed[6][..., 19:] = 1000
    torch.testing.assert_close(model(*changed)['scores'], original['scores'])
    perm = torch.tensor([2, 0, 1]); changed = [a.clone() for a in x]
    for i in (4, 5, 6):
        changed[i] = changed[i][:, perm]
    torch.testing.assert_close(model(*changed)['scores'][:, :-1], original['scores'][:, perm])


def test_empty_and_fully_masked_history_have_finite_null_and_geometry():
    model = CandidateRelationStudent().eval()
    x = inputs(); x[2].fill_(False); x[1].zero_()
    assert torch.isfinite(model(*x)['scores']).all()
    x[5].fill_(False)
    output = model(*x)
    assert torch.isfinite(output['scores'][:, -1]).all() and not torch.isfinite(output['scores'][:, :-1]).any()
    output['scores'][:, -1].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_padding_single_candidate_does_not_create_fake_margin():
    model = CandidateRelationStudent().eval()
    x = inputs()
    x[5][:, 1:] = False
    padded = model(*x)['scores']
    single = [v.clone() for v in x]
    for i in (4, 5, 6):
        single[i] = single[i][:, :1]
    actual = model(*single)['scores']
    torch.testing.assert_close(padded[:, 0], actual[:, 0])
    torch.testing.assert_close(padded[:, -1], actual[:, -1])


def test_teacher_relations_stopped_success_updates_only():
    model = CandidateRelationStudent()
    target = CandidateRelationTarget(model)
    x = inputs()
    output = target(torch.randn(2, 16), x[4], x[5])
    assert not output.requires_grad and output.shape == (2, 3, 16)
    target.update(model, False); assert target.updates == 0
    target.update(model, True); assert target.updates == 1
    assert all(p.grad is None and not p.requires_grad for p in target.parameters())

