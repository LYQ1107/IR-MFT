import numpy as np
import torch
from cfp.dictionary_separated_joint import DictionarySeparatedJoint
from cfp.joint_set_supervision import consistent_past_identity, joint_identity_target_sets, balanced_set_identity_loss


def inputs():
    torch.manual_seed(4)
    return dict(history=torch.randn(3, 4, 384), times=torch.tensor([[2, 4, 6, 0]]*3),
                valid=torch.tensor([[True, True, True, False]]*3), frame=torch.full((3,), 10),
                detections=torch.randn(4, 384), detection_valid=torch.ones(4, dtype=torch.bool),
                geometry=torch.rand(3, 4), appearance=torch.rand(3, 4), track_state=torch.rand(3, 3),
                confidence=torch.full((4,), .8), codes=torch.tensor([7, 20, 91]))


def test_content_association_is_bitexact_invariant_to_arbitrary_code_bindings():
    model = DictionarySeparatedJoint().eval()
    a = inputs(); b = dict(a, codes=torch.tensor([501, 70, 1]))
    first, second = model(**a), model(**b)
    assert torch.equal(first['logits'], second['logits'])
    assert not torch.equal(first['slot_codes'], second['slot_codes'])
    first['logits'].sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.dictionary.parameters())


def test_content_and_code_controls_have_exact_active_parameter_capacity():
    content = DictionarySeparatedJoint(binding_mode='content')
    codes = DictionarySeparatedJoint(binding_mode='codes')
    assert sum(p.numel() for p in content.parameters() if p.requires_grad) == sum(p.numel() for p in codes.parameters() if p.requires_grad)
    assert sum(p.numel() for p in content.dictionary.parameters()) == 512*64


def test_multi_positive_old_slots_and_unknowns_are_not_arbitrary_negative_or_null():
    targets = joint_identity_target_sets(torch.tensor([7, 7, -1, 10]), torch.tensor([7, 30, -1]))
    assert targets['positives'][0].tolist() == [True, True, False, False, False]
    assert not targets['positives'][1:].any()
    logits = torch.tensor([[1., 2., 0., -.5, -.3], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], requires_grad=True)
    legal = torch.ones(4, 3, dtype=torch.bool)
    loss, parts = balanced_set_identity_loss(logits, targets, legal, consistency_weight=0)
    altered = logits.detach().clone(); altered[:, 2] = 10000
    other, _ = balanced_set_identity_loss(altered, targets, legal, consistency_weight=0)
    assert torch.equal(loss, other) and parts['multi_positive'] == 1
    loss.backward(); assert logits.grad[:, 2].abs().sum() == 0 and logits.grad[1:].abs().sum() == 0


def test_only_wholly_known_consistent_recent_history_can_supply_identity():
    indices = np.array([0, 1, 2])
    assert consistent_past_identity(indices, np.array([7, 7, 7])) == 7
    assert consistent_past_identity(indices, np.array([7, 7, -1])) == -1
    assert consistent_past_identity(indices, np.array([7, 7, 8])) == -1
    assert consistent_past_identity(indices[:2], np.array([7, 7, 7])) == -1
