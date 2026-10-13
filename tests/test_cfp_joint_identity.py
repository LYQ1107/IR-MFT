import pytest
import torch
import numpy as np
from types import SimpleNamespace

from cfp.joint_identity_decoder import JointFishIDDecoder, bounded_joint_cost, decode_joint_assignment
from cfp.joint_identity_supervision import joint_identity_targets, joint_identity_loss
from cfp.online_joint_identity import install_joint_identity
from cfp.joint_identity_controls import UniformPositiveResidual


def inputs(n=3, m=4):
    generator = torch.Generator().manual_seed(81)
    history = torch.randn(n, 3, 8, generator=generator)
    times = torch.tensor([1, 3, 5]).repeat(n, 1)
    valid = torch.ones(n, 3, dtype=torch.bool)
    return dict(history=history, times=times, valid=valid, frame=torch.full((n,), 7),
                detections=torch.randn(m, 8, generator=generator), detection_valid=torch.ones(m, dtype=torch.bool),
                geometry=torch.rand(n, m, generator=generator), appearance=torch.rand(n, m, generator=generator),
                track_state=torch.rand(n, 3, generator=generator), confidence=torch.rand(m, generator=generator),
                codes=torch.tensor([5, 1, 7])[:n])


def model():
    torch.manual_seed(11)
    return JointFishIDDecoder(dimension=8, hidden=16, capacity=8).eval()


def test_joint_current_or_future_history_is_rejected():
    args = inputs()
    args['times'][0, 2] = 7
    with pytest.raises(ValueError, match='strictly before'):
        model()(**args)


def test_joint_is_equivariant_to_track_and_detection_order_with_local_codes():
    net = model()
    args = inputs()
    expected = net(**args)
    tp, dp = torch.tensor([2, 0, 1]), torch.tensor([3, 1, 0, 2])
    permuted = {k: v[tp] if k in ('history', 'times', 'valid', 'frame', 'track_state', 'codes')
                else v[dp] if k in ('detections', 'detection_valid', 'confidence')
                else v[tp][:, dp] for k, v in args.items()}
    actual = net(**permuted)
    torch.testing.assert_close(actual['logits'][:, :-1], expected['logits'][dp][:, tp], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual['logits'][:, -1], expected['logits'][dp, -1], atol=2e-6, rtol=2e-6)


def test_joint_has_real_cross_track_and_cross_detection_competition():
    net = model()
    args = inputs()
    base = net(**args)['logits']
    changed_track = dict(args, history=args['history'].clone())
    changed_track['history'][2] *= -1
    # Track 0's score changes when only a competing track's history changes.
    assert not torch.allclose(base[:, 0], net(**changed_track)['logits'][:, 0])
    changed_detection = dict(args, detections=args['detections'].clone())
    changed_detection['detections'][3] *= -1
    assert not torch.allclose(base[0], net(**changed_detection)['logits'][0])


def test_pair_control_cannot_see_competing_track_or_detection_features():
    torch.manual_seed(11)
    net = JointFishIDDecoder(dimension=8, hidden=16, capacity=8, architecture='pair').eval()
    args = inputs()
    base = net(**args)['logits']
    args['history'][2] *= -1
    args['detections'][3] *= -1
    torch.testing.assert_close(base[0, 0], net(**args)['logits'][0, 0], rtol=0, atol=1e-7)


def test_joint_invalid_padding_empty_history_and_closed_pairs_are_finite():
    net, args = model(), inputs()
    args['valid'][1] = False
    args['times'][1] = 0
    args['detection_valid'][3] = False
    args['pair_valid'] = torch.zeros(3, 4, dtype=torch.bool)
    output = net(**args)
    assert torch.isneginf(output['logits'][:, :-1]).all()
    assert torch.isfinite(output['logits'][:, -1]).all()
    output['logits'][:, -1].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    for n, m in ((0, 4), (3, 0), (0, 0)):
        assert net(**inputs(n, m))['logits'].shape == (m, n + 1)


def test_unknown_physical_ids_never_become_null_or_known_negative():
    supervision = joint_identity_targets('SN-015', torch.tensor([10, -1, 20]), torch.tensor([10, 30, -1]))
    assert supervision['target'].tolist() == [0, -1, -1]
    assert supervision['classes'][0].tolist() == [True, False, True, True]
    logits = torch.tensor([[2., 80., 1., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]], requires_grad=True)
    loss, metrics = joint_identity_loss(logits, supervision, torch.ones(3, 3, dtype=torch.bool))
    loss.backward()
    assert metrics['supervised_detections'] == 1 and metrics['null'] == 0
    assert logits.grad[0, 1] == 0 and (logits.grad[1:] == 0).all()
    with pytest.raises(ValueError, match='video'):
        joint_identity_targets('', torch.tensor([10]), torch.tensor([10]))


def test_known_no_old_match_null_is_distinct_from_verified_birth_and_duplicates():
    supervision = joint_identity_targets('BT-001', torch.tensor([10, 20]), torch.tensor([10, 30, 20, 20]))
    assert supervision['target'].tolist() == [0, 2, -1, -1]
    assert supervision['reason'][1] == 'known_no_old_match_not_verified_birth'
    ambiguous = joint_identity_targets('BT-001', torch.tensor([10, 10]), torch.tensor([10]))
    assert ambiguous['target'].tolist() == [-1]


def test_joint_assignment_allows_multiple_nulls_but_only_one_detection_per_track():
    logits = torch.tensor([[10., 0.], [9., 0.], [8., 0.], [30., 0.]])
    assigned = decode_joint_assignment(logits, torch.tensor([True, True, True, False]))
    assert assigned.tolist() == [0, -1, -1, -1]
    assert decode_joint_assignment(torch.zeros(3, 1), torch.ones(3, dtype=torch.bool)).tolist() == [-1] * 3


def test_joint_residual_initially_exactly_preserves_cost_and_remains_bounded():
    args = inputs()
    output = model()(**args)
    base = torch.rand(3, 4)
    torch.testing.assert_close(bounded_joint_cost(base, output['residual'], output['legal']), base, rtol=0, atol=0)
    legal = output['legal'].clone()
    legal[0, 0] = False
    changed = bounded_joint_cost(base, torch.full_like(base, 1000.), legal, .03)
    assert (changed - base).abs().max() <= .030001
    assert changed[0, 0] == base[0, 0]


def online_fixture():
    class Predictor:
        def __call__(self, history, times, valid, frame):
            assert (times[valid] < frame[:, None].expand_as(times)[valid]).all()
            return torch.nn.functional.normalize(history[:, 0], dim=-1), valid.any(1)

    class Tracker:
        def __init__(self):
            self.history_view = 'anchor_recent'
            self.predictor = Predictor()
            self.frame_id = 2
            self.args = SimpleNamespace(track_thresh=.58)

        def association_cost(self, tracks, detections):
            return np.full((len(tracks), len(detections)), .25, np.float64)

    feature = np.ones(8, np.float32) / np.sqrt(8)
    tracks = [SimpleNamespace(is_activated=True, initial_identity_rgb=feature, initial_identity_ir=None,
        initial_identity_frame=1, identity_history=[(1, feature, None)], start_frame=1, end_frame=1,
        state=1, tlbr=np.array([0., 0., 10., 10.]))]
    detections = [SimpleNamespace(score=.8, current_rgb=feature, tlbr=np.array([0., 0., 10., 10.]))]
    return Tracker(), tracks, detections


def test_online_zero_residual_exact_parity_and_low_stage_unmodified():
    tracker, tracks, detections = online_fixture()
    tracker = install_joint_identity(tracker, model())
    np.testing.assert_array_equal(tracker.association_cost(tracks, detections), np.array([[.25]]))
    assert tracker.joint_counts['scored_frames'] == 1
    detections[0].score = .5
    np.testing.assert_array_equal(tracker.association_cost(tracks, detections), np.array([[.25]]))
    assert tracker.joint_counts['scored_frames'] == 1


def test_uniform_residual_control_is_only_the_fixed_cost_threshold_effect():
    tracker, tracks, detections = online_fixture()
    tracker = install_joint_identity(tracker, UniformPositiveResidual(dimension=8, capacity=8))
    expected = np.array([[.25 + float(torch.tensor(.03))]])
    np.testing.assert_array_equal(tracker.association_cost(tracks, detections), expected)
    # Different current visual identity cannot change this untrained control.
    detections[0].current_rgb *= -1
    np.testing.assert_array_equal(tracker.association_cost(tracks, detections), expected)


def test_online_full_null_prevents_high_stage_old_match_without_fabricated_birth():
    class NullHead(JointFishIDDecoder):
        def forward(self, **kwargs):
            assert set(kwargs) == {'history', 'times', 'valid', 'frame', 'detections', 'detection_valid',
                                   'geometry', 'appearance', 'track_state', 'confidence', 'codes'}
            m, n = len(kwargs['detections']), len(kwargs['history'])
            return dict(logits=torch.cat([torch.full((m, n), -10.), torch.zeros(m, 1)], dim=1))
    tracker, tracks, detections = online_fixture()
    tracker = install_joint_identity(tracker, NullHead(dimension=8, hidden=16, capacity=8), route='full')
    assert tracker.association_cost(tracks, detections)[0, 0] > .97
    assert tracker.joint_counts['null_detections'] == 1
