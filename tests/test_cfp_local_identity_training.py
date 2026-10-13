import numpy as np
import torch

from cfp.local_identity_training import SameFrameTemporalSampler, local_identity_objective


def test_temporal_sampler_preserves_real_same_frame_and_video_identity_namespaces():
    records = [dict(identity=identity, frame=frame, view='gt') for frame in (1, 41, 81) for identity in (1, 2)]
    banks = {video: dict(records=records, arrays={'valid': np.ones(len(records), bool)}) for video in ('BT-001', 'PF-001')}
    sampler = SameFrameTemporalSampler(banks, list(banks), seed=4)
    assert sampler.labels[('BT-001', 1)] != sampler.labels[('PF-001', 1)]
    observations = sampler.sample()
    anchors = observations[::4]
    assert len(anchors) == 2
    assert len({video for video, _ in observations}) == 1
    assert len({banks[video]['records'][i]['frame'] for video, i in anchors}) == 1
    for group in (observations[:4], observations[4:]):
        video, first = group[0]
        anchor = banks[video]['records'][first]
        assert all(banks[v]['records'][i]['identity'] == anchor['identity'] for v, i in group)
        assert all(abs(banks[v]['records'][i]['frame'] - anchor['frame']) >= 25 for v, i in group[1:])


def test_same_frame_identity_loss_penalizes_confusable_fish_with_real_temporal_positives():
    embedding = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
    output = dict(embedding=embedding, logits=torch.zeros(4, 2, requires_grad=True))
    loss, metrics = local_identity_objective(output, torch.tensor([0, 0, 1, 1]),
        torch.tensor([1, 41, 1, 41]), torch.zeros(4, dtype=torch.int64))
    assert metrics['same_frame_anchors'] == 4 and metrics['same_frame'] > 1
    loss.backward()
    assert torch.isfinite(embedding.grad).all()
