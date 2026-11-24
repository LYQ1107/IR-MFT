import torch

from tools.spatial_native_fusion import SpatialNativeFusion
from tools.spatial_native_observer import SpatialObserver


def test_read_only_observer_records_spatial_cells_and_missing_without_changing_output():
    torch.manual_seed(0)
    module = SpatialNativeFusion(channels=(8, 16, 32), use_redundancy=True).eval()
    rgb = tuple(torch.randn(1, c, 64//s, 96//s) for c, s in zip((8, 16, 32), (8, 16, 32)))
    cue = torch.randn(1, 3, 64, 96)
    scores = tuple(torch.full_like(v[:, :1], .2) for v in rgb)
    valid = torch.ones_like(cue[:, :1], dtype=torch.bool)
    valid[:, :, :, 64:] = False
    expected = tuple(v.detach().clone() for v in module(rgb, cue, scores, valid=valid)[0])
    observer = SpatialObserver()
    observer.attach('candidate', module)
    actual = module(rgb, cue, scores, valid=valid)[0]
    module(rgb, None, scores, available=False)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    record = observer.summary('candidate')
    assert (record['frames'], record['available_frames'], record['missing_frames']) == (2, 1, 1)
    for row in record['levels']:
        assert row['gate']['count'] > 0 and row['frame_quality']['count'] == 1
        assert row['feature_changed']['mean'] == 0 and row['redundancy']['mean'] > 0
    observer.close()
    module(rgb, cue, scores, valid=valid)
    assert observer.summary('candidate')['frames'] == 2
