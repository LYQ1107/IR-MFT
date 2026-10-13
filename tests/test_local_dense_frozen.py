import pytest
import torch

from cfp.local_dense_frozen_controls import FrozenSceneControl


@pytest.mark.parametrize('mode', ('cls_last', 'cls_fifo', 'local_corresponding', 'local_flexible'))
def test_frozen_controls_no_params_future_rejected_and_unknown_regions_excluded(mode):
    model = FrozenSceneControl(mode)
    assert not list(model.parameters())
    x = dict(history=torch.randn(2, 3, 4, 12), history_fraction=torch.ones(2, 3, 4),
        history_global=torch.randn(2, 3, 12), times=torch.tensor([[1, 3, 5], [1, 3, 5]]),
        history_valid=torch.ones(2, 3, dtype=torch.bool), frame=10,
        candidates=torch.randn(3, 4, 12), candidate_fraction=torch.ones(3, 4),
        candidate_global=torch.randn(3, 12))
    with pytest.raises(ValueError):
        model(**dict(x, frame=5))
    x['history_valid'][:, 0] = False
    x['times'][:, 0] = 0
    x['history_fraction'][:, :, 0] = 0
    x['candidate_fraction'][:, 0] = 0
    original = model(**x)
    changed = {n: v.clone() if isinstance(v, torch.Tensor) else v for n, v in x.items()}
    changed['history_global'][:, 0] = 1.e6
    changed['history'][:, :, 0] = 1.e6
    changed['history'][:, 0] = 1.e6
    changed['candidates'][:, 0] = 1.e6
    torch.testing.assert_close(model(**changed)['scores'], original['scores'], rtol=0, atol=0)
