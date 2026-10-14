import numpy as np
import torch

from cfp.visible_token_native_encoder import CurrentVisibleTokenIdentityEncoder


class _Backbone:
    def eval(self):
        return self

    def requires_grad_(self, value):
        return self

    def to(self, device):
        return self

    def forward_features(self, images):
        batch = len(images)
        # 126x252 with patch14 gives a 9x18=162 spatial grid.
        values = torch.arange(batch * 384, dtype=torch.float32).reshape(batch, 384)
        tokens = torch.ones(batch, 162, 384, dtype=torch.float32)
        return {'x_norm_clstoken': values, 'x_norm_patchtokens': tokens}


def test_visible_token_native_encoder_preserves_current_rows_and_empty_proposals():
    encoder = CurrentVisibleTokenIdentityEncoder.__new__(CurrentVisibleTokenIdentityEncoder)
    encoder.backbone = _Backbone()
    encoder.device = 'cpu'
    encoder.batch_size = 128
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    boxes = np.array([[2, 3, 24, 21], [60, 60, 70, 70]], dtype=np.float32)
    output = encoder.encode(image, boxes)
    assert output['embedding'].shape == (2, 384)
    assert output['raw_cls'].shape == (2, 384)
    assert output['local4'].shape == (2, 4, 384)
    assert output['visibility4'].shape == (2, 4)
    assert output['valid'].tolist() == [True, False]
    assert np.all(output['embedding'][1] == 0)
    assert np.isfinite(output['embedding']).all()


def test_visible_token_native_encoder_zero_proposals_are_supported():
    encoder = CurrentVisibleTokenIdentityEncoder.__new__(CurrentVisibleTokenIdentityEncoder)
    encoder.backbone = _Backbone()
    encoder.device = 'cpu'
    encoder.batch_size = 128
    output = encoder.encode(np.zeros((16, 16, 3), dtype=np.uint8), np.empty((0, 4), dtype=np.float32))
    assert output['embedding'].shape == (0, 384)
    assert output['local4'].shape == (0, 4, 384)
    assert output['valid'].shape == (0,)
