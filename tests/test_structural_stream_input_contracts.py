import numpy as np
import pytest
import torch
from tools.structural_stream_training import StreamingTrainInputs


def _provider(backend):
    obj=object.__new__(StreamingTrainInputs)
    obj.device=torch.device('cpu');obj.feature_backend=backend;obj.encoder=None
    return obj


def test_frozen_features_remain_usable_for_trainable_adapter_backward():
    feature=torch.ones(4,3,3)
    provider=_provider(lambda *args:(feature,feature,feature))
    outputs=provider._encode('Train',1,'ir',np.zeros((3,4,4),np.float16))
    layer=torch.nn.Conv2d(4,2,1)
    layer(outputs[0][None]).sum().backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert all(not v.requires_grad and not v.is_inference()for v in outputs)


def test_inference_tensors_are_rejected_before_adapter_training():
    with torch.inference_mode():feature=torch.ones(4,3,3)
    provider=_provider(lambda *args:(feature,feature,feature))
    with pytest.raises(ValueError,match='normal no-grad tensors'):
        provider._encode('Train',1,'ir',np.zeros((3,4,4),np.float16))


def test_frame_address_lookup_rejects_heldout_videos_and_outside_budget():
    provider=object.__new__(StreamingTrainInputs)
    provider.schedule={'Train':np.tile(np.arange(1,193),(24,1))}
    assert provider.addresses([('Train',0),('Train',191)],1)==[('Train',1),('Train',192)]
    for draw,epoch in [([('Dev',0)],1),([('Test',0)],1),([('Train',192)],1),([('Train',0)],25)]:
        with pytest.raises(ValueError):provider.addresses(draw,epoch)
