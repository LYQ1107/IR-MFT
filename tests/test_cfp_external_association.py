import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('cfp_external_cli', Path(__file__).parents[1] / 'tools/cfp_replay_external_association.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_provider_preserves_native_confidence_filter_and_feature_order():
    provider = module.PrecomputedFishInterface()
    detections = np.array([[0, 0, 2, 2, .8], [2, 0, 4, 2, .1], [4, 0, 6, 2, .58], [6, 0, 8, 2, .7]])
    features = np.eye(4)
    provider.set_frame(detections, features, .58, .1)
    result = provider.inference(None, detections[[0, 3], :4])
    assert np.array_equal(result, features[[0, 3]])
    result[:] = 0
    assert np.array_equal(provider.features, features[[0, 3]])
    with pytest.raises(ValueError):
        provider.inference(None, detections[[3, 0], :4])


def test_scalar_adapter_rejects_multivalue_or_nonfinite_scores():
    assert module.scalar_score(np.array([.75])) == .75
    assert module.scalar_score(.5) == .5
    for invalid in (np.array([1, 2]), np.array([np.nan])):
        with pytest.raises(ValueError):
            module.scalar_score(invalid)
