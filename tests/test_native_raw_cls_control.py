import numpy as np
import pytest

from cfp.native_raw_cls_control import source_matched_raw_cls


def test_raw_cls_control_requires_explicit_source_field():
    values = np.zeros((2, 384), dtype=np.float32)
    values[:, 0] = 1.0
    assert source_matched_raw_cls({"raw_cls": values}).shape == (2, 384)
    with pytest.raises(ValueError, match="explicit raw_cls"):
        source_matched_raw_cls({"embedding": values}, field="embedding")


def test_raw_cls_control_rejects_bad_coordinates():
    values = np.zeros((2, 384), dtype=np.float32)
    with pytest.raises(ValueError, match="nonzero"):
        source_matched_raw_cls({"raw_cls": values})
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        source_matched_raw_cls({"raw_cls": values})
