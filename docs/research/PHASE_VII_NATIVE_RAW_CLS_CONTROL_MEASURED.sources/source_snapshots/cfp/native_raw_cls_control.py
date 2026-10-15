"""Validation helpers for the source-matched raw-CLS native control."""

import numpy as np


def source_matched_raw_cls(bank, field="raw_cls"):
    """Return the declared normalized raw CLS coordinate without aliasing a head output."""
    if field != "raw_cls":
        raise ValueError("The registered control must use the explicit raw_cls field")
    if field not in bank:
        raise ValueError("Native bank is missing raw_cls")
    values = np.asarray(bank[field])
    if values.ndim != 2 or values.shape[1] != 384 or not np.isfinite(values).all():
        raise ValueError("raw_cls must be a finite N x 384 native coordinate")
    norms = np.linalg.norm(values, axis=1)
    if len(values) and (not np.isfinite(norms).all() or np.any(norms <= 1e-8)):
        raise ValueError("raw_cls must contain nonzero normalized coordinates")
    return values
