"""Explicit frozen latent views, without new image/feature-bank copies."""
import numpy as np


REPRESENTATIONS = ('rgb', 'ir', 'rgb_ir', 'two_rgb')


def identity_view(rgb, ir, representation='rgb', second_rgb=None):
    if representation not in REPRESENTATIONS:
        raise ValueError('Undeclared latent representation')
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 2 or rgb.shape[1] != 512 or not np.isfinite(rgb).all():
        raise ValueError('Finite frozen 512-dimensional RGB bank required')
    if representation == 'rgb':
        return rgb
    other = second_rgb if representation == 'two_rgb' else ir
    if other is None or np.shape(other) != rgb.shape or not np.isfinite(other).all():
        raise ValueError('Aligned frozen auxiliary coordinate space required')
    other = np.asarray(other, dtype=np.float32)
    if representation == 'ir':
        return other
    joined = np.concatenate([rgb, other], axis=1)
    norms = np.linalg.norm(joined, axis=1, keepdims=True)
    return joined / np.maximum(norms, 1e-8)
