"""Small 2D MIND-inspired self-similarity diagnostic, no learned parameters."""
import cv2
import numpy as np


OFFSETS = [(0, -2), (0, 2), (-2, 0), (2, 0), (-2, -2), (-2, 2), (2, -2), (2, 2)]


def shifted(image, dy, dx):
    """At each source location sample the reflected image at (y+dy,x+dx)."""
    h, w = image.shape[:2]
    pad = max(abs(dy), abs(dx))
    if not pad:
        return image.copy()
    pads = [(pad, pad), (pad, pad)] + [(0, 0)]*(image.ndim-2)
    value = np.pad(image, pads, mode='reflect')
    return value[pad+dy:pad+dy+h, pad+dx:pad+dx+w]


def descriptor(image):
    """Eight same-image patch SSDs; four-neighbor variance and max normalization.

    A diagnostic 2D neighborhood choice inspired by Heinrich et al. (2012),
    not the full published registration pipeline or a newly claimed descriptor.
    """
    image = np.asarray(image, np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('A finite scalar image is required')
    distances = np.stack([cv2.boxFilter((image-shifted(image, dy, dx))**2, -1,
                    (3, 3), normalize=True, borderType=cv2.BORDER_REFLECT_101)
                    for dy, dx in OFFSETS], -1)
    variance = distances[:, :, :4].mean(-1)
    denominator = np.maximum(variance, max(float(variance.mean())*.001, 1e-12))
    distances -= distances.min(-1, keepdims=True)
    value = np.exp(-distances/denominator[:, :, None]).astype(np.float32)
    return value, variance


def match(first, second, radius=4):
    """Independent local search; returns forward offsets and inverse consistency.

    Positive offsets mean the corresponding second-view pixel is at source+offset.
    All image-wide matching precedes any foreground-label inspection.
    """
    a, av = descriptor(first); b, bv = descriptor(second)
    if a.shape != b.shape:
        raise ValueError('Paired image dimensions differ')
    offsets = sorted([(dy, dx) for dy in range(-radius, radius+1)
                      for dx in range(-radius, radius+1)], key=lambda v: (v[0]**2+v[1]**2, v))

    def search(query, candidates):
        cost = np.full(query.shape[:2], np.inf, np.float32)
        field = np.zeros((*cost.shape, 2), np.int8)
        for dy, dx in offsets:
            distance = np.mean(abs(query-shifted(candidates, dy, dx)), -1)
            update = distance < cost
            cost[update] = distance[update]
            field[update] = (dy, dx)
        return cost, field

    cost, forward = search(a, b); _, backward = search(b, a)
    yy, xx = np.indices(cost.shape)
    ty = np.clip(yy+forward[:, :, 0], 0, cost.shape[0]-1)
    tx = np.clip(xx+forward[:, :, 1], 0, cost.shape[1]-1)
    inverse = backward[ty, tx].astype(np.float32)
    cycle_error = np.linalg.norm(forward.astype(np.float32)+inverse, axis=-1)
    interior = ((yy >= radius+3) & (xx >= radius+3) &
                (yy < cost.shape[0]-radius-3) & (xx < cost.shape[1]-radius-3))
    textured = (av > 1e-6) & (bv[ty, tx] > 1e-6) & interior
    return dict(cost=cost, offset=forward, cycle_error=cycle_error, textured=textured,
                inverse_consistent=textured & (cycle_error <= 1.),
                zero_offset_cost=np.mean(abs(a-b), -1))
