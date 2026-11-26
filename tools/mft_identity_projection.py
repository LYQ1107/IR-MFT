"""Train-only regularized Fisher descriptors; no file, GT or device access."""
import numpy as np


def unit_rows(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite descriptor matrix required')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 1e-12)


def paired_features(rgb, infrared):
    if np.shape(rgb) != np.shape(infrared):
        raise ValueError('Paired current RGB/actual IR ROI vectors required')
    return np.concatenate((unit_rows(rgb), unit_rows(infrared)), axis=1) / np.sqrt(2.)


def fit_fisher(features, videos, identities, dimensions=64, ridge_fraction=.01):
    """Equal video/identity weights; between-ID scatter is centered within video.

    Train IDs are local to their video. The resulting shared transform uses no
    video ID, identity label, future frame or GT at inference.
    """
    x = np.asarray(features, dtype=np.float64)
    videos, identities = np.asarray(videos), np.asarray(identities)
    if (x.ndim != 2 or len(x) != len(videos) or len(x) != len(identities)
            or not np.isfinite(x).all() or dimensions < 1 or ridge_fraction <= 0):
        raise ValueError('Valid finite Train observations and positive shrinkage required')
    within = np.zeros((x.shape[1], x.shape[1]), np.float64)
    between = np.zeros_like(within)
    center = np.zeros(x.shape[1], np.float64)
    counts = []
    unique_videos = np.unique(videos)
    for video in unique_videos:
        labels = np.unique(identities[videos == video])
        groups = [x[(videos == video) & (identities == label)] for label in labels]
        if not len(groups) or any(len(g) < 2 for g in groups):
            raise ValueError('Every fitted identity needs at least two Train observations')
        means = np.array([g.mean(axis=0) for g in groups])
        video_center = means.mean(axis=0)
        center += video_center / len(unique_videos)
        weight = 1. / (len(unique_videos) * len(groups))
        for group, mean in zip(groups, means):
            residual = group - mean
            within += weight * (residual.T @ residual) / len(group)
            delta = mean - video_center
            between += weight * np.outer(delta, delta)
        counts.append(dict(video=int(video), identities=len(groups), observations=sum(map(len, groups))))
    ridge = max(ridge_fraction * np.trace(within) / x.shape[1], 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(within + ridge * np.eye(x.shape[1]))
    whitening = eigenvectors / np.sqrt(eigenvalues)[None, :]
    discriminant = whitening.T @ between @ whitening
    values, vectors = np.linalg.eigh((discriminant + discriminant.T) / 2)
    order = np.argsort(values)[::-1]
    order = order[values[order] > 1e-9][:dimensions]
    if not len(order):
        raise ValueError('No positive identity-discriminating direction')
    projection = whitening @ vectors[:, order]
    # Deterministic signs remove the irrelevant LAPACK eigenvector ambiguity.
    signs = np.sign(projection[np.abs(projection).argmax(axis=0), np.arange(len(order))])
    projection *= np.where(signs == 0, 1, signs)[None, :]
    return dict(center=center.astype(np.float32), projection=projection.astype(np.float32),
                eigenvalues=values[order].astype(np.float64)), dict(
        dimensions=len(order), requested_dimensions=dimensions, ridge_fraction=ridge_fraction,
        absolute_ridge=float(ridge), input_dimensions=x.shape[1], observations=len(x),
        video_counts=counts, within_trace=float(np.trace(within)), between_trace=float(np.trace(between)),
        video_local_identity_classes=sum(row['identities'] for row in counts),
        inference_uses_video_or_identity_labels=False)


def transform(features, parameters):
    x = np.asarray(features, dtype=np.float32)
    center, projection = parameters['center'], parameters['projection']
    if x.ndim != 2 or x.shape[1] != len(center) or projection.shape[0] != len(center):
        raise ValueError('Fitted descriptor dimensions changed')
    return unit_rows((x - center) @ projection).astype(np.float32)


def temporal_retrieval(features, videos, identities, frames, fraction=.7):
    """Within each Train video, early identity templates retrieve late queries."""
    features = unit_rows(features)
    rows = []
    for video in np.unique(videos):
        labels = np.unique(identities[videos == video])
        gallery, queries, truth = [], [], []
        for label in labels:
            indices = np.flatnonzero((videos == video) & (identities == label))
            indices = indices[np.argsort(frames[indices], kind='stable')]
            split = max(1, min(len(indices)-1, int(len(indices)*fraction)))
            gallery.append(features[indices[:split]].mean(axis=0))
            queries.extend(indices[split:])
            truth.extend([len(gallery)-1]*(len(indices)-split))
        scores = features[queries] @ unit_rows(gallery).T
        correct = int((scores.argmax(axis=1) == np.asarray(truth)).sum())
        rows.append(dict(video=int(video), queries=len(queries), correct=correct,
                         top1_percent=100.*correct/max(len(queries), 1)))
    return dict(rows=rows, queries=sum(r['queries'] for r in rows),
                correct=sum(r['correct'] for r in rows),
                equal_video_top1_percent=float(np.mean([r['top1_percent'] for r in rows])))
