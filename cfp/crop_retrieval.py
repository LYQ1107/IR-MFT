"""Offline physical-label evaluation; never used by online association."""
import numpy as np
import torch

from .identity_evidence import query_retrieval


def embedding_crop_retrieval(embedding, records, valid, device='cpu', query_cap=64, gaps=(1, 25, 100)):
    if embedding.shape[0] != len(records) or valid.shape != (len(records),) or not np.isfinite(embedding).all():
        raise ValueError('Finite aligned per-current-crop embeddings required')
    embedding = np.array(embedding, copy=True)
    embedding[~valid] = 0
    result = {}
    for view in ('gt', 'detector'):
        indices = np.asarray([i for i, row in enumerate(records) if row['view'] == view], dtype=np.int64)
        times = np.asarray([records[i]['frame'] for i in indices])
        labels = np.asarray([records[i]['identity'] for i in indices])
        available = valid[indices]
        queries = []
        for identity in np.unique(labels[labels >= 0]):
            observed = np.flatnonzero((labels == identity) & available)
            if len(observed):
                selected = np.unique(np.linspace(0, len(observed) - 1, min(len(observed), query_cap), dtype=int))
                queries.extend(observed[selected].tolist())
        queries.sort()
        values = torch.from_numpy(embedding[indices]).to(device)
        rows = {str(gap): [] for gap in gaps}
        with torch.inference_mode():
            for start in range(0, len(queries), 64):
                chosen = queries[start:start + 64]
                scores = (values[chosen] @ values.T).cpu().numpy()
                for j, query in enumerate(chosen):
                    labels_for_query = np.where(labels < 0, -1, (labels != labels[query]).astype(np.int8)).astype(np.int8)
                    for gap in gaps:
                        gallery = times <= times[query] - gap
                        rows[str(gap)].append(query_retrieval(scores[j, gallery], labels_for_query[gallery]))
        result[view] = {}
        for gap, entries in rows.items():
            supported = [r for r in entries if r['target_present']]
            result[view][gap] = dict(queries=len(entries), supported_queries=len(supported), missing_target=len(entries) - len(supported),
                rank1_percent=100 * np.mean([r['rank1'] for r in supported]) if supported else None,
                rank5_percent=100 * np.mean([r['rank5'] for r in supported]) if supported else None,
                mAP_percent=100 * np.mean([r['average_precision'] for r in supported]) if supported else None,
                unknown_selected=sum(r['unknown_selected'] for r in supported))
    return result
