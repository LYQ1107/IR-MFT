"""Physical cue mismatch and conservative RGB-posterior preservation."""
import numpy as np
import torch
from torch.nn import functional as F

from yolox.tracker.native_query_features import identity_pair_features


def permuted_current_packet(reference_past, reference_current, native_past, native_current, permutation):
    permutation=np.asarray(permutation,np.int64)
    if not np.array_equal(np.sort(permutation),np.arange(len(reference_current))):
        raise ValueError('Current cue permutation must preserve the candidate set')
    # Target agreement must be recomputed against its actual RGB location.
    # Rolling an already computed cue packet would keep a hidden correct pair.
    return identity_pair_features(reference_past,reference_current,native_past,native_current[permutation])[1]


def visible_mismatch(corrupt, reference, available, valid, dropped):
    """An additionally deleted long target cannot survive via its permutation."""
    cue=torch.where(dropped[:,None,None],reference,corrupt)
    enabled=available & valid & ~dropped[:,None]
    return cue,enabled


def preserve_RGB_posterior(real_logits, null_logits, delta, valid):
    real_logits=real_logits.detach();null_logits=null_logits.detach()
    fixed=torch.cat([real_logits.masked_fill(~valid,-1e9),null_logits[...,None]],-1)
    changed=torch.cat([(real_logits+delta).masked_fill(~valid,-1e9),null_logits[...,None]],-1)
    target=F.softmax(fixed.detach(),-1)
    return F.kl_div(F.log_softmax(changed,-1),target,reduction='batchmean')
