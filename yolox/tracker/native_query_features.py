"""Annotation-free identity competition and an explicit null alternative."""
import numpy as np

from yolox.tracker.association_features import cosine


def competition_margins(matrix):
    matrix=np.asarray(matrix,np.float32);n,m=matrix.shape
    if not n or not m:return np.zeros_like(matrix),np.zeros_like(matrix)
    other_row=np.broadcast_to(matrix.max(1,keepdims=True),(n,m)).copy()
    other_column=np.broadcast_to(matrix.max(0,keepdims=True),(n,m)).copy()
    other_row[np.arange(n),matrix.argmax(1)]=np.partition(matrix,-2,axis=1)[:,-2] if m>1 else 0
    other_column[matrix.argmax(0),np.arange(m)]=np.partition(matrix,-2,axis=0)[-2] if n>1 else 0
    return matrix-other_row,matrix-other_column


def self_agreement(first,second):
    first=np.asarray(first,np.float32);second=np.asarray(second,np.float32)
    denominator=np.linalg.norm(first,axis=-1)*np.linalg.norm(second,axis=-1)
    return np.clip((first*second).sum(-1)/np.maximum(denominator,1e-8),-1,1)


def identity_pair_features(reference_past,reference_current,native_past=None,native_current=None):
    """Return RGB3, native7 and reference7; contrast qualities are appended.

    Source competition includes all proposals in the same past frame. For a
    genuinely appearance-only single-source query the column alternative is0.
    Equal cue/reference vectors produce bit-identical feature packets.
    """
    n,m=len(reference_past),len(reference_current)
    rgb=cosine(reference_past,reference_current);row,column=competition_margins(rgb)
    extra=np.stack([rgb,row,column],axis=-1)
    source=self_agreement(reference_past,reference_past);target=self_agreement(reference_current,reference_current)
    def packet(similarity,cross_first,cross_second,source_quality,target_quality):
        r,c=competition_margins(similarity)
        return np.stack([similarity,cross_first,cross_second,np.broadcast_to(source_quality[:,None],(n,m)),np.broadcast_to(target_quality[None],(n,m)),r,c],axis=-1).astype(np.float32)
    reference=packet(rgb,rgb,rgb,source,target)
    if native_past is None and native_current is None:return extra,reference.copy(),reference
    if native_past is None or native_current is None:raise ValueError('Both cue observation times are required')
    native=packet(cosine(native_past,native_current),cosine(native_past,reference_current),cosine(reference_past,native_current),self_agreement(native_past,reference_past),self_agreement(native_current,reference_current))
    return extra,native,reference


def posterior_with_null(real_logits,null_logits,include_null=True):
    real=np.asarray(real_logits,np.float32);null=np.broadcast_to(np.asarray(null_logits,np.float32),real.shape[:-1])
    if not real.shape[-1]:return np.zeros_like(real),np.ones(real.shape[:-1],np.float32)
    if not include_null:null=np.full_like(null,-np.inf)
    maximum=np.maximum(real.max(-1),null);exponent=np.exp(real-maximum[...,None]);null_exponent=np.exp(null-maximum)
    denominator=exponent.sum(-1)+null_exponent
    return exponent/denominator[...,None],null_exponent/denominator
