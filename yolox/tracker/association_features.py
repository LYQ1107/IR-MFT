"""Causal, annotation-free pair features shared by Train and replay."""
import numpy as np


def cosine(first,second):
    first=np.asarray(first,np.float32);second=np.asarray(second,np.float32)
    first=first/np.maximum(np.linalg.norm(first,axis=1,keepdims=True),1e-8)
    second=second/np.maximum(np.linalg.norm(second,axis=1,keepdims=True),1e-8)
    return np.clip(first@second.T,-1,1)


def vote_features(matrix):
    matrix=np.asarray(matrix,np.float32);n,m=matrix.shape
    if not n or not m:return np.zeros((n,m,5),np.float32)
    row=np.max(matrix,axis=1,keepdims=True);column=np.max(matrix,axis=0,keepdims=True)
    row_other=np.broadcast_to(row,(n,m)).copy();column_other=np.broadcast_to(column,(n,m)).copy()
    row_other[np.arange(n),matrix.argmax(1)]=np.partition(matrix,-2,axis=1)[:,-2] if m>1 else 0
    column_other[matrix.argmax(0),np.arange(m)]=np.partition(matrix,-2,axis=0)[-2] if n>1 else 0
    return np.stack([matrix,matrix/np.maximum(row,1e-8),matrix-row_other,
                     matrix/np.maximum(column,1e-8),matrix-column_other],axis=-1)


def pair_features(past,current,matrices,good,total,lag,cue):
    """Return base23, cue17 and RGB-reference17, never accessing labels.

    Matrices include every candidate in the past/current frame, so each edge
    includes competition from alternative ownership hypotheses. RGB controls
    require no IR descriptor, IR matrix or auxiliary-quality value.
    """
    if cue not in {'rgb','ir'}:raise ValueError('Unknown cue')
    a=np.asarray(past['boxes'],np.float32);b=np.asarray(current['boxes'],np.float32);n,m=len(a),len(b)
    if not n or not m:return (np.zeros((n,m,23),np.float32),np.zeros((n,m,17),np.float32),np.zeros((n,m,17),np.float32))
    wa=np.maximum(a[:,2:]-a[:,:2],1);wb=np.maximum(b[:,2:]-b[:,:2],1)
    centers_a=(a[:,:2]+a[:,2:])/2;centers_b=(b[:,:2]+b[:,2:])/2
    relative=(centers_b[None]-centers_a[:,None])/np.maximum((wa[:,None]+wb[None])/2,1)
    size=np.log(wb[None]/wa[:,None]).clip(-3,3)
    overlap=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0)
    area=overlap.prod(-1);union=wa.prod(-1)[:,None]+wb.prod(-1)[None]-area
    geometry=np.stack([area/np.maximum(union,1),relative[:,:,0].clip(-10,10),relative[:,:,1].clip(-10,10),size[:,:,0],size[:,:,1],np.full((n,m),lag/8)],axis=-1)
    def broadcast_source(value):return np.broadcast_to(np.asarray(value,np.float32)[:,None],(n,m))
    def broadcast_target(value):return np.broadcast_to(np.asarray(value,np.float32)[None],(n,m))
    similarities=np.stack([cosine(past[k],current[k]) for k in ['embedding','rgb_mask_embedding','color','rgb_hog']],axis=-1)
    qualities=np.stack([broadcast_source(past['quality'][:,0]),broadcast_target(current['quality'][:,0]),
                        broadcast_source(past['point_quality'][:,0]),broadcast_target(current['point_quality'][:,0])],axis=-1)
    rgb_vote=vote_features(matrices[('rgb','rgb')])
    coverage=np.stack([broadcast_source(good['rgb']/np.maximum(total['rgb'],1)),broadcast_source(good['rgb']/20)],axis=-1)
    counts=np.broadcast_to(np.array([np.log1p(n)/4,np.log1p(m)/4],np.float32),(n,m,2))
    base=np.concatenate([geometry,similarities,qualities,rgb_vote,coverage,counts],axis=-1).astype(np.float32)
    def extra(kind):
        mode={'reference':'rgb','rgb':'bbox','ir':'joint'}[kind]
        embedding='ir_mask_embedding' if kind=='ir' else 'embedding' if kind=='rgb' else 'rgb_mask_embedding'
        shape='ir_hog' if kind=='ir' else 'rgb_hog';quality=1 if kind=='ir' else 0
        parts=[cosine(past[embedding],current[embedding])[:,:,None],cosine(past[shape],current[shape])[:,:,None],
               np.stack([broadcast_source(past['quality'][:,quality]),broadcast_target(current['quality'][:,quality])],axis=-1),
               vote_features(matrices[('rgb',mode)]),vote_features(matrices[(mode,'rgb')]),
               broadcast_source(good[mode]/np.maximum(total[mode],1))[:,:,None],
               np.stack([broadcast_source(past['point_quality'][:,quality]),broadcast_target(current['point_quality'][:,quality])],axis=-1)]
        return np.concatenate(parts,axis=-1).astype(np.float32)
    reference=extra('reference');additional=extra(cue)
    if base.shape!=(n,m,23) or additional.shape!=(n,m,17) or reference.shape!=(n,m,17):raise ValueError('Pair feature dimensions')
    if not all(np.isfinite(v).all() for v in [base,additional,reference]):raise FloatingPointError('Nonfinite pair evidence')
    return base,additional,reference
