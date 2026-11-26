"""Explicit FP16 descriptor storage and a process-safe hard payload budget."""
import fcntl
import json
import os
from pathlib import Path
import re
import numpy as np


def payload_bytes(proposals,frames):
    assert type(proposals)is int and type(frames)is int and proposals>=0 and frames>=0
    return proposals*(5*4+1024*2)+(frames*2+1)*8


def quantize_frame(detections,embeddings):
    d=np.asarray(detections)
    assert d.dtype==np.float32 and d.ndim==2 and d.shape[1]==5 and np.isfinite(d).all()
    assert((d[:,4]>=0)&(d[:,4]<=1)).all()and(d[:,2:4]>d[:,:2]).all()
    packed=dict(detections=d.copy());errors={}
    for cue in ['rgb','ir']:
        values=np.asarray(embeddings[cue])
        assert values.dtype==np.float32 and values.shape==(len(d),512)and np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values,axis=1),1,atol=2e-5)
        compact=values.astype(np.float16);restored=compact.astype(np.float32)
        norm_error=float(np.max(np.abs(np.linalg.norm(restored,axis=1)-1)))if len(d)else 0.
        maximum=float(np.max(np.abs(restored-values)))if len(d)else 0.
        assert norm_error<=1e-3 and maximum<=1e-3
        packed[cue]=compact
        errors[cue]=dict(maximum_absolute_error=maximum,maximum_unit_norm_error=norm_error)
    return packed,errors


class PayloadBudget:
    def __init__(self,directory,maximum_bytes):
        assert type(maximum_bytes)is int and maximum_bytes>0
        self.base=Path(directory);self.base.mkdir(parents=True,exist_ok=True)
        self.maximum=maximum_bytes
        self.path=self.base/'payload_reservations.json'

    def reserve(self,sequence,amount):
        assert re.fullmatch(r'[A-Za-z0-9_-]+',sequence)and type(amount)is int and amount>=0
        with(self.base/'payload_reservations.lock').open('a+')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=json.loads(self.path.read_text())if self.path.exists()else dict(maximum_bytes=self.maximum,sequences={})
            assert state['maximum_bytes']==self.maximum
            previous=state['sequences'].get(sequence,0)
            assert amount>=previous,'Payload reservations cannot silently shrink or overwrite prior work'
            total=sum(state['sequences'].values())-previous+amount
            if total>self.maximum:
                raise RuntimeError('Preserve attempted writer: global Train payload would exceed its hard budget')
            state['sequences'][sequence]=amount;state['total_reserved_bytes']=total
            temporary=self.base/('payload_reservations.'+str(os.getpid())+'.tmp.json')
            temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,self.path)
            return state
