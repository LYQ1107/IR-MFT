"""Retain the first three reliable observations without updating them later."""
import numpy as np
from tools.mft_tracklet_permanence import unit


class IdentityAnchors:
    def __init__(self,minimum=3):
        assert type(minimum)is int and minimum>0;self.minimum=minimum;self.entries={}

    def add(self,identity,frame,rgb,ir,*,reliable):
        assert type(identity)is int and identity>0 and type(frame)is int and frame>0
        if not reliable:return
        old=self.entries.get(identity)
        if old is not None:
            assert frame>old['last_frame']
            if old['count']==self.minimum:return
        r=unit(rgb);a=None if ir is None else unit(ir)
        assert r is not None and(ir is None or a is not None)
        if old is None:
            old=dict(count=0,last_frame=0,RGB_sum=np.zeros_like(r),IR_sum=None if a is None else np.zeros_like(a),
                IR_all_available=a is not None);self.entries[identity]=old
        assert r.shape==old['RGB_sum'].shape
        old['count']+=1;old['last_frame']=frame;old['RGB_sum']+=r
        old['IR_all_available']=old['IR_all_available']and a is not None
        if old['IR_all_available']:
            assert a.shape==old['IR_sum'].shape;old['IR_sum']+=a
        if old['count']==self.minimum:
            old['RGB']=unit(old['RGB_sum']);old['IR']=unit(old['IR_sum'])if old['IR_all_available']else None
            assert old['RGB']is not None
            if old['IR_all_available']:assert old['IR']is not None

    def get(self,identity,current_frame):
        old=self.entries.get(identity)
        if old is None or old['count']<self.minimum:return None
        assert old['last_frame']<current_frame,'Current observation cannot become a past identity anchor'
        return dict(RGB=old['RGB'].copy(),IR=None if old['IR']is None else old['IR'].copy(),
            observations=old['count'],last_frame=old['last_frame'])

    def prune(self,active_ids):self.entries={k:v for k,v in self.entries.items()if k in active_ids}
