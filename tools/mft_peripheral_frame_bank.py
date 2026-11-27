"""Compact GT-free observation replay; no image/descriptor or label bank.

Only immutable current-frame snapshots are delivered to a publisher. Geometry
and confidence values retain float64 precision; IDs and boundaries are int64.
"""
from types import MappingProxyType
import numpy as np
from tools.mft_protected_peripheral import _readonly
from tools.mft_scene_coverage import KINDS

MODES=('original','expanded','expanded_without_explanation')
MAIN=np.dtype([('frame','<i8'),('identity','<i8'),('proposal','<i8'),('tlwh','<f8',(4,)),('score','<f8')])
SHADOW=np.dtype([('identity','<i8'),('proposal','<i8'),('streak','<i8'),('score','<f8'),
    ('xyxy','<f8',(4,)),('tlwh','<f8',(4,)),('explained','<f8')])


class FrameBankBuilder:
    def __init__(self):
        self.main=[];self.shadow=[];self.assigned=[];self.needs=[]
        self.main_offsets=[0];self.shadow_offsets=[0];self.assigned_offsets=[0];self.frames=0

    def append(self,value):
        assert value['frame']==self.frames+1
        for frame,box,identity,score in value['main_rows']:
            self.main.append((frame,identity,value['main_proposals'][identity],box.copy(),score))
        for row in value['shadow_records']:
            self.shadow.append((row['shadow_id'],row['proposal_index'],row['streak'],row['score'],
                row['box_xyxy'].copy(),row['tlwh'].copy(),value['explained'][row['proposal_index']]))
        self.assigned.extend(sorted(value['main_assigned']))
        self.needs.append([[value['needs'][mode][kind]for kind in KINDS]for mode in MODES])
        for offsets,data in [(self.main_offsets,self.main),(self.shadow_offsets,self.shadow),(self.assigned_offsets,self.assigned)]:offsets.append(len(data))
        self.frames+=1

    def arrays(self):
        a=dict(main=np.asarray(self.main,dtype=MAIN),shadow=np.asarray(self.shadow,dtype=SHADOW),
            assigned=np.asarray(self.assigned,dtype='<i8'),needs=np.asarray(self.needs,dtype='<f8'),
            main_offsets=np.asarray(self.main_offsets,dtype='<i8'),shadow_offsets=np.asarray(self.shadow_offsets,dtype='<i8'),
            assigned_offsets=np.asarray(self.assigned_offsets,dtype='<i8'),frame_ids=np.arange(1,self.frames+1,dtype='<i8'))
        validate(a,self.frames);return a


def validate(a,frames):
    assert set(a)=={'main','shadow','assigned','needs','main_offsets','shadow_offsets','assigned_offsets','frame_ids'}
    assert a['main'].dtype==MAIN and a['shadow'].dtype==SHADOW and a['assigned'].dtype==np.int64
    assert a['needs'].dtype==np.float64 and a['needs'].shape==(frames,3,len(KINDS))
    assert a['frame_ids'].dtype==np.int64 and a['frame_ids'].shape==(frames,)
    assert np.isfinite(a['needs']).all()and ((a['needs']>=0)&(a['needs']<=1)).all()
    np.testing.assert_array_equal(a['frame_ids'],np.arange(1,frames+1))
    for kind in ['main','shadow','assigned']:
        offsets=a[kind+'_offsets'];assert offsets.dtype==np.int64 and offsets.shape==(frames+1,)
        assert offsets[0]==0 and offsets[-1]==len(a[kind])and (np.diff(offsets)>=0).all()
    for kind in ['main','shadow']:
        for field in a[kind].dtype.names:assert np.isfinite(a[kind][field]).all()
        assert (a[kind]['identity']>0).all()and (a[kind]['proposal']>=0).all()
        assert ((a[kind]['score']>=0)&(a[kind]['score']<=1)).all()
        assert (a[kind]['tlwh'][:,2:]>0).all()
    assert (a['shadow']['streak']>=1).all()
    assert ((a['shadow']['explained']>=0)&(a['shadow']['explained']<=1)).all()
    np.testing.assert_array_equal(a['needs'][:,:,KINDS.index('RGB')],a['needs'][:,:,KINDS.index('IR_missing')])


class SequentialFrameBank:
    def __init__(self,sequence,arrays):
        self.sequence,self.arrays=sequence,arrays;self.frames=len(arrays['frame_ids']);self.frame=0
        validate(arrays,self.frames)

    def next(self,frame):
        assert frame==self.frame+1 and frame<=self.frames
        a=self.arrays;i=frame-1;ml,mh=map(int,a['main_offsets'][i:i+2]);sl,sh=map(int,a['shadow_offsets'][i:i+2]);al,ah=map(int,a['assigned_offsets'][i:i+2])
        main=a['main'][ml:mh];shadow=a['shadow'][sl:sh];assigned=a['assigned'][al:ah]
        assert (main['frame']==frame).all()and len(np.unique(main['identity']))==len(main)
        assert len(np.unique(shadow['proposal']))==len(shadow)and len(np.unique(shadow['identity']))==len(shadow)
        assert len(np.unique(assigned))==len(assigned)
        maximum=max([*map(int,assigned),*map(int,shadow['proposal']),-1])+1
        explained=np.zeros(maximum,np.float64)
        for r in shadow:explained[int(r['proposal'])]=r['explained']
        records=tuple(MappingProxyType(dict(shadow_id=int(r['identity']),proposal_index=int(r['proposal']),streak=int(r['streak']),
            score=float(r['score']),box_xyxy=_readonly(r['xyxy']),tlwh=_readonly(r['tlwh'])))for r in shadow)
        value=MappingProxyType(dict(sequence=self.sequence,frame=frame,
            main_rows=tuple((frame,_readonly(r['tlwh']),int(r['identity']),float(r['score']))for r in main),
            main_proposals=MappingProxyType({int(r['identity']):int(r['proposal'])for r in main}),
            main_assigned=frozenset(map(int,assigned)),shadow_records=records,explained=_readonly(explained),
            needs=MappingProxyType({mode:MappingProxyType({k:float(a['needs'][i,j,h])for h,k in enumerate(KINDS)})for j,mode in enumerate(MODES)})))
        self.frame=frame;return value
