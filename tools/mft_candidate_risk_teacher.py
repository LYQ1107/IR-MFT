"""Causal original-association teacher and GT-free proposal/template interface.

An optional observer receives the current primary candidate graph before LAP.
Its return value is ignored; the original costs always enter the teacher LAP.
Training label histories belong to the caller, never to tracks or features.
"""
from types import SimpleNamespace
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker,STrack
from yolox.tracker import matching
from tools.mft_tracklet_permanence import unit
from tools.mft_candidate_risk_features import candidate_features


class CandidateTrack(STrack):
    def __init__(self,box,score,observation,counter):
        super().__init__(box,score)
        self.observation,self.counter=observation,counter
        self.rgb,self.ir,self.last_box=None,None,None
        self.accepted=0;self.memory_frame=None;self.memory_quality=0.
        self.proposal_index=None;self.measurement_frame=None
        self.current_measurement=self._tlwh.copy()

    def next_id(self):
        self.counter[0]+=1
        return self.counter[0]

    def accept(self,observation):
        assert observation is not None
        self.proposal_index=int(observation['proposal_index'])
        self.measurement_frame=self.frame_id
        self.last_box=observation['box'].copy()
        self.current_measurement=STrack.tlbr_to_tlwh(self.last_box)
        if not observation['reliable']:return
        rgb=unit(observation['rgb'])
        assert rgb is not None
        self.rgb=rgb if self.rgb is None else unit(.8*self.rgb+.2*rgb)
        if observation['ir_available']:
            ir=unit(observation['ir']);assert ir is not None
            self.ir=ir if self.ir is None else unit(.8*self.ir+.2*ir)
        else:self.ir=None
        self.accepted+=1;self.memory_frame=self.frame_id
        self.memory_quality=float(observation['quality'])

    def activate(self,kalman_filter,frame_id):
        super().activate(kalman_filter,frame_id);self.accept(self.observation)

    def update(self,new_track,frame_id):
        super().update(new_track,frame_id);self.accept(new_track.observation)

    def re_activate(self,new_track,frame_id,new_id=False):
        super().re_activate(new_track,frame_id,new_id);self.accept(new_track.observation)


def normalized_descriptors(values,count):
    values=np.asarray(values,dtype=np.float32)
    assert values.ndim==2 and len(values)==count and values.shape[1]>0
    assert np.isfinite(values).all()
    norms=np.linalg.norm(values,axis=1,keepdims=True)
    assert (norms>1e-9).all()
    return values/np.maximum(norms,1e-9)


def observations(values,rgb,ir=None):
    """Restore FP16 banks to FP32 and normalize one current frame at a time."""
    rgb=normalized_descriptors(rgb,len(values))
    available=ir is not None
    if available:
        ir=normalized_descriptors(ir,len(values));assert ir.shape==rgb.shape
    overlap=matching.ious(values[:,:4],values[:,:4])if len(values)else np.empty((0,0))
    np.fill_diagonal(overlap,0)
    maximum=overlap.max(axis=1)if len(values)else np.empty(0)
    return [dict(proposal_index=i,box=values[i,:4].copy(),rgb=rgb[i],
        ir=ir[i]if available else None,ir_available=available,
        quality=float(values[i,4])*(1-float(maximum[i])),
        reliable=bool(values[i,4]>.6 and maximum[i]<=.5))for i in range(len(values))]


class CandidateTeacher:
    def __init__(self,speed_upper,observer=None):
        assert 0<=speed_upper<=1
        self.speed_upper,self.observer=speed_upper,observer
        self.sequence,self.core=None,None

    def reset(self,sequence):
        self.sequence=sequence;counter=[0]
        args=SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False)
        self.core=BYTETracker(args,frame_rate=25,
            track_factory=lambda box,score,obs:CandidateTrack(box,score,obs,counter),
            association_hook=self.observe)
        assert self.core.max_time_lost==25
        self.statistics=dict(frames=0,proposals=0,primary_frames=0,eligible_pairs=0,
            original_association_cost_entries_changed=0)

    def observe(self,tracks,detections,base,frame,stage):
        if stage=='primary'and self.observer is not None:
            x,eligible=candidate_features(tracks,detections,base,frame,self.speed_upper)
            self.statistics['primary_frames']+=1
            self.statistics['eligible_pairs']+=int(eligible.sum())
            # An observer gets no mutable tracker, detection or cost references.
            self.observer(x,eligible,tuple(int(t.track_id)for t in tracks),
                tuple(int(d.observation['proposal_index'])for d in detections),frame)
        return base

    def update(self,sequence,frame,detections,rgb,ir=None):
        if sequence!=self.sequence:
            assert frame==1;self.reset(sequence)
        assert frame==self.core.frame_id+1
        values=np.asarray(detections,dtype=np.float32)
        assert values.ndim==2 and values.shape[1]==5 and np.isfinite(values).all()
        assert not len(values)or((values[:,2:4]>values[:,:2]).all()and(values[:,4]>=0).all()and(values[:,4]<=1).all())
        evidence=observations(values,rgb,ir)
        active=self.core.update(values.copy(),(1,1),(1,1),evidence=evidence)
        self.statistics['frames']+=1;self.statistics['proposals']+=len(values)
        rows=[(frame,t.tlwh.copy(),int(t.track_id),t.score)for t in active if t.tlwh[2]*t.tlwh[3]>100]
        live=self.core.tracked_stracks+self.core.lost_stracks
        for track in live:track.observation=None
        # Retire large descriptors once objects really leave the matching pool.
        keep={id(t)for t in live}
        for track in self.core.removed_stracks:
            if id(track)not in keep:track.observation=None;track.rgb=None;track.ir=None
        return rows

    def current_proposal_assignments(self,frame):
        """GT-free current detection indices; callers may join Train labels later."""
        assert self.core is not None and frame==self.core.frame_id
        result={}
        for track in self.core.tracked_stracks:
            if track.frame_id==frame and track.measurement_frame==frame:
                assert track.proposal_index is not None
                result[int(track.track_id)]=int(track.proposal_index)
        assert len(set(result.values()))==len(result)
        return result
