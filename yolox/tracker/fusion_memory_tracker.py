"""Conditional F4 engineering over unchanged ByteTrack geometry/decisions.

Only a later Dev-authorized visual workflow may supply learned P3 features.
This module alone proves no current-IR accuracy, training or innovation.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .byte_tracker import BYTETracker,STrack
from . import matching


@dataclass(frozen=True)
class FusionMemoryConfig:
    appearance_weight: float = .05
    feature_source: str = 'fused'
    quality_mode: str = 'provided'
    minimum_reliability: float = .2
    update_alpha: float = .2
    minimum_IoU: float = .1
    decay_frames: float = 25.

    def __post_init__(self):
        values=[self.appearance_weight,self.minimum_reliability,self.update_alpha,self.minimum_IoU,self.decay_frames]
        if (not np.isfinite(values).all() or not 0<=self.appearance_weight<=.1 or
                not 0<=self.minimum_reliability<=1 or not 0<self.update_alpha<=1 or
                not 0<=self.minimum_IoU<=1 or self.decay_frames<=0 or
                self.feature_source not in ['rgb','fused'] or self.quality_mode not in ['provided','uniform']):
            raise ValueError('Invalid preregistered fusion-memory configuration')


def unit(value):
    value=np.array(value,dtype=np.float32,copy=True)
    if value.ndim!=1 or not len(value) or not np.isfinite(value).all():
        raise ValueError('Each supplied embedding must be a finite nonempty vector')
    norm=float(np.linalg.norm(value.astype(np.float64)))
    return None if norm<=1e-12 else (value/norm).astype(np.float32)


@dataclass(frozen=True)
class FusionObservation:
    rgb: object
    auxiliary: object
    fused: object
    reliability: float
    auxiliary_available: bool


def observations(detections,packet,quality_mode,feature_source='fused'):
    """Current-frame evidence only; unavailable auxiliary arrays are not read."""
    count=len(detections)
    if not count:return []
    if packet is None:raise ValueError('Enabled memory requires explicit current RGB/P3 evidence')
    rgb=np.asarray(packet['rgb'])
    if rgb.ndim!=2 or rgb.shape[0]!=count:raise ValueError('RGB embeddings must align with every candidate')
    if feature_source=='rgb' and quality_mode=='uniform':
        available=np.zeros(count,dtype=bool)
    else:
        available=np.asarray(packet['auxiliary_available'])
        if available.shape!=(count,) or available.dtype!=np.bool_:
            raise ValueError('Explicit per-observation boolean auxiliary availability required')
    quality=np.ones(count,dtype=np.float64)
    aux,fused=None,None
    if available.any():
        if feature_source=='fused':
            aux=np.asarray(packet['auxiliary']);fused=np.asarray(packet['fused'])
            if (aux.ndim!=2 or aux.shape[0]!=count or fused.shape!=rgb.shape):
                raise ValueError('Actual P3 triplets must align; fused and RGB dimensions must match')
        if quality_mode=='provided':
            provided=np.asarray(packet['quality'],dtype=np.float64)
            if provided.shape!=(count,) or not np.isfinite(provided[available]).all() or (
                    (provided[available]<0)|(provided[available]>1)).any():
                raise ValueError('Available fusion observations require a bounded explicit quality proxy')
            quality[available]=provided[available]
    active=detections[:,4]>.1
    overlap=np.zeros(count,dtype=np.float64)
    if active.any():
        indices=np.flatnonzero(active);ious=matching.ious(detections[active,:4],detections[active,:4])
        np.fill_diagonal(ious,0.)
        overlap[indices]=ious.max(axis=1)
    result=[]
    for index in range(count):
        reliability=float(detections[index,4])*(1-float(overlap[index]))*quality[index]
        result.append(FusionObservation(unit(rgb[index]),unit(aux[index]) if aux is not None and available[index] else None,
            unit(fused[index]) if fused is not None and available[index] else None,float(np.clip(reliability,0,1)),bool(available[index])))
    return result


class FusionMemoryTrack(STrack):
    """One normalized vector per modality; low reliability cannot overwrite it."""
    def __init__(self,tlwh,score,observation,config,counter,minimum_score):
        super().__init__(tlwh,score)  # Preserve original score dtype and KF state.
        self.observation,self.memory_config=observation,config
        self.identity_counter,self.minimum_memory_score=counter,minimum_score
        self.rgb_memory,self.aux_memory,self.fused_memory=None,None,None
        self.memory_last_frame={key:None for key in ['rgb','auxiliary','fused']}
        self.memory_last_reliability={key:0. for key in self.memory_last_frame}
        self.current_memory_frame=0

    def next_id(self):
        self.identity_counter[0]+=1
        return self.identity_counter[0]

    @property
    def memory_reliability(self):return self.memory_last_reliability[self.memory_config.feature_source]

    @property
    def memory_age(self):
        previous=self.memory_last_frame[self.memory_config.feature_source]
        return None if previous is None else self.current_memory_frame-previous

    def accept(self,observation,frame):
        self.current_memory_frame=frame
        if observation is None or self.score<=self.minimum_memory_score:
            return
        if observation.reliability<=0 or observation.reliability<self.memory_config.minimum_reliability:return
        for key,name,value in [('rgb','rgb_memory',observation.rgb),
                ('auxiliary','aux_memory',observation.auxiliary),('fused','fused_memory',observation.fused)]:
            if value is None:continue
            value=unit(value)
            if value is None:continue
            previous=getattr(self,name)
            if previous is not None and previous.shape!=value.shape:
                raise ValueError('A memory stream changed its learned feature dimensions')
            amount=self.memory_config.update_alpha*observation.reliability
            updated=value.copy() if previous is None else unit((1-amount)*previous+amount*value)
            if updated is None:continue
            setattr(self,name,updated)
            self.memory_last_frame[key]=frame
            self.memory_last_reliability[key]=observation.reliability

    def activate(self,kalman_filter,frame_id):
        super().activate(kalman_filter,frame_id);self.accept(self.observation,frame_id)

    def update(self,new_track,frame_id):
        super().update(new_track,frame_id);self.accept(new_track.observation,frame_id)

    def re_activate(self,new_track,frame_id,new_id=False):
        super().re_activate(new_track,frame_id,new_id);self.accept(new_track.observation,frame_id)

    def mark_removed(self):
        super().mark_removed()
        self.rgb_memory,self.aux_memory,self.fused_memory=None,None,None

    def memory_for(self,key,frame):
        name={'rgb':'rgb_memory','fused':'fused_memory'}[key]
        vector=getattr(self,name);previous=self.memory_last_frame[key]
        if vector is None or previous is None:return None,0.
        if frame<=previous:raise ValueError('Association memory must strictly precede the current frame')
        reliability=self.memory_last_reliability[key]*np.exp(-(frame-previous)/self.memory_config.decay_frames)
        return vector,float(reliability)


class FusionMemoryTracker:
    """Explicit sequence/frame API for source-coordinate current detections.

No visual model, feature cache, labels, future frame, train loop or GPU launch.
The original BYTETracker remains unchanged and performs all association stages.
"""
    def __init__(self,config=None,args=None,frame_rate=25,identity_start=0):
        if not isinstance(identity_start,int) or identity_start<0:
            raise ValueError('Explicit nonnegative initial identity counter required')
        self.config=FusionMemoryConfig() if config is None else config
        self.args=args if args is not None else SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False)
        self.frame_rate,self.identity_start=frame_rate,identity_start;self.sequence=None;self.core=None
        self.statistics=dict(primary_cost_calls=0,changed_edges=0)

    def reset(self,sequence):
        self.sequence=sequence;counter=[self.identity_start]
        factory=lambda tlwh,score,evidence:FusionMemoryTrack(tlwh,score,evidence,self.config,counter,self.args.track_thresh)
        self.core=BYTETracker(self.args,frame_rate=self.frame_rate,track_factory=factory,
            association_hook=self.cost if self.config.appearance_weight else None)
        self.statistics=dict(primary_cost_calls=0,changed_edges=0)

    def cost(self,tracks,detections,base,frame,stage):
        if stage!='primary' or not self.config.appearance_weight or not base.size:return base
        self.statistics['primary_cost_calls']+=1
        result=base.copy();ious=1-matching.iou_distance(tracks,detections)
        for row,track in enumerate(tracks):
            for column,detection in enumerate(detections):
                obs=detection.observation
                if (obs is None or ious[row,column]<self.config.minimum_IoU or
                        not np.isfinite(base[row,column]) or base[row,column]>self.args.match_thresh):continue
                key=self.config.feature_source if obs.auxiliary_available else 'rgb'
                current=obs.fused if key=='fused' else obs.rgb
                previous,reliability=track.memory_for(key,frame)
                if current is None or previous is None:continue
                if previous.shape!=current.shape:raise ValueError('Current and past appearance spaces differ')
                distance=float(np.clip((1-float(np.dot(previous,current)))/2,0,1))
                penalty=self.config.appearance_weight*reliability*obs.reliability*distance
                result[row,column]+=penalty
                self.statistics['changed_edges']+=int(penalty>0)
        return result

    def update(self,sequence,frame,detections,feature_packet=None):
        if not sequence or not isinstance(frame,int) or frame<1:raise ValueError('Explicit causal video/frame required')
        if self.sequence!=sequence:
            if frame!=1:raise ValueError('A new video must start at frame1')
        elif frame!=self.core.frame_id+1:raise ValueError('Every consecutive frame must be delivered once')
        array=np.asarray(detections)
        if (array.ndim!=2 or array.shape[1]!=5 or not np.isfinite(array).all() or
                ((array[:,4]<0)|(array[:,4]>1)).any() or
                (array[:,2]<=array[:,0]).any() or (array[:,3]<=array[:,1]).any()):
            raise ValueError('Finite source-coordinate xyxy/score packets required')
        evidence=observations(array,feature_packet,self.config.quality_mode,self.config.feature_source) if self.config.appearance_weight else None
        if self.sequence!=sequence:self.reset(sequence)
        for track in self.core.tracked_stracks+self.core.lost_stracks:track.current_memory_frame=frame
        return self.core.update(array.copy(),(1,1),(1,1),evidence=evidence)
