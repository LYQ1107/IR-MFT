"""Causal cue competition within the original admissible ByteTrack graph.

Confident continuous geometric matches are protected. Appearance only adds
bounded penalties to competing admissible edges; no unsupported edge is added.
"""
import numpy as np
from tools.mft_tracklet_permanence import MemoryTrack,cases as original_cases
from tools.mft_tracklet_permanence_r2 import TrackletPermanence
from yolox.tracker import matching
from yolox.tracker.basetrack import TrackState


def policies():
    return [dict(id='AR_native_zero',weight=0.,late=False,cue='none'),
            dict(id='AR_late_only',weight=0.,late=True,cue='none'),
            dict(id='AR_RGB_primary',weight=.5,late=False,cue='none'),
            dict(id='AR_RGB_IR_primary',weight=.5,late=False,cue='ir'),
            dict(id='AR_RGB_primary_late',weight=.5,late=True,cue='none'),
            dict(id='AR_RGB_IR_primary_late',weight=.5,late=True,cue='ir'),
            dict(id='AR_RGB_RGB_primary_late',weight=.5,late=True,cue='rgb'),
            dict(id='AR_IR_descriptor_roll',weight=.5,late=True,cue='ir_roll'),
            dict(id='AR_IR_descriptor_unavailable',weight=.5,late=True,cue='missing')]


class QualityTrack(MemoryTrack):
    def __init__(self,*args):
        super().__init__(*args)
        self.memory_quality,self.memory_frame=0.,None

    def accept(self,obs):
        before=self.accepted
        super().accept(obs)
        if self.accepted>before:
            self.memory_quality=float(obs.get('quality',1.))
            self.memory_frame=self.frame_id


def unique_preferences(scores,threshold,margin=.1):
    """Competitors below the acceptance threshold remain in the margin."""
    result={}
    if not scores.size:return result
    for i in range(len(scores)):
        if not np.isfinite(scores[i]).any():continue
        j=int(np.argmax(scores[i]));value=scores[i,j]
        if value<threshold or int(np.argmax(scores[:,j]))!=i:continue
        row,col=np.delete(scores[i],j),np.delete(scores[:,j],i)
        second=max(float(row.max())if len(row)else -np.inf,float(col.max())if len(col)else -np.inf)
        if value-second>=margin:result[i]=j
    return result


class AmbiguityReasoning(TrackletPermanence):
    def __init__(self,policy,thresholds):
        assert policy in policies()
        self.policy=dict(policy)
        # The old component supplies only the tested lifetime/first-output links.
        needs_features=bool(policy['weight']or policy['late'])
        super().__init__(original_cases()[2 if needs_features else 0],thresholds)

    def reset(self,sequence):
        super().reset(sequence)
        counter=[0]
        self.core.track_factory=lambda box,score,obs:QualityTrack(box,score,obs,counter)
        self.core.association_hook=self.cost if self.policy['weight']else None
        self.statistics.update(primary_calls=0,protected_geometric_pairs=0,RGB_preference_pairs=0,
            IR_rescue_pairs=0,positive_penalty_edges=0,actual_changed_cost_entries=0,primary_assignment_changed_frames=0)

    def link(self,new,frame,active_roots):
        return super().link(new,frame,active_roots)if self.policy['late']else{}

    def cost(self,tracks,detections,base,frame,stage):
        if stage!='primary'or not base.size:return base
        self.statistics['primary_calls']+=1
        original_matches=matching.linear_assignment(base,thresh=self.core.args.match_thresh)[0]
        result=base.copy();allowed=np.isfinite(base)&(base<=self.core.args.match_thresh)
        ious=1-matching.iou_distance(tracks,detections);allowed&=ious>=.1
        protected=[]
        for i,j in original_matches:
            row=np.delete(base[i],j);col=np.delete(base[:,j],i)
            gap=min(float(row.min())if len(row)else np.inf,float(col.min())if len(col)else np.inf)-base[i,j]
            if(tracks[i].state==TrackState.Tracked and tracks[i].frame_id==frame-1 and base[i,j]<=.4 and gap>=.1):
                protected.append((int(i),int(j)))
        for i,j in protected:
            allowed[i,:]=False;allowed[:,j]=False
            result[i,:]=np.inf;result[:,j]=np.inf
        for i,j in protected:result[i,j]=base[i,j]
        self.statistics['protected_geometric_pairs']+=len(protected)
        rgb=np.full(base.shape,-np.inf,np.float64);ir=np.full(base.shape,-np.inf,np.float64)
        reliability=np.zeros(base.shape,np.float64)
        for i,track in enumerate(tracks):
            if track.accepted<3 or track.rgb is None:continue
            assert track.memory_frame is not None and track.memory_frame<frame
            for j,det in enumerate(detections):
                obs=det.observation
                if not allowed[i,j]or obs is None:continue
                rgb[i,j]=float(track.rgb@obs['rgb'])
                reliability[i,j]=track.memory_quality*obs['quality']
                if obs['auxiliary_available']and track.ir is not None:ir[i,j]=float(track.ir@obs['ir'])
        preferred_rgb=unique_preferences(rgb,self.thresholds['primary_RGB_cosine_threshold'])
        preferred_ir={}
        if self.policy['cue']not in['none','missing']:
            remaining=ir.copy()
            for i,j in preferred_rgb.items():remaining[i,:]=-np.inf;remaining[:,j]=-np.inf
            theta=self.thresholds['primary_RGB_cosine_threshold']if self.policy['cue']=='rgb'else self.thresholds['primary_IR_cosine_threshold']
            preferred_ir=unique_preferences(remaining,theta)
        self.statistics['RGB_preference_pairs']+=len(preferred_rgb);self.statistics['IR_rescue_pairs']+=len(preferred_ir)
        penalty=np.zeros(base.shape,np.float64)
        for preferred,score in [(preferred_rgb,rgb),(preferred_ir,ir)]:
            for i,j in preferred.items():
                for k in range(len(detections)):
                    if allowed[i,k]and k!=j and np.isfinite(score[i,k]):
                        value=self.policy['weight']*reliability[i,k]*np.clip((score[i,j]-score[i,k])/2,0,1)
                        penalty[i,k]=max(penalty[i,k],value)
                for h in range(len(tracks)):
                    if allowed[h,j]and h!=i and np.isfinite(score[h,j]):
                        value=self.policy['weight']*reliability[h,j]*np.clip((score[i,j]-score[h,j])/2,0,1)
                        penalty[h,j]=max(penalty[h,j],value)
        # Without affirmative cue competition, preserve even LAP tie-breaking.
        if not np.any(penalty>0):return base
        self.statistics['positive_penalty_edges']+=int(np.count_nonzero(penalty>0))
        result+=penalty
        assert np.all(result>=base),'Appearance cannot admit an originally unsupported edge'
        changed=matching.linear_assignment(result,thresh=self.core.args.match_thresh)[0]
        count=int(np.count_nonzero(result!=base))
        self.statistics['actual_changed_cost_entries']+=count
        self.statistics['primary_association_costs_changed']+=count
        self.statistics['primary_assignment_changed_frames']+=int(set(map(tuple,changed))!=set(map(tuple,original_matches)))
        return result

    def update(self,sequence,frame,detections,rgb=None,ir=None):
        if self.sequence!=sequence:
            assert frame==1;self.reset(sequence)
        assert frame==self.core.frame_id+1
        values=np.asarray(detections,dtype=np.float32)
        assert values.ndim==2 and values.shape[1]==5 and np.isfinite(values).all()
        evidence=None
        if self.policy['weight']or self.policy['late']:
            rgb=np.asarray(rgb,dtype=np.float32)
            assert rgb.ndim==2 and len(rgb)==len(values)and np.isfinite(rgb).all()
            cue=self.policy['cue'];available=cue not in['none','missing']
            if available:
                auxiliary=rgb if cue=='rgb'else np.asarray(ir,dtype=np.float32)
                if cue=='ir_roll':auxiliary=np.roll(auxiliary,1,axis=0)
                assert auxiliary.shape==rgb.shape and np.isfinite(auxiliary).all()
            else:auxiliary=rgb  # Explicit RGB-only placeholder; no IR array read.
            overlap=matching.ious(values[:,:4],values[:,:4])if len(values)else np.empty((0,0))
            np.fill_diagonal(overlap,0);maximum=overlap.max(1)if len(values)else np.empty(0)
            evidence=[dict(rgb=rgb[i],ir=auxiliary[i],box=values[i,:4].copy(),
                quality=float(values[i,4])*(1-float(maximum[i])),
                reliable=bool(values[i,4]>.6 and maximum[i]<=.5),auxiliary_available=available)for i in range(len(values))]
        tracks=self.core.update(values.copy(),(1,1),(1,1),evidence=evidence)
        visible=[t for t in tracks if t.tlwh[2]*t.tlwh[3]>100]
        self.archive(frame);active={self.published[t.track_id]for t in visible if t.track_id in self.published}
        new=[t for t in visible if t.track_id not in self.published];links=self.link(new,frame,active)
        for t in new:self.published[t.track_id]=links.get(t.track_id,t.track_id)
        rows=[]
        for t in visible:
            identity=self.published[t.track_id];assert identity not in[r[2]for r in rows]
            self.last_published[identity]=frame;rows.append((frame,t.tlwh.copy(),identity,t.score))
        self.statistics['eligible_removed_memories']=len(self.closed)
        for t in self.core.tracked_stracks+self.core.lost_stracks:t.observation=None
        return rows
