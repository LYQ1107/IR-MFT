"""Wait for actual matching-pool departure before reusing an output identity.

The preserved ByteTrack appends newly removed tracks after subtracting its old
removed list. A just-removed object can therefore remain in lost_stracks for
one more update. Retiring its public identity at that point is premature.
"""
import numpy as np
from yolox.tracker.basetrack import TrackState
from tools.mft_tracklet_permanence import TrackletPermanence as Original,cases


class TrackletPermanence(Original):
    def reset(self,sequence):
        super().reset(sequence)
        self.pending_removed={}
        self.statistics['deferred_removed_still_in_matching_pool']=0

    def archive(self,frame):
        removed=self.core.removed_stracks
        assert len(removed)>=self.removed_prefix
        for track in removed[self.removed_prefix:]:
            self.pending_removed[track.track_id]=track
        self.removed_prefix=len(removed)
        eligible_for_matching={track.track_id for track in self.core.tracked_stracks+self.core.lost_stracks}
        for internal,track in list(self.pending_removed.items()):
            if internal in eligible_for_matching:
                self.statistics['deferred_removed_still_in_matching_pool']+=1
                continue
            identity=self.published.get(internal)
            if (track.state==TrackState.Removed and identity is not None and track.accepted>=3
                    and track.rgb is not None and track.last_box is not None):
                self.closed[identity]=dict(rgb=track.rgb.copy(),ir=track.ir.copy(),box=track.last_box.copy(),
                    last_frame=self.last_published[identity],internal_track=internal)
            track.rgb,track.ir,track.observation=None,None,None
            del self.pending_removed[internal]
        self.closed={key:value for key,value in self.closed.items()if frame-value['last_frame']<=self.case['maximum_gap']}

    def link(self,new,frame,active_roots):
        if not self.case['alias'] or not new:
            return {}
        old=[(key,value)for key,value in sorted(self.closed.items())if key not in active_roots]
        if not old:return {}
        scores=np.full((len(new),len(old)),-np.inf,np.float64)
        eligible=np.zeros(scores.shape,bool)
        for i,track in enumerate(new):
            if track.accepted<2 or track.rgb is None:continue
            for j,(identity,memory)in enumerate(old):
                gap=frame-memory['last_frame'];assert gap>0
                if not 2<=gap<=self.case['maximum_gap']:continue
                old_box,current_box=memory['box'],track.last_box
                a,b=np.maximum(old_box[2:]-old_box[:2],1.),np.maximum(current_box[2:]-current_box[:2],1.)
                ratio=b/a
                if(ratio<.5).any()or(ratio>2.).any():continue
                distance=np.linalg.norm(((old_box[:2]+old_box[2:])-(current_box[:2]+current_box[2:]))/2)/np.linalg.norm(a)
                if distance>.5+self.thresholds['Train_unknown_speed_upper']*gap:continue
                self.statistics['candidate_pairs']+=1
                rgb,ir=float(track.rgb@memory['rgb']),float(track.ir@memory['ir'])
                # Competitors below the acceptance threshold still affect margin.
                scores[i,j]=rgb
                if rgb<self.thresholds['RGB_cosine_threshold']:continue
                self.statistics['RGB_gate_passes']+=1
                if self.case['use_IR']and ir<self.thresholds['IR_cosine_threshold']:continue
                self.statistics['IR_gate_passes']+=int(self.case['use_IR']);eligible[i,j]=True
        result={}
        for i,track in enumerate(new):
            if not np.isfinite(scores[i]).any():continue
            j=int(np.argmax(scores[i]))
            if not eligible[i,j]or int(np.argmax(scores[:,j]))!=i:continue
            row,column=np.delete(scores[i],j),np.delete(scores[:,j],i)
            second=max(float(row.max())if len(row)else -np.inf,float(column.max())if len(column)else -np.inf)
            if scores[i,j]-second<.1:continue
            identity=old[j][0]
            assert identity not in result.values()and track.track_id not in self.published
            result[track.track_id]=identity;del self.closed[identity];self.statistics['identity_links']+=1
        return result
