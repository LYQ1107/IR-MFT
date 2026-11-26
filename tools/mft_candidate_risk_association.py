"""Learned cue risk for ambiguous primary association, with RGB fallback."""
import numpy as np
from yolox.tracker import matching
from yolox.tracker.basetrack import TrackState
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_candidate_risk_features import candidate_features


def policies():
    return [dict(id='CR_native_zero',kind='reference_native',blend=0.,shape=False),
        dict(id='CR_pose_late_control',kind='reference_pose_late',blend=0.,shape=True),
        dict(id='CR_RGB_risk',kind='RGB',blend=.5,shape=False),
        dict(id='CR_paired_risk',kind='paired',blend=.5,shape=False),
        dict(id='CR_paired_risk_shape',kind='paired',blend=.5,shape=True),
        dict(id='CR_paired_direct_shape',kind='paired',blend=1.,shape=True)]


class RiskAssociation(CandidateTeacher):
    def __init__(self,case,models,thresholds,speed_upper):
        assert case in policies()and case['kind']in ['RGB','paired']
        assert set(models)==set(thresholds)=={'RGB','paired'}
        assert all(models[k].kind==k and 0<=thresholds[k]<=1 for k in models)
        self.case=dict(case);self.models=models;self.thresholds=thresholds
        super().__init__(speed_upper)

    def reset(self,sequence):
        super().reset(sequence);self.core.association_hook=self.cost
        self.statistics.update(protected_pairs=0,actual_changed_cost_entries=0,
            primary_assignment_changed_frames=0,new_reachable_edge_costs=0,
            RGB_fallback_primary_calls=0,paired_primary_calls=0,shape_rows=0)

    def cost(self,tracks,detections,base,frame,stage):
        if stage!='primary'or not base.size:return base
        x,eligible=candidate_features(tracks,detections,base,frame,self.speed_upper)
        kind=self.case['kind']
        if kind=='paired'and not all(d.observation['ir_available']for d in detections):kind='RGB'
        self.statistics['paired_primary_calls'if kind=='paired'else'RGB_fallback_primary_calls']+=1
        self.statistics['primary_frames']+=1;self.statistics['eligible_pairs']+=int(eligible.sum())
        if not eligible.any():return base
        p=np.zeros(base.shape,np.float64);p[eligible]=self.models[kind].probabilities(x[eligible])
        original=matching.linear_assignment(base,thresh=self.core.args.match_thresh)[0]
        protected=[]
        for i,j in original:
            row=np.delete(base[i],j);column=np.delete(base[:,j],i)
            second=min(float(row.min())if row.size else np.inf,float(column.min())if column.size else np.inf)
            if(tracks[i].state==TrackState.Tracked and tracks[i].frame_id==frame-1
                    and base[i,j]<=.4 and second-base[i,j]>=.1):
                protected.append((int(i),int(j)))
        allowed=eligible.copy()
        for i,j in protected:allowed[i,:]=False;allowed[:,j]=False
        result=base.copy();ordinary=allowed&(base<=self.core.args.match_thresh)
        w=self.case['blend']
        result[ordinary]=(1-w)*base[ordinary]+w*(1-p[ordinary])
        # Only a calibrated mutual preference can admit a reachable lost track
        # outside the original IoU/score graph. Weak competitors still count.
        scored=np.where(eligible,p,-np.inf)
        rescue=np.zeros_like(eligible)
        for i,t in enumerate(tracks):
            if t.state!=TrackState.Lost or not 2<=frame-t.frame_id<=25:continue
            if not np.isfinite(scored[i]).any():continue
            j=int(np.argmax(scored[i]))
            if not allowed[i,j]or base[i,j]<=self.core.args.match_thresh:continue
            if int(np.argmax(scored[:,j]))!=i or p[i,j]<self.thresholds[kind]:continue
            row=np.delete(scored[i],j);column=np.delete(scored[:,j],i)
            other=max(float(row.max())if row.size else -np.inf,float(column.max())if column.size else -np.inf)
            if p[i,j]-other>=.05:rescue[i,j]=True
        result[rescue]=(1-w)*self.core.args.match_thresh+w*(1-p[rescue])
        if np.array_equal(result,base):return base
        for i,j in protected:
            result[i,:]=np.inf;result[:,j]=np.inf;result[i,j]=base[i,j]
        changed=int(np.count_nonzero(result!=base))
        after=matching.linear_assignment(result,thresh=self.core.args.match_thresh)[0]
        self.statistics['protected_pairs']+=len(protected)
        self.statistics['actual_changed_cost_entries']+=changed
        self.statistics['original_association_cost_entries_changed']+=changed
        self.statistics['new_reachable_edge_costs']+=int(rescue.sum())
        self.statistics['primary_assignment_changed_frames']+=int(set(map(tuple,original))!=set(map(tuple,after)))
        return result

    def update(self,sequence,frame,detections,rgb,ir=None):
        rows=super().update(sequence,frame,detections,rgb,ir if self.case['kind']=='paired'else None)
        if not self.case['shape']:return rows
        live={int(t.track_id):t for t in self.core.tracked_stracks if t.is_activated and t.frame_id==frame}
        result=[]
        for f,filtered,identity,score in rows:
            t=live[identity];assert t.measurement_frame==frame
            measured=t.current_measurement
            if measured[2]*measured[3]>100:
                box=measured.copy();box[:2]=filtered[:2]+filtered[2:]/2-box[2:]/2
                self.statistics['shape_rows']+=1
            else:box=filtered
            result.append((f,box,identity,score))
        return result
