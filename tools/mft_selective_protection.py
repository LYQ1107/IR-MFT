"""Train-calibrated exceptions to the frozen geometric-protection rule."""
import numpy as np
from yolox.tracker import matching
from yolox.tracker.basetrack import TrackState
from tools.mft_candidate_risk_features import candidate_features
from tools.mft_candidate_risk_association import RiskAssociation,policies as parent_policies


def policies():
    return [dict(id='SP_RGB_static',source_case='CR_RGB_risk',selective=False),
        dict(id='SP_paired_static',source_case='CR_paired_risk',selective=False),
        dict(id='SP_paired_shape_static',source_case='CR_paired_risk_shape',selective=False),
        dict(id='SP_paired_direct_shape_static',source_case='CR_paired_direct_shape',selective=False),
        dict(id='SP_RGB_selective',source_case='CR_RGB_risk',selective=True),
        dict(id='SP_paired_selective',source_case='CR_paired_risk',selective=True),
        dict(id='SP_paired_shape_selective',source_case='CR_paired_risk_shape',selective=True),
        dict(id='SP_paired_direct_shape_selective',source_case='CR_paired_direct_shape',selective=True)]


def protected_matches(tracks,base,frame,threshold):
    matches=matching.linear_assignment(base,thresh=threshold)[0];protected=[]
    for i,j in matches:
        row=np.delete(base[i],j);column=np.delete(base[:,j],i)
        second=min(float(row.min())if row.size else np.inf,float(column.min())if column.size else np.inf)
        if(tracks[i].state==TrackState.Tracked and tracks[i].frame_id==frame-1
                and base[i,j]<=.4 and second-base[i,j]>=.1):protected.append((int(i),int(j)))
    return matches,protected


def selective_exceptions(p,eligible,structural,protected,threshold):
    """Weak and protected competitors still count; unknown baselines stay safe."""
    scored=np.where(eligible,p,-np.inf);exceptions=set();qualified=[]
    for i in range(len(scored)):
        if not np.isfinite(scored[i]).any():continue
        j=int(np.argmax(scored[i]))
        if not structural[i,j]or p[i,j]<threshold or int(np.argmax(scored[:,j]))!=i:continue
        row=np.delete(scored[i],j);column=np.delete(scored[:,j],i)
        second=max(float(row.max())if row.size else -np.inf,float(column.max())if column.size else -np.inf)
        if p[i,j]-second<.05:continue
        conflicts=[(a,b)for a,b in protected if(a==i or b==j)and(a,b)!=(i,j)]
        if not conflicts:continue
        # An unscored young/ambiguous baseline cannot be labelled wrong just
        # because another track has an old appearance template.
        if any(not eligible[a,b]or p[i,j]-p[a,b]<.05 for a,b in conflicts):continue
        qualified.append((i,j));exceptions.update(conflicts)
    return exceptions,qualified


def fused_cost(tracks,base,p,eligible,protected,frame,blend,rescue_threshold,match_threshold):
    allowed=eligible.copy()
    for i,j in protected:allowed[i,:]=False;allowed[:,j]=False
    result=base.copy();ordinary=allowed&(base<=match_threshold)
    result[ordinary]=(1-blend)*base[ordinary]+blend*(1-p[ordinary])
    scored=np.where(eligible,p,-np.inf);rescue=np.zeros_like(eligible)
    for i,t in enumerate(tracks):
        if t.state!=TrackState.Lost or not 2<=frame-t.frame_id<=25:continue
        if not np.isfinite(scored[i]).any():continue
        j=int(np.argmax(scored[i]))
        if not allowed[i,j]or base[i,j]<=match_threshold:continue
        if int(np.argmax(scored[:,j]))!=i or p[i,j]<rescue_threshold:continue
        row=np.delete(scored[i],j);column=np.delete(scored[:,j],i)
        other=max(float(row.max())if row.size else -np.inf,float(column.max())if column.size else -np.inf)
        if p[i,j]-other>=.05:rescue[i,j]=True
    result[rescue]=(1-blend)*match_threshold+blend*(1-p[rescue])
    if np.array_equal(result,base):return base,rescue
    for i,j in protected:result[i,:]=np.inf;result[:,j]=np.inf;result[i,j]=base[i,j]
    return result,rescue


class SelectiveProtection(RiskAssociation):
    def __init__(self,case,models,thresholds,override_thresholds,speed_upper):
        assert case in policies()and set(override_thresholds)=={'RGB','paired'}
        assert all(thresholds[k]<=override_thresholds[k]<=1 for k in override_thresholds)
        self.selective_case=dict(case);self.override_thresholds=dict(override_thresholds)
        parent=next(c for c in parent_policies()if c['id']==case['source_case'])
        super().__init__(parent,models,thresholds,speed_upper)

    def reset(self,sequence):
        super().reset(sequence)
        self.statistics.update(qualified_override_preferences=0,unlocked_geometric_pairs=0,
            selective_vs_static_cost_entries_changed=0,selective_vs_static_assignment_changed_frames=0)

    def cost(self,tracks,detections,base,frame,stage):
        if not self.selective_case['selective']:return super().cost(tracks,detections,base,frame,stage)
        if stage!='primary'or not base.size:return base
        x,eligible=candidate_features(tracks,detections,base,frame,self.speed_upper)
        kind=self.case['kind']
        if kind=='paired'and not all(d.observation['ir_available']for d in detections):kind='RGB'
        self.statistics['paired_primary_calls'if kind=='paired'else'RGB_fallback_primary_calls']+=1
        self.statistics['primary_frames']+=1;self.statistics['eligible_pairs']+=int(eligible.sum())
        if not eligible.any():return base
        p=np.zeros(base.shape,np.float64);p[eligible]=self.models[kind].probabilities(x[eligible])
        original,protected=protected_matches(tracks,base,frame,self.core.args.match_thresh)
        structural=eligible&(base<=self.core.args.match_thresh)
        for i,t in enumerate(tracks):
            if t.state==TrackState.Lost and 2<=frame-t.frame_id<=25:structural[i]|=eligible[i]
        exceptions,qualified=selective_exceptions(p,eligible,structural,protected,self.override_thresholds[kind])
        remaining=[pair for pair in protected if pair not in exceptions]
        result,rescue=fused_cost(tracks,base,p,eligible,remaining,frame,self.case['blend'],
            self.thresholds[kind],self.core.args.match_thresh)
        self.statistics['qualified_override_preferences']+=len(qualified)
        self.statistics['unlocked_geometric_pairs']+=len(exceptions)
        if exceptions:
            static,_=fused_cost(tracks,base,p,eligible,protected,frame,self.case['blend'],
                self.thresholds[kind],self.core.args.match_thresh)
            self.statistics['selective_vs_static_cost_entries_changed']+=int(np.count_nonzero(result!=static))
            a=matching.linear_assignment(result,thresh=self.core.args.match_thresh)[0]
            b=matching.linear_assignment(static,thresh=self.core.args.match_thresh)[0]
            self.statistics['selective_vs_static_assignment_changed_frames']+=int(set(map(tuple,a))!=set(map(tuple,b)))
        if np.array_equal(result,base):return base
        changed=int(np.count_nonzero(result!=base))
        after=matching.linear_assignment(result,thresh=self.core.args.match_thresh)[0]
        self.statistics['protected_pairs']+=len(remaining)
        self.statistics['actual_changed_cost_entries']+=changed
        self.statistics['original_association_cost_entries_changed']+=changed
        self.statistics['new_reachable_edge_costs']+=int(rescue.sum())
        self.statistics['primary_assignment_changed_frames']+=int(set(map(tuple,original))!=set(map(tuple,after)))
        return result
