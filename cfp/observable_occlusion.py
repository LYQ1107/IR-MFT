"""Independent association-only controls using current high-stage box overlap.

Overlap is an observable ambiguity proxy, not a physical occlusion label.
No detector/features/KF/lifecycle/history-write changes or learned novelty claim.
"""
import numpy as np
from yolox.tracker import matching
from .fish_geometry import box_terms
from .history import anchor_recent_history


MODES=('native','constant_half','overlap','row_mean','permuted_overlap')


def current_overlap(boxes):
    boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    if not np.isfinite(boxes).all():raise ValueError('Finite actual current boxes required')
    if len(boxes)<2:return np.zeros(len(boxes),np.float64)
    iou=box_terms(boxes,boxes)[8];np.fill_diagonal(iou,0.)
    return iou.max(1)


def quality_gate(overlap,eligible,mode,rng):
    overlap=np.asarray(overlap,np.float64);eligible=np.asarray(eligible,bool)
    if mode not in MODES or eligible.ndim!=2 or eligible.shape[1]!=len(overlap) or not np.isfinite(overlap).all() or (overlap<0).any() or (overlap>1).any():raise ValueError('Aligned observable overlap and native graph required')
    gate=np.ones(eligible.shape,np.float64)
    quality=1-overlap
    for row in range(len(gate)):
        indices=np.flatnonzero(eligible[row])
        if not len(indices):continue
        values=quality[indices]
        if mode=='constant_half':values=np.full(len(indices),.5)
        elif mode=='row_mean':values=np.full(len(indices),values.mean())
        elif mode=='permuted_overlap':values=values[rng.permutation(len(indices))]
        elif mode=='native':values=np.ones(len(indices))
        gate[row,indices]=values
    return gate


def install_observable_occlusion(tracker,mode,seed=0):
    if mode not in MODES or seed not in (0,1,2) or tracker.frame_id!=0 or tracker.memory_mode!='predictor':raise ValueError('Fresh registered native predictor tracker required')
    base=type(tracker)
    class ObservableTracker(base):
        def association_cost(self,tracks,detections):
            baseline=super().association_cost(tracks,detections)
            if mode=='native' or not baseline.size or self.appearance_weight==0:return baseline
            raw=matching.iou_distance(tracks,detections);geometry=matching.fuse_score(raw,detections) if not self.args.mot20 else raw
            eligible=np.zeros_like(raw,dtype=bool);valid=np.array([d.current_rgb is not None for d in detections])
            for row,track in enumerate(tracks):
                history=list(track.identity_history)
                if self.history_view=='anchor_recent':
                    anchor=None if track.initial_identity_rgb is None else (track.initial_identity_frame,track.initial_identity_rgb,track.initial_identity_ir)
                    history=anchor_recent_history(history,anchor,8)
                if any(time>=self.frame_id for time,_,_ in history):raise ValueError('Only actual strictlypast identity writes may condition current gate')
                if history:eligible[row]=valid&(raw[row]<=1-self.proximity_iou)
            overlap=current_overlap(np.array([d.tlbr for d in detections]));gate=quality_gate(overlap,eligible,mode,self.occlusion_rng)
            # Baseline=G+w(A-G); multiplying the residual only changes w.
            # Outside the native eligibility graph remain original cost bytes.
            result=baseline.copy();changed=eligible&(gate!=1)
            result[changed]=geometry[changed]+gate[changed]*(baseline[changed]-geometry[changed])
            if not np.isfinite(result).all():raise ValueError('Nonfinite observable association cost')
            c=self.occlusion_counts;c['calls']+=1;c['current_high_candidates']+=len(detections);c['eligible_edges']+=int(eligible.sum());c['changed_edges']+=int(changed.sum());c['overlap_sum']+=float(overlap.sum());c['overlap_positive_candidates']+=int((overlap>0).sum());c['eligible_gate_sum']+=float(gate[eligible].sum());c['threshold_crossings']+=int(((result<=self.args.match_thresh)!=(baseline<=self.args.match_thresh)).sum())
            return result
    tracker.__class__=ObservableTracker;tracker.occlusion_mode=mode;tracker.occlusion_seed=seed;tracker.occlusion_rng=np.random.default_rng(seed);tracker.occlusion_counts=dict(calls=0,current_high_candidates=0,eligible_edges=0,changed_edges=0,overlap_sum=0.,overlap_positive_candidates=0,eligible_gate_sum=0.,threshold_crossings=0)
    return tracker
