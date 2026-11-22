"""RGB memory owns the first decision; native cues consult only unclaimed IDs."""
import numpy as np

from yolox.tracker.matching import ious,linear_assignment
from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker
from yolox.tracker.counterfactual_query_tracker import CounterfactualQueryTracker


class OrderedIdentityMemoryTracker(CounterfactualQueryTracker):
    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        self.statistics.update(RGB_memory_recoveries=0,native_memory_recoveries=0)

    def reference_memory_scores(self,frame,context,data):
        a,b=map(int,data['offsets'][frame-1:frame+1])
        if frame-context['frame']>self.config.get('RGB_memory_maximum_lag',8):
            return np.zeros(b-a,np.float64),np.zeros(b-a,np.float32)
        packet=self.query_posterior(frame,context);index=context['index']-packet['start']
        probability=packet['RGB_probability'][index].astype(np.float64).copy()
        probability[probability<=packet['RGB_null'][index]+self.config.get('query_margin',.1)]=0
        return probability,np.zeros(b-a,np.float32)

    def recover_phase(self,frame,data,rows,scorer,kind):
        retention=int(self.config.get('identity_retention',25))
        if retention not in {25,128}:raise ValueError('Unregistered identity retention')
        old=[(identity,entry) for identity,entry in sorted(self.identity_archive.items())
             if 0<frame-entry['last_seen']<=retention and entry['track'].frame_id<frame]
        owners={}
        for track in self.inner.tracked_stracks:
            if track.frame_id==frame:owners.setdefault(track.evidence['index'],[]).append(track)
        observations=[o for o in self.created if o.score>=self.config.get('prebirth_minimum_score',.3)
            and o.tlwh[2]*o.tlwh[3]>100 and not any(t.track_id in self.emitted_identities for t in owners.get(o.evidence['index'],[]))]
        if old and observations:
            a=int(data['offsets'][frame-1]);columns=np.array([o.evidence['index']-a for o in observations],int)
            values=np.zeros((len(old),len(observations)),np.float64);increments=np.zeros_like(values)
            use_clear=bool(self.config.get('clear_identity_anchor',False));margin=float(self.config.get('prebirth_margin',.1))
            for i,(identity,entry) in enumerate(old):
                context=(entry.get('clear') or entry['latest']) if use_clear else entry['latest']
                scores,delta=scorer(frame,context,data)
                for j,column in enumerate(columns):
                    alternatives=np.delete(scores,column)
                    if scores[column]-(float(alternatives.max()) if len(alternatives) else 0.)<margin:continue
                    blocked=False
                    for peer in self.inner.tracked_stracks:
                        if peer.frame_id!=frame or peer.track_id not in self.emitted_identities:continue
                        if float(ious(observations[j].tlbr[None],peer.tlbr[None])[0,0])<=.7:continue
                        pair=tuple(sorted((identity,peer.track_id)))
                        if frame-self.co_visible.get(pair,-100000)>128:blocked=True;break
                    if not blocked:values[i,j]=scores[column];increments[i,j]=delta[column]
            self.statistics['prebirth_candidate_pairs']+=int((values>0).sum())
            if len(old)>1:
                ordered=np.sort(values,axis=0)
                values[:,ordered[-1]-ordered[-2]<margin]=0
            minimum=float(self.config.get('prebirth_probability',.9))
            matches,_,_=linear_assignment(1-values,1-minimum)
            for i,j in matches:
                if values[i,j]<minimum:continue
                identity,entry=old[i];track=entry['track'];observation=observations[j]
                index=observation.evidence['index']
                fresh=owners.get(index,[])
                if any(t.track_id in self.emitted_identities for t in fresh):raise ValueError('Attempted to rewrite an emitted identity')
                suppressed={t.track_id for t in fresh}
                for new in fresh:new.mark_removed()
                self.inner.tracked_stracks=[t for t in self.inner.tracked_stracks if t.track_id not in suppressed and t.track_id!=identity]
                self.inner.lost_stracks=[t for t in self.inner.lost_stracks if t.track_id not in suppressed and t.track_id!=identity]
                self.inner.removed_stracks=[t for t in self.inner.removed_stracks if t.track_id!=identity]
                known_removed={t.track_id for t in self.inner.removed_stracks}
                self.inner.removed_stracks.extend(t for t in fresh if t.track_id not in known_removed)
                gap=frame-entry['last_seen']
                track.mean,track.covariance=self.inner.kalman_filter.initiate(observation.to_xyah())
                track.re_activate(observation,frame,new_id=False)
                self.inner.tracked_stracks.append(track)
                rows=[row for row in rows if row[2] not in suppressed]
                rows.append((int(frame),track.tlwh.copy(),identity,track.score))
                self.trusted[(identity,index)]=float(values[i,j])
                self.statistics['prebirth_recoveries']+=1
                self.statistics[kind+'_memory_recoveries']+=1
                self.statistics['prebirth_long_recoveries']+=int(gap>25)
                self.statistics['prebirth_clear_recoveries']+=int(use_clear and entry.get('clear') is not None and entry['clear']['index']!=entry['latest']['index'])
                self.statistics['prebirth_IR_increment_links']+=int(getattr(self,'native_cue',self.cue)=='ir' and abs(increments[i,j])>1e-4)
                self.statistics['suppressed_unemitted_IDs']+=len(suppressed)
        return rows

    def update(self,frame,data):
        self.native_cache={}
        # Execute the original point matcher once. Both memory phases finish
        # before any fresh ID is returned, and never revisit an emitted ID.
        rows=CalibratedForegroundTracker.update(self,frame,data)
        if not self.config.get('prebirth_recovery',False):return rows
        if self.config.get('reference_memory',True):
            rows=self.recover_phase(frame,data,rows,self.reference_memory_scores,'RGB')
        if not self.config.get('disable_native_increment',False) and self.config.get('native_memory',True):
            rows=self.recover_phase(frame,data,rows,self.recovery_scores,'native')
        self.emitted_identities.update(row[2] for row in rows)
        self.remember_current(frame,data)
        return rows
