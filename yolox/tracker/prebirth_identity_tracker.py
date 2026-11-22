"""Consult past identities before the first output of a new identity."""
import numpy as np

from yolox.tracker.association_features import pair_features,cosine
from yolox.tracker.basetrack import TrackState
from yolox.tracker.matching import ious,linear_assignment
from yolox.tracker.strong_rgb_tracker import StrongRGBForegroundTracker


class PrebirthIdentityMixin:
    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        self.identity_archive={};self.emitted_identities=set()
        self.statistics.update(prebirth_recoveries=0,prebirth_long_recoveries=0,
            prebirth_clear_recoveries=0,prebirth_IR_increment_links=0,
            prebirth_candidate_pairs=0,suppressed_unemitted_IDs=0)

    def appearance_scores(self,frame,context,data):
        a,b=map(int,data['offsets'][frame-1:frame+1]);index=int(context['index'])
        if context['frame']>=frame or index>=a:raise ValueError('Identity memory must be strictly past')
        fields=['boxes','embedding','rgb_mask_embedding','rgb_hog','color','quality']
        if self.cue=='ir':fields+=['ir_mask_embedding','ir_hog']
        past={k:data[k][index:index+1] for k in fields};current={k:data[k][a:b] for k in fields}
        # Match the registered Train-long-bank missing-evidence regime exactly.
        past['point_quality']=np.zeros((1,2),np.float32);current['point_quality']=np.zeros((b-a,2),np.float32)
        modes=['rgb','bbox']+(['joint'] if self.cue=='ir' else [])
        matrices={(s,t):np.zeros((1,b-a),np.float32) for s in modes for t in modes}
        missing={s:np.zeros(1,np.float32) for s in modes}
        base,RGB_extra,reference=pair_features(past,current,matrices,missing,missing,8,'rgb')
        extra=pair_features(past,current,matrices,missing,missing,8,'ir')[1] if self.cue=='ir' else RGB_extra.copy()
        base[...,:3]=0;base[...,12:14]=0
        for value in [RGB_extra,extra,reference]:value[...,15:17]=0
        if self.cue=='ir':available=(past['quality'][:,1:2]>=.15)&(current['quality'][None,:,1]>=.15)
        else:available=np.ones((1,b-a),bool)
        probability,increment=self.score_features(base,RGB_extra,extra,reference,available)
        return probability[0],increment[0]

    def recovery_scores(self,frame,context,data):
        lag=frame-context['frame'];a,b=map(int,data['offsets'][frame-1:frame+1])
        probability=increment=None
        if lag in self.config.get('lags',[1,2,4,8]):
            start,p,delta,vote,good,total=self.predict_pairs(frame,lag);row=context['index']-start
            if 0<=row<len(p) and good[row]>=4 and good[row]/max(float(total[row]),1.)>=.25 and vote[row].max(initial=0)>=.15:
                probability=p[row].copy();increment=delta[row].copy()
                probability[vote[row]<.15]=0
        if probability is None:probability,increment=self.appearance_scores(frame,context,data)
        similarity=cosine(data['embedding'][context['index']:context['index']+1],data['embedding'][a:b])[0]
        probability=np.asarray(probability,np.float64).copy()
        probability[similarity<self.config.get('prebirth_RGB_cosine',.5)]=0
        return probability,np.asarray(increment,np.float32)

    def remember_current(self,frame,data):
        a,b=map(int,data['offsets'][frame-1:frame+1]);boxes=data['boxes'][a:b]
        high=np.flatnonzero(data['scores'][a:b]>.6)
        for track in self.inner.tracked_stracks:
            if track.frame_id!=frame or track.track_id not in self.emitted_identities:continue
            latest=track.evidence
            entry=self.identity_archive.get(track.track_id,dict(track=track,clear=None))
            entry.update(track=track,latest=latest,last_seen=frame)
            index=latest['index'];column=index-a
            others=high[high!=column]
            separate=not len(others) or float(ious(boxes[column:column+1],boxes[others]).max())<=.3
            if separate and data['quality'][index,0]>=.15:entry['clear']=latest
            self.identity_archive[track.track_id]=entry
        retention=int(self.config.get('identity_retention',25))
        self.identity_archive={identity:entry for identity,entry in self.identity_archive.items() if frame-entry['last_seen']<=retention}

    def update(self,frame,data):
        rows=super().update(frame,data)
        if not self.config.get('prebirth_recovery',False):return rows
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
                scores,delta=self.recovery_scores(frame,context,data)
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
                self.statistics['prebirth_long_recoveries']+=int(gap>25)
                self.statistics['prebirth_clear_recoveries']+=int(use_clear and entry.get('clear') is not None and entry['clear']['index']!=entry['latest']['index'])
                self.statistics['prebirth_IR_increment_links']+=int(getattr(self,'native_cue',self.cue)=='ir' and abs(increments[i,j])>1e-4)
                self.statistics['suppressed_unemitted_IDs']+=len(suppressed)
        self.emitted_identities.update(row[2] for row in rows)
        self.remember_current(frame,data)
        return rows


class PrebirthStrongRGBTracker(PrebirthIdentityMixin,StrongRGBForegroundTracker):
    pass
