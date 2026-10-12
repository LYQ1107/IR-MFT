"""Observable assignment-margin refusal, independent of geometry updates/GT."""
import numpy as np


class ReliableWritesMixin:
    def __init__(self,*args,write_policy=None,**kwargs):
        self.write_policy=write_policy
        self.write_counts=dict(accepted_existing=0,rejected_existing=0,newborn_initializations=0,rejected_low_or_unscored=0,rejected_confidence=0,rejected_margin=0)
        self.last_write_decisions=[];self._write_evidence={}
        if write_policy is not None:
            required=['minimum_confidence','minimum_row_margin','minimum_column_margin']
            if any(k not in write_policy or not np.isfinite(write_policy[k]) or write_policy[k]<0 for k in required) or write_policy['minimum_confidence']>1:
                raise ValueError('Finite observable write-policy thresholds required')
        super().__init__(*args,**kwargs)

    def _guard_track(self,track):
        if getattr(track,'_cfp_reliable_write_wrapped',False):return
        original=track.write_identity
        def guarded(frame_id,ema):
            if not track.pending:return False
            evidence=self._write_evidence.get((track.track_id,track.current_proposal_index))
            reason='accepted';accepted=True
            if evidence is None or evidence['frame']!=frame_id:
                accepted=False;reason='low_or_unscored'
            elif track.score<self.write_policy['minimum_confidence']:
                accepted=False;reason='confidence'
            elif evidence['row_margin']<self.write_policy['minimum_row_margin'] or evidence['column_margin']<self.write_policy['minimum_column_margin']:
                accepted=False;reason='margin'
            decision=dict(frame=int(frame_id),track_id=int(track.track_id),proposal_index=track.current_proposal_index,accepted=accepted,written=False,reason=reason,confidence=float(track.score),row_margin=evidence['row_margin'] if evidence else None,column_margin=evidence['column_margin'] if evidence else None)
            self.last_write_decisions.append(decision)
            if not accepted:
                # Geometry has already updated; retain all past identity state.
                track.pending=False;self.write_counts['rejected_existing']+=1;self.write_counts['rejected_'+reason]+=1
                return False
            written=original(frame_id,ema)
            decision['written']=bool(written)
            if written:self.write_counts['accepted_existing']+=1
            return written
        track.write_identity=guarded;track._cfp_reliable_write_wrapped=True

    def association_cost(self,tracks,detections):
        costs=super().association_cost(tracks,detections)
        if self.write_policy is None:return costs
        for i,track in enumerate(tracks):
            self._guard_track(track)
            null=self.args.match_thresh if track.is_activated else .7
            for j,detection in enumerate(detections):
                # NULL and every native competing edge are alternatives.
                row_other=np.delete(costs[i],j);column_other=np.delete(costs[:,j],i)
                row_best=min(null,float(row_other.min())) if len(row_other) else null
                column_best=min(null,float(column_other.min())) if len(column_other) else null
                self._write_evidence[(track.track_id,detection.current_proposal_index)]=dict(frame=self.frame_id,row_margin=row_best-float(costs[i,j]),column_margin=column_best-float(costs[i,j]))
        return costs

    def update(self,*args,**kwargs):
        self._write_evidence={};self.last_write_decisions=[]
        result=super().update(*args,**kwargs)
        if self.write_policy is not None:
            for track in self.tracked_stracks:
                if not getattr(track,'_cfp_reliable_write_wrapped',False):
                    # A confident native newborn needs an initial identity;
                    # no prior track state exists to compute an assignment margin.
                    if track.last_identity_frame==self.frame_id:
                        self.write_counts['newborn_initializations']+=1
                        self.last_write_decisions.append(dict(frame=self.frame_id,track_id=int(track.track_id),proposal_index=track.current_proposal_index,accepted=True,written=True,reason='newborn_initialization',confidence=float(track.score),row_margin=None,column_margin=None))
                    self._guard_track(track)
        return result
