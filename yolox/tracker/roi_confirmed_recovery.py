"""Keep the original observation graph; use crop evidence only for old-ID rescue."""
import numpy as np

from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker
from yolox.tracker.weak_recovery_tracker import WeakRecoveryMixin


class _ConfirmAfterPrimary:
    def update(self,frame,data):
        rows=super().update(frame,data)
        if self.config.get('disable_ROI_confirmation',False):
            self.created=[]
        else:
            support=data['ROI_confirmation']
            self.created=[observation for observation in self.created
                          if support[observation.evidence['index']]>=.7]
        return rows


class ROIConfirmedRecovery(WeakRecoveryMixin,_ConfirmAfterPrimary,CalibratedForegroundTracker):
    def __init__(self,config):
        if config['calibration_cue']!='rgb':raise ValueError('The complete frozen primary must be RGB')
        super().__init__(config)
        auxiliary=dict(config)
        auxiliary.update(association_checkpoint=config['auxiliary_checkpoint'],calibration_cue=config['auxiliary_cue'])
        self.auxiliary=CalibratedForegroundTracker(auxiliary)
        self.auxiliary_metadata=self.auxiliary.calibration_metadata
        self.statistics.update(ROI_confirmed_sources=0,added_cue_proposals=0,added_IR_proposals=0)

    def update(self,frame,data):
        self.auxiliary.data=data;self.auxiliary.pair_cache={}
        return super().update(frame,data)

    def weak_scores(self,frame,context,data):
        reference=WeakRecoveryMixin.weak_scores(self,frame,context,data)
        if reference is None:return None
        reference_score,_=reference
        if self.auxiliary.cue=='ir':
            a,b=map(int,data['offsets'][frame-1:frame+1]);quality=data['foreground_transport']['quality']
            if (self.config.get('disable_IR_evidence',False)or quality[context['index'],1]<.15
                    or not np.any(quality[a:b,1]>=.15)):
                return reference_score,np.zeros_like(reference_score,bool)
        candidate=WeakRecoveryMixin.weak_scores(self.auxiliary,frame,context,data)
        if candidate is None:return reference_score,np.zeros_like(reference_score,bool)
        additional_score,changed=candidate
        eligible=additional_score>=reference_score+.05
        if self.auxiliary.cue=='ir':
            a,b=map(int,data['offsets'][frame-1:frame+1]);quality=data['foreground_transport']['quality']
            eligible&=(quality[context['index'],1]>=.15)&(quality[a:b,1]>=.15)&changed
        result=np.where(eligible,additional_score,reference_score)
        self.statistics['added_cue_proposals']+=int(eligible.sum())
        if self.auxiliary.cue=='ir':self.statistics['added_IR_proposals']+=int(eligible.sum())
        return result,eligible&(self.auxiliary.cue=='ir')
