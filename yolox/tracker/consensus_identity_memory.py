"""Require a second past observation before accepting complementary identity."""
import numpy as np

from yolox.tracker.counterfactual_query_tracker import CounterfactualQueryTracker
from yolox.tracker.ordered_identity_memory import OrderedIdentityMemoryTracker


class ConsensusIdentityMemoryTracker(OrderedIdentityMemoryTracker):
    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        self.statistics.update(consensus_queries=0,consensus_confirmed_pairs=0,
                               consensus_rejected_pairs=0,consensus_missing_anchor=0)

    def recovery_scores(self,frame,context,data):
        first,increment=CounterfactualQueryTracker.recovery_scores(self,frame,context,data)
        minimum=float(self.config.get('prebirth_probability',.9))
        if not np.any(first>=minimum):return first,increment
        entry=next((e for e in self.identity_archive.values() if e['latest']['index']==context['index']),None)
        if entry is None:raise ValueError('Complementary memory source is not a retained identity')
        maximum_lag=int(self.config.get('consensus_maximum_lag',8))
        separation=int(self.config.get('consensus_anchor_separation',2))
        quality_column=1 if self.native_cue=='ir' else 0
        anchors=[observation for observation in entry['track'].object_observations
                 if context['frame']-observation['frame']>=separation
                 and 0<frame-observation['frame']<=maximum_lag
                 and observation['index']<context['index']
                 and data['quality'][observation['index'],quality_column]>=.15]
        self.statistics['consensus_queries']+=1
        if not anchors:
            self.statistics['consensus_missing_anchor']+=1
            return np.zeros_like(first),np.zeros_like(increment)
        anchor=max(anchors,key=lambda observation:observation['frame'])
        second,_=CounterfactualQueryTracker.recovery_scores(self,frame,anchor,data)
        confirmed=np.minimum(first,second)
        self.statistics['consensus_confirmed_pairs']+=int((confirmed>=minimum).sum())
        self.statistics['consensus_rejected_pairs']+=int(((first>=minimum)&(confirmed<minimum)).sum())
        return confirmed,increment
