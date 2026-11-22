import numpy as np

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.consensus_identity_memory import ConsensusIdentityMemoryTracker
from test_native_query_tracker import IdentityContrast
from test_prebirth_identity import observations_at


def run(anchor_probability):
    data=observations_at({1:[[20,20,50,40]],2:[[22,20,52,40]],3:[[24,20,54,40]],
                          5:[[220,20,250,40]]},5)
    BaseTrack._count=0
    tracker=ConsensusIdentityMemoryTracker(dict(native_cue='ir',query_ranking=False,
        prebirth_recovery=True,reference_memory=False,identity_retention=25,
        counterfactual_recovery_maximum_lag=8,consensus_anchor_separation=2),
        model=AssociationCalibrator(),native_model=IdentityContrast(),prior=dict(t=0,a=0,b=0,c=0))
    used=[]
    def posterior(frame,context):
        used.append(context['frame'])
        p=.99 if context['frame']==3 else anchor_probability
        return dict(start=context['index'],probability=np.array([[p]],np.float32),
                    null=np.array([1-p],np.float32),delta=np.array([[4.]],np.float32))
    tracker.query_posterior=posterior
    for frame in range(1,6):rows=tracker.update(frame,data)
    return tracker,rows,used


def test_a_second_distinct_past_observation_can_confirm_complementary_identity():
    tracker,rows,used=run(.99)
    assert [row[2] for row in rows]==[1]
    assert used==[3,1]
    assert tracker.statistics['native_memory_recoveries']==1
    assert tracker.statistics['consensus_confirmed_pairs']==1


def test_one_overconfident_native_observation_cannot_override_an_inconsistent_history():
    tracker,rows,used=run(.4)
    assert rows==[]
    assert used==[3,1]
    assert tracker.statistics['native_memory_recoveries']==0
    assert tracker.statistics['consensus_rejected_pairs']==1
