import numpy as np
from yolox.tracker.basetrack import TrackState
from tools.run_mft_dev_candidate_graph_diagnosis import graph_snapshot,classify_event
from tests.test_mft_candidate_risk_features_labels import track,detection


def fixture(state=TrackState.Tracked,frame=5):
    tracks=[track((1,0),identity=1),track((0,1),identity=2)]
    for t in tracks:t.state=state
    dets=[detection((1,0)),detection((0,1))]
    for i,d in enumerate(dets):d.observation['proposal_index']=i
    return tracks,dets,dict(previous_prediction_ID='1',current_prediction_ID='3'),{3:1},{0:.9,1:.9},frame


def test_hard_protection_masks_a_different_candidate_on_the_same_track_row():
    t,d,e,a,s,f=fixture();graph=graph_snapshot(t,d,np.asarray([[.1,.6],[.6,.1]]),f,.13)
    result=classify_event(e,a,s,graph)
    assert result['reason']=='hard_geometric_protection_masks_edge'
    assert not result['passes_nonlearned_policy_masks']and result['protected_previous_track_row']


def test_reachable_tracked_gap1_is_blocked_but_lost_gap2_still_needs_learned_gate():
    t,d,e,a,s,f=fixture();graph=graph_snapshot(t,d,np.ones((2,2)),f,.13)
    assert classify_event(e,a,s,graph)['reason']=='tracked_gap1_edge_excluded_by_lost_only_rescue'
    for v in t:v.state=TrackState.Lost
    graph=graph_snapshot(t,d,np.ones((2,2)),6,.13);r=classify_event(e,a,s,graph)
    assert r['passes_nonlearned_policy_masks']and r['reason']=='lost_edge_still_requires_learned_threshold_mutual_margin'
    assert not r['learned_score_or_model_used']


def test_secondary_proposal_does_not_become_a_primary_risk_candidate():
    t,d,e,a,s,f=fixture();graph=graph_snapshot(t,d,np.full((2,2),.7),f,.13)
    a={3:99};s[99]=.45
    result=classify_event(e,a,s,graph)
    assert result['reason']=='current_output_uses_nonprimary_proposal'and not result['passes_nonlearned_policy_masks']


def test_insufficient_past_memory_does_not_claim_appearance_recoverability():
    t,d,e,a,s,f=fixture();t[0].accepted=1
    graph=graph_snapshot(t,d,np.full((2,2),.7),f,.13)
    r=classify_event(e,a,s,graph)
    assert r['reason']=='past_template_history_unavailable'and not r['passes_nonlearned_policy_masks']
