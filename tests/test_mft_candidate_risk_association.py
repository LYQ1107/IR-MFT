from types import SimpleNamespace
import numpy as np
from yolox.tracker import matching
from yolox.tracker.basetrack import TrackState
from tools.mft_candidate_risk_association import RiskAssociation,policies
from tools.mft_candidate_risk_model import RiskLogistic
from tools.mft_candidate_risk_features import FEATURE_NAMES
from tools.native_pyramid_stream_tracks import mot_line
from tests.test_mft_candidate_risk_teacher_stream import packet
from tests.test_mft_candidate_risk_features_labels import track,detection


def models():
    result={k:RiskLogistic(np.zeros(26),np.ones(26),k)for k in ['RGB','paired']}
    for model in result.values():
        model.coefficients[FEATURE_NAMES.index('RGB_cosine')]=6
        model.coefficients[-1]=-2
    return result


def tracker(index=3):
    return RiskAssociation(policies()[index],models(),dict(RGB=.95,paired=.95),.13137420405549355)


def test_learned_primary_preference_changes_ambiguous_geometry_and_counts_actual_edges():
    t=tracker();t.reset('synthetic')
    tracks=[track((1,0)),track((0,1))]
    for v in tracks:v.state=TrackState.Tracked
    dets=[detection((1,0)),detection((0,1))]
    base=np.asarray([[.55,.5],[.5,.55]])
    result=t.cost(tracks,dets,base,5,'primary')
    assert set(map(tuple,matching.linear_assignment(base,thresh=.9)[0]))=={(0,1),(1,0)}
    assert set(map(tuple,matching.linear_assignment(result,thresh=.9)[0]))=={(0,0),(1,1)}
    assert t.statistics['actual_changed_cost_entries']==4
    assert t.statistics['primary_assignment_changed_frames']==1
    np.testing.assert_array_equal(t.cost(tracks,dets,base,5,'secondary'),base)


def test_confident_continuous_geometry_is_protected_from_appearance():
    t=tracker();t.reset('synthetic')
    tracks=[track((0,1)),track((1,0))]
    for v in tracks:v.state=TrackState.Tracked
    dets=[detection((1,0)),detection((0,1))]
    base=np.asarray([[.1,.6],[.6,.1]])
    np.testing.assert_array_equal(t.cost(tracks,dets,base,5,'primary'),base)


def test_lost_edge_rescue_requires_calibrated_mutual_unique_reachable_evidence():
    t=tracker();t.reset('synthetic')
    tracks=[track((1,0)),track((0,1))]
    for v in tracks:v.state=TrackState.Lost
    dets=[detection((1,0)),detection((0,1))]
    base=np.ones((2,2))
    result=t.cost(tracks,dets,base,6,'primary')
    assert (result.diagonal()<.9).all()and result[0,1]==result[1,0]==1
    assert t.statistics['new_reachable_edge_costs']==2
    t.thresholds={'RGB':1.,'paired':1.}
    np.testing.assert_array_equal(t.cost(tracks,dets,base,6,'primary'),base)


def test_missing_ir_is_exact_independent_rgb_fallback_and_shape_keeps_identity_scores():
    paired=tracker(3);rgb=tracker(2)
    shape=RiskAssociation(dict(policies()[4]),models(),dict(RGB=.95,paired=.95),.13137420405549355)
    for f in range(1,10):
        d,r,a=packet(f)
        x=paired.update('synthetic',f,d,r,None)
        y=rgb.update('synthetic',f,d,r,a)
        assert ''.join(map(mot_line,x))==''.join(map(mot_line,y))
        z=shape.update('synthetic',f,d,r,None)
        assert [(v[0],v[2],v[3])for v in x]==[(v[0],v[2],v[3])for v in z]
