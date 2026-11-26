import numpy as np
from yolox.tracker import matching
from yolox.tracker.basetrack import TrackState
from tools.mft_selective_protection import SelectiveProtection,policies
from tools.mft_candidate_risk_association import RiskAssociation,policies as parent_policies
from tools.mft_candidate_risk_features import FEATURE_NAMES
from tools.mft_candidate_risk_model import RiskLogistic
from tools.native_pyramid_stream_tracks import mot_line
from tests.test_mft_candidate_risk_features_labels import track,detection
from tests.test_mft_candidate_risk_teacher_stream import packet


def models():
    result={k:RiskLogistic(np.zeros(26),np.ones(26),k)for k in ['RGB','paired']}
    for m in result.values():m.coefficients[FEATURE_NAMES.index('RGB_cosine')]=12;m.coefficients[-1]=-6
    return result


def fixture(index=5,threshold=.99):
    m=models();t=SelectiveProtection(policies()[index],m,dict(RGB=.95,paired=.95),
        dict(RGB=threshold,paired=threshold),.13137420405549355);t.reset('artificial')
    tracks=[track((1,0),identity=1),track((0,1),identity=2)]
    for v in tracks:v.state=TrackState.Tracked
    dets=[detection((0,1)),detection((1,0))]
    return t,tracks,dets,np.asarray([[.1,.6],[.6,.1]])


def test_static_control_and_insufficient_score_are_exact_original_protection():
    for index,threshold in [(1,.99),(5,1.)]:
        t,tracks,dets,base=fixture(index,threshold)
        parent=RiskAssociation(parent_policies()[3],t.models,t.thresholds,t.speed_upper);parent.reset('artificial')
        np.testing.assert_array_equal(t.cost(tracks,dets,base,5,'primary'),parent.cost(tracks,dets,base,5,'primary'))
        assert t.statistics['unlocked_geometric_pairs']==0


def test_strong_mutual_preference_can_revise_two_conflicting_protected_pairs():
    t,tracks,dets,base=fixture();before=base.copy();result=t.cost(tracks,dets,base,5,'primary')
    assert set(map(tuple,matching.linear_assignment(result,thresh=.9)[0]))=={(0,1),(1,0)}
    assert t.statistics['unlocked_geometric_pairs']==2
    assert t.statistics['selective_vs_static_assignment_changed_frames']==1
    np.testing.assert_array_equal(base,before)
    np.testing.assert_array_equal(t.cost(tracks,dets,base,5,'secondary'),base)


def test_below_threshold_competitor_still_prevents_insufficient_probability_margin():
    t,tracks,dets,base=fixture()
    class CloseScores:
        kind='paired'
        def probabilities(self,values):return np.where(values[:,FEATURE_NAMES.index('RGB_cosine')]>.5,.995,.96)
    t.models['paired']=CloseScores()
    np.testing.assert_array_equal(t.cost(tracks,dets,base,5,'primary'),base)
    assert t.statistics['unlocked_geometric_pairs']==0


def test_unscored_protected_baseline_remains_safe_from_competing_old_template():
    t,tracks,dets,base=fixture();tracks[1].accepted=2
    np.testing.assert_array_equal(t.cost(tracks,dets,base,5,'primary'),base)
    assert t.statistics['unlocked_geometric_pairs']==0


def test_missing_ir_fallback_and_future_suffix_are_exact_at_output():
    def run(length,corrupt=False):
        m=models();a=SelectiveProtection(policies()[5],m,dict(RGB=.95,paired=.95),dict(RGB=.99,paired=.99),.13)
        b=SelectiveProtection(policies()[4],m,dict(RGB=.95,paired=.95),dict(RGB=.99,paired=.99),.13)
        text=[]
        for f in range(1,length+1):
            d,r,ir=packet(f)
            if corrupt and f>8:d[:,:4]+=777;r=-r;ir=-ir
            actual=''.join(map(mot_line,a.update('artificial',f,d,r,None)))
            assert actual==''.join(map(mot_line,b.update('artificial',f,d,r,ir)))
            text.append(actual)
        return text
    assert run(12)[:8]==run(12,True)[:8]
