import numpy as np
from tools.mft_archive_evaluation import ArchiveTracker,policies,matched_case,interventions,intervene
from tools.native_pyramid_stream_tracks import mot_line
from tests.test_mft_archive_reasoning import high_models,packet


def test_fixed_cases_have_complete_RGB_paired_shape_and_threshold_controls():
    p=policies();assert len(p)==11 and len({c['id']for c in p})==11
    for c in p[3:]:
        assert matched_case(matched_case(c,'RGB'),c['kind'])==c
        a=interventions(c);assert len(a)==2 and all(v['kind']=='paired'for v in a)
        assert all(v['shape']==c['shape']and v['quantile']==c['quantile']for v in a)


def test_shape_keeps_identity_decisions_and_missing_IR_full_episode_equals_RGB():
    cases=[next(c for c in policies()if c['id']==n)for n in ['AR_RGB_q99','AR_paired_q99','AR_paired_q99_shape']]
    thresholds={k:dict(q99=.99,q999=.999)for k in ['RGB','paired']};models=high_models()
    trackers=[ArchiveTracker(c,models,thresholds,.13)for c in cases]
    missing=ArchiveTracker(interventions(cases[1])[0],models,thresholds,.13)
    outputs=[[],[],[]]
    for frame in range(1,49):
        d,r,a=packet(frame)
        rows=[t.update('one',frame,d,r,a)for t in trackers]
        assert ''.join(map(mot_line,missing.update('one',frame,d,r,a)))==''.join(map(mot_line,rows[0]))
        assert [(v[0],v[2],v[3])for v in rows[1]]==[(v[0],v[2],v[3])for v in rows[2]]
        for store,values in zip(outputs,rows):store.extend(values)
    assert trackers[1].statistics['archive_identity_links']==1
    assert outputs[1][-1][2]==outputs[2][-1][2]==1


def test_roll_intervention_uses_only_current_frame_and_never_mutates_bank():
    c=interventions(policies()[4])[1];a=np.arange(12,dtype=np.float32).reshape(3,4);before=a.copy()
    result=intervene(a,c);np.testing.assert_array_equal(result,before[[2,0,1]])
    np.testing.assert_array_equal(a,before)
    result[:]=-1;np.testing.assert_array_equal(a,before)
