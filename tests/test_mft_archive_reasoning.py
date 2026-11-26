from types import SimpleNamespace
import numpy as np
import pytest
from tools.mft_archive_reasoning import (ArchiveLinker,ArchiveTrainCollector,ArchiveLogistic,
    FEATURE_NAMES,IR_FEATURES,choose_links,features)
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line


def packet(frame,variant=False):
    if 9<=frame<=40:return np.empty((0,5),np.float32),np.empty((0,2),np.float16),np.empty((0,2),np.float16)
    d=np.asarray([[0,0,20,20,.95]],np.float32)
    r=np.asarray([[1,0]],np.float16);a=np.asarray([[0,1]],np.float16)
    if variant and frame>45:d[:,:4]+=777;r*=-1;a*=-1
    return d,r,a


def gt(frame,label=1):
    d=packet(frame)[0]
    return np.asarray([[frame,label,b[0],b[1],20,20,1,1,1]for b in d],np.float64).reshape(-1,9)


def high_models():
    result={k:ArchiveLogistic(np.zeros(26),np.ones(26),k)for k in ['RGB','paired']}
    for m in result.values():
        m.coefficients[FEATURE_NAMES.index('log_new_reliable_count')]=20
        m.coefficients[-1]=-20
    return result


def test_archive_only_after_real_pool_departure_and_observer_cannot_change_teacher():
    calls=[]
    def observe(x,mask,*args):
        calls.append(args[-1]);x[:]=-99;mask[:]=False
    linked=ArchiveLinker(CandidateTeacher(.13),.13,observer=observe)
    original=IndependentByteTrack()
    for f in range(1,48):
        d,r,a=packet(f);rows=linked.update('one',f,d,r,a)
        assert ''.join(map(mot_line,rows))==''.join(map(mot_line,original.update(d,f)))
        if 9<=f<=33:assert not linked.closed  # Actual lifetime25, not a guessed removal time.
        pool={t.track_id for t in linked.inner.core.tracked_stracks+linked.inner.core.lost_stracks}
        assert all(v['internal']not in pool for v in linked.closed.values())
    assert calls and linked.statistics['archived_templates']==1
    assert linked.statistics['identity_links']==0


def test_link_changes_only_current_and_future_ids_and_missing_IR_is_exact_RGB():
    models=high_models();thresholds=dict(RGB=.99,paired=.99)
    def replay(kind,missing=False,variant=False):
        t=ArchiveLinker(CandidateTeacher(.13),.13,kind=kind,models=models,thresholds=thresholds)
        text=[];ids=[]
        for f in range(1,49):
            d,r,a=packet(f,variant);rows=t.update('one',f,d,r,None if missing else a)
            text.append(''.join(map(mot_line,rows)));ids.append([row[2]for row in rows])
        return text,ids,t
    text,ids,t=replay('paired')
    assert ids[41]==[2]and ids[42]==[1]  # First output is immutable; second public view resolves it.
    assert t.statistics['identity_links']==t.statistics['delayed_identity_links']==1
    assert replay('paired',missing=True)[0]==replay('RGB',missing=True)[0]
    assert text[:45]==replay('paired',variant=True)[0][:45]
    d,r,a=packet(1);assert ''.join(map(mot_line,t.update('two',1,d,r,a)))==text[0]


def test_archive_labels_change_only_targets_and_Dev_Test_rejected():
    batches=[[],[]];streams=[ArchiveTrainCollector(.13,b.append)for b in batches]
    output=[[],[]]
    for f in range(1,48):
        for k,s in enumerate(streams):
            rows=s.step('one',f,*packet(f),gt(f,2 if k and f>=41 else 1),split='train')
            output[k].extend(map(mot_line,rows))
    assert output[0]==output[1]and len(batches[0])==len(batches[1])>0
    for a,b in zip(*batches):
        np.testing.assert_array_equal(a.x,b.x);assert a.y.sum()==1 and b.y.sum()==0
    for split in ['dev','test']:
        with pytest.raises(AssertionError):streams[0].step('one',48,*packet(48),gt(48),split=split)


def test_weak_competitors_and_mutuality_prevent_ambiguous_archive_links():
    p=np.asarray([[.99,.97],[.98,.1]])
    assert choose_links(p,np.ones_like(p,bool),.985)==[]
    assert choose_links(np.asarray([[.99,.8],[.7,.98]]),np.ones_like(p,bool),.95)==[(0,0),(1,1)]
    assert choose_links(np.asarray([[.99,.999]]),np.asarray([[True,False]]),.98)==[(0,0)]


def test_RGB_coefficients_and_features_have_no_IR_path_and_geometry_is_bounded():
    x=np.arange(52,dtype=np.float32).reshape(2,26)/20
    a=ArchiveLogistic(np.zeros(26),np.ones(26),'RGB');a.coefficients[:]=.7
    changed=x.copy();changed[:,IR_FEATURES]=-123
    np.testing.assert_array_equal(a.probabilities(x),a.probabilities(changed))
    np.testing.assert_array_equal(a.design(x)[:,IR_FEATURES],0)
    b=ArchiveLogistic.from_dict(a.to_dict());np.testing.assert_array_equal(a.probabilities(x),b.probabilities(x))
    memory=dict(internal=1,rgb=np.asarray([1.,0.]),ir=np.asarray([0.,1.]),box=np.asarray([0.,0.,20.,20.]),
        last_frame=8,template_frame=8,quality=.9,accepted=8,score=.95)
    current=dict(memory,last_frame=42,template_frame=42,first_public_frame=41)
    values,valid=features([current],[memory],42,.13);assert valid[0,0]and np.isfinite(values).all()
    far=dict(current,box=current['box']+10000)
    assert not features([far],[memory],42,.13)[1].any()
    assert not features([dict(current,last_frame=300,template_frame=300)],[memory],300,.13)[1].any()
