import numpy as np
from tools.mft_proposal_quality import (ProposalConfirmation,ForegroundView,current_features,
    FEATURE_NAMES,IR_FEATURES,QualityHead,promoted_scores,ProposalQualityTracker)
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.native_pyramid_stream_tracks import mot_line


def descriptors(n):
    a=np.zeros((n,512),np.float32);a[:,0]=1;return a


def test_confirmation_uses_original_proposals_and_resets_after_missing_observation():
    t=ProposalConfirmation();d=np.asarray([[0.,0.,12.,12.,.5]],np.float32)
    for f in [1,2,3]:np.testing.assert_array_equal(t.update('a',f,d),[f])
    np.testing.assert_array_equal(t.update('a',4,np.empty((0,5),np.float32)),[])
    np.testing.assert_array_equal(t.update('a',5,d),[1])
    np.testing.assert_array_equal(t.update('b',1,d),[1])
    assert d[0,4]==np.float32(.5)


def test_promotion_crosses_actual_ByteTrack_birth_threshold_and_preserves_other_rows():
    d=np.asarray([[0,0,12,12,.5],[30,0,42,12,.3],[60,0,72,12,.8]],np.float32)
    a,mask=promoted_scores(d,[.99,.99,.99],[3,3,3],.9,3)
    np.testing.assert_array_equal(mask,[1,0,0]);assert a[0,4]==np.float32(.75)
    np.testing.assert_array_equal(a[1:],d[1:]);np.testing.assert_array_equal(a[:,:4],d[:,:4])
    teacher=CandidateTeacher(.13)
    assert not teacher.update('s',1,d[:1],descriptors(1),descriptors(1))
    assert not teacher.update('s',2,a[:1],descriptors(1),descriptors(1))
    assert len(teacher.update('s',3,a[:1],descriptors(1),descriptors(1)))==1


def test_RGB_and_duplicate_heads_ignore_all_actual_IR_features():
    x=np.zeros((2,len(FEATURE_NAMES)),np.float32);x[:,7]=[.2,.6]
    for kind in ['RGB','duplicatedRGB']:
        m=QualityHead(np.zeros(len(FEATURE_NAMES)),np.ones(len(FEATURE_NAMES)),kind)
        m.coefficients[:]=np.linspace(-.3,.3,len(m.coefficients));before=m.probabilities(x)
        changed=x.copy();changed[:,IR_FEATURES]=777
        np.testing.assert_array_equal(m.probabilities(changed),before)


class ConstantHead:
    def __init__(self,value):self.value=value
    def probabilities(self,x):return np.full(len(x),self.value)


def run(kind,mode=None,future=False):
    v=np.zeros(512);v[0]=1;views={k:ForegroundView(np.zeros(512),v)for k in ['RGB','IR']}
    t=ProposalQualityTracker(dict(kind=kind,confirmation=3,quantile='q99',intervention=mode),views,
        {'RGB':ConstantHead(.99),'paired':ConstantHead(0.)},
        {k:{'q99':{'confirmed':.9,'all':.9}}for k in ['RGB','paired']},CandidateTeacher(.13))
    d=np.asarray([[0.,0.,12.,12.,.5]],np.float32);r=descriptors(1);rows=[]
    for f in range(1,9):
        current=d.copy();rgb=r.copy();ir=r.copy()
        if future and f>4:current[:,:4]+=500;rgb*=-1;ir*=-1
        source=[a.copy()for a in [current,rgb,ir]]
        rows.append(''.join(map(mot_line,t.update('s',f,current,rgb,ir))))
        for a,b in zip([current,rgb,ir],source):np.testing.assert_array_equal(a,b)
    return t,rows


def test_missing_IR_is_byte_exact_RGB_and_future_suffix_cannot_change_past():
    t,a=run('RGB');missing,b=run('paired','missing');paired,c=run('paired');future,f=run('RGB',future=True)
    assert a==b and a[:4]==f[:4]
    assert t.statistics['promoted']==6 and missing.statistics['promoted']==6 and paired.statistics['promoted']==0
    assert not any(c)and any(a)
    assert t.calibrated_packet_hash.hexdigest()==missing.calibrated_packet_hash.hexdigest()


def test_confirmation_removal_can_keep_the_selected_threshold_fixed():
    v=np.zeros(512);v[0]=1;views={k:ForegroundView(np.zeros(512),v)for k in ['RGB','IR']}
    d=np.asarray([[0.,0.,12.,12.,.5]],np.float32);r=descriptors(1)
    rows=[]
    for condition in ['all','confirmed']:
        t=ProposalQualityTracker(dict(kind='RGB',confirmation=0,quantile='q99',calibration_condition=condition),
            views,{'RGB':ConstantHead(.5)},{'RGB':{'q99':{'all':.2,'confirmed':.9}}},CandidateTeacher(.13))
        rows.append(t.update('s',1,d,r,r))
    assert len(rows[0])==1 and rows[1]==[]
