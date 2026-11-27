import numpy as np
import pytest
from tools.mft_identity_ledger import Memory,Ledger,LedgerTracker,LedgerLogistic,FEATURE_NAMES,IR_FEATURES
from tools.mft_identity_ledger_labels import OriginLabels


def obs(rgb=(1.,0.),ir=(0.,1.),box=(0.,0.,10.,20.),reliable=True):
    return dict(rgb=np.asarray(rgb,np.float32),ir=None if ir is None else np.asarray(ir,np.float32),
        box=np.asarray(box,np.float32),reliable=reliable,quality=.9,score=.9)


def ready_ledger():
    a=Ledger(.13)
    for f in [1,2,3]:a.commit(f,[obs()],{1:0},{1})
    return a


def test_fixed_origin_and_trusted_position_reject_physical_contamination():
    a=ready_ledger();m=a.memories[1];anchor=m.RGB.copy();box=m.box.copy()
    a.commit(4,[obs(rgb=(-1.,0.),box=(1000.,1000.,1010.,1020.))],{1:0},{1})
    assert m.trusted_frame==3 and m.support==3
    np.testing.assert_array_equal(m.RGB,anchor);np.testing.assert_array_equal(m.box,box)
    a.commit(5,[obs(box=(2.,0.,12.,20.))],{1:0},{1})
    assert m.trusted_frame==5 and m.support==4
    np.testing.assert_array_equal(m.RGB,anchor)


def test_all_candidates_include_active_roots_rotation_and_no_current_template():
    a=ready_ledger();x,e,ids=a.graph(4,[obs(box=(0.,0.,20.,10.)),obs(rgb=(-1.,0.))])
    assert ids==(1,)and e.shape==(2,1)and e.all()and a.memories[1].trusted_frame==3
    assert x[0,0,FEATURE_NAMES.index('absolute_log_diagonal_scale')]==0
    assert x[1,0,FEATURE_NAMES.index('RGB_anchor')]==-1
    with pytest.raises(AssertionError):a.graph(3,[obs()])


def test_frozen_origin_labels_do_not_become_history_majorities_or_unknown_negatives():
    o=OriginLabels()
    for f,label in enumerate([7,7,7],1):
        assert o.origin(1,f)is None;o.commit(f,[(1,0)],np.asarray([label]))
    assert o.origin(1,4)==7
    for f in range(4,20):o.commit(f,[],np.asarray([88]))
    assert o.origin(1,20)==7
    u=OriginLabels()
    for f,label in enumerate([7,-1,7],1):u.commit(f,[(2,0)],np.asarray([label]))
    assert u.origin(2,4)is None


def test_RGB_logits_ignore_all_IR_and_caller_mutation():
    m=LedgerLogistic(np.zeros(len(FEATURE_NAMES)),np.ones(len(FEATURE_NAMES)),'RGB')
    m.coefficients[:]=np.linspace(-.1,.1,len(m.coefficients));m.coefficients[list(IR_FEATURES)]=0
    x=np.zeros((2,len(FEATURE_NAMES)),np.float32);a=m.probabilities(x)
    x[:,IR_FEATURES]=1000;np.testing.assert_array_equal(m.probabilities(x),a)
    restored=LedgerLogistic.from_dict(m.to_dict());np.testing.assert_array_equal(restored.probabilities(x),a)


class FixedModel:
    def __init__(self,values):self.values=np.asarray(values)
    def probabilities(self,x):assert len(x)==len(self.values);return self.values.copy()


def test_delayed_swap_is_exclusive_and_missing_IR_exactly_falls_back_to_RGB():
    a=LedgerTracker(.13,kind='RGB',models={'RGB':FixedModel([.1,.99,.99,.1])},
        thresholds={'RGB':{'q99':.9},'paired':{'veto':.1}})
    a.ledger=Ledger(.13);a.ledger.bindings={1:1,2:2};a.pending={}
    a.statistics=dict(rebind_proposals=0,IR_vetoes=0,rebindings=0,displaced_owners=0,blocked_occupied_roots=0)
    x=np.zeros((2,2,len(FEATURE_NAMES)),np.float32);e=np.ones((2,2),bool)
    for f in [10,11]:
        a.remap(f,x,e,(1,2),{1:0,2:1},[obs(),obs()]);assert a.ledger.bindings=={1:1,2:2}
    a.remap(12,x,e,(1,2),{1:0,2:1},[obs(),obs()]);assert a.ledger.bindings=={1:2,2:1}
    assert a.statistics['rebindings']==2
    # A contradictory critic is ignored when there is no IR evidence.
    b=LedgerTracker(.13,kind='critic',models={'RGB':FixedModel([.1,.99,.99,.1]),'paired':FixedModel([0.,0.,0.,0.])},thresholds=a.thresholds)
    b.ledger=Ledger(.13);b.ledger.bindings={1:1,2:2};b.pending={};b.statistics={k:0 for k in a.statistics}
    for f in [10,11,12]:b.remap(f,x,e,(1,2),{1:0,2:1},[obs(ir=None),obs(ir=None)])
    assert b.ledger.bindings==a.ledger.bindings and b.statistics['IR_vetoes']==0


def test_occupied_root_is_preserved_without_confident_owner_contradiction():
    a=LedgerTracker(.13,kind='RGB',models={'RGB':FixedModel([.99,.6])},thresholds={'RGB':{'q99':.9}})
    a.ledger=Ledger(.13);a.ledger.bindings={1:1,2:2};a.pending={}
    a.statistics=dict(rebind_proposals=0,IR_vetoes=0,rebindings=0,displaced_owners=0,blocked_occupied_roots=0)
    x=np.zeros((2,1,len(FEATURE_NAMES)),np.float32);e=np.ones((2,1),bool)
    for f in [10,11,12]:a.remap(f,x,e,(2,),{1:0,2:1},[obs(),obs()])
    assert a.ledger.bindings=={1:1,2:2}and a.statistics['rebindings']==0
    assert a.statistics['blocked_occupied_roots']==1
