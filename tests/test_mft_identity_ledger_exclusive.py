import numpy as np
from tools.mft_identity_ledger import Ledger,LedgerTracker,FEATURE_NAMES
from tools.mft_identity_ledger_exclusive import ExclusiveLedgerTracker


class Scores:
    def __init__(self,p):self.p=np.asarray(p)
    def probabilities(self,x):assert len(x)==len(self.p);return self.p.copy()


def tracker(cls,p,bindings):
    t=cls(.13,kind='RGB',models={'RGB':Scores(p)},thresholds={'RGB':{'q99':.9}})
    t.ledger=Ledger(.13);t.ledger.bindings=bindings.copy();t.pending={}
    t.statistics=dict(rebind_proposals=0,IR_vetoes=0,rebindings=0,displaced_owners=0,blocked_occupied_roots=0)
    return t


def test_rejected_dependency_cannot_turn_unscored_owner_into_a_negative():
    e=np.asarray([[1,1,0],[1,0,1],[1,0,1]],bool);x=np.zeros((3,3,len(FEATURE_NAMES)),np.float32)
    p=[.1,.99,.1,.99,.1,.6];assign={1:0,2:1,3:2};obs=[dict(reliable=True)]*3
    old=tracker(LedgerTracker,p,{1:1,2:2,3:3});fixed=tracker(ExclusiveLedgerTracker,p,{1:1,2:2,3:3})
    for f in [10,11,12]:
        old.remap(f,x,e,(1,2,3),assign,obs);fixed.remap(f,x,e,(1,2,3),assign,obs)
    assert old.ledger.bindings[1]==2 and old.ledger.bindings[2]>1_000_000_000
    assert fixed.ledger.bindings=={1:1,2:2,3:3}and fixed.statistics['rebindings']==0
    assert fixed.statistics['unscored_owner_frame_rollbacks']==1


def test_confident_exclusive_cycle_is_unchanged_and_remains_past_only():
    e=np.ones((2,2),bool);x=np.zeros((2,2,len(FEATURE_NAMES)),np.float32);assign={1:0,2:1};obs=[dict(reliable=True)]*2
    old=tracker(LedgerTracker,[.1,.99,.99,.1],{1:1,2:2});fixed=tracker(ExclusiveLedgerTracker,[.1,.99,.99,.1],{1:1,2:2})
    for f in [10,11]:
        old.remap(f,x,e,(1,2),assign,obs);fixed.remap(f,x,e,(1,2),assign,obs)
        assert fixed.ledger.bindings==old.ledger.bindings=={1:1,2:2}
    for t in [old,fixed]:t.remap(12,x,e,(1,2),assign,obs)
    assert fixed.ledger.bindings==old.ledger.bindings=={1:2,2:1}
    assert fixed.statistics['rebindings']==old.statistics['rebindings']==2
    assert fixed.statistics.get('unscored_owner_frame_rollbacks',0)==0
