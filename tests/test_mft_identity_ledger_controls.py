import numpy as np
from tools.mft_identity_ledger import LedgerTracker,FEATURE_NAMES
from tools.native_pyramid_stream_tracks import mot_line


class AppearanceModel:
    def probabilities(self,x):return .05+.9*(np.asarray(x)[:,FEATURE_NAMES.index('RGB_anchor')]+1)/2


class VetoModel:
    def probabilities(self,x):return np.zeros(len(x),np.float64)


def run(kind,intervention=None):
    t=LedgerTracker(.13,kind=kind,models={'RGB':AppearanceModel(),'paired':VetoModel()},
        thresholds={'RGB':{'q99':.9},'paired':{'veto':.1}},intervention=intervention)
    d=np.asarray([[0.,0.,12.,12.,.9],[10.,0.,22.,12.,.9]],np.float32)
    rgb=np.eye(2,dtype=np.float32);ir=rgb.copy();history=[];boxes=[]
    for frame in range(1,13):
        current=rgb if frame<7 else rgb[::-1].copy()
        before=[d.copy(),current.copy(),ir.copy()]
        rows=t.update('synthetic',frame,d,current,ir)
        history.append(''.join(map(mot_line,rows)))
        boxes.append([(tuple(box),score)for f,box,identity,score in rows])
        for a,b in zip([d,current,ir],before):np.testing.assert_array_equal(a,b)
    return t,history,boxes


def test_real_tracker_rebinds_after_three_frames_without_geometry_feedback():
    t,history,boxes=run('RGB')
    assert t.statistics['rebindings']==2
    assert [int(v.split(',')[1])for v in history[7].splitlines()]==[1,2]
    assert [int(v.split(',')[1])for v in history[8].splitlines()]==[2,1]
    assert all(v==boxes[0]for v in boxes)
    assert len(t.ledger.memories)==2
    assert t.ledger.memories[1].owner==2 and t.ledger.memories[2].owner==1
    assert t.ledger.memories[1].trusted_frame==12


def test_strong_available_IR_critic_vetoes_but_missing_IR_is_byte_exact_RGB():
    rgb,h,b=run('RGB');critic,c,cb=run('critic');missing,m,mb=run('critic','missing')
    assert critic.statistics['IR_vetoes']>0 and critic.statistics['rebindings']==0
    assert missing.statistics['IR_vetoes']==0 and m==h and mb==b==cb
    assert c[:8]==h[:8]and c[8:]!=h[8:]


def test_sequence_reset_clears_bindings_anchors_and_pending_votes():
    t,history,boxes=run('RGB')
    d=np.asarray([[0.,0.,12.,12.,.9]],np.float32);v=np.asarray([[1.,0.]],np.float32)
    rows=t.update('new_sequence',1,d,v,v)
    assert [r[2]for r in rows]==[1]
    assert t.ledger.bindings=={1:1}and not t.pending and not t.ledger.memories[1].ready
    assert t.statistics['rebindings']==0 and t.statistics['frames']==1
