from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cfp.native_local_identity import RecentIdentityControl, actual_model_inputs


def bank():
    return dict(local4=np.random.default_rng(4).normal(size=(6,4,12)).astype(np.float32),
        visibility4=np.ones((6,4),np.float32),raw_cls=np.ones((6,12),np.float32),
        valid=np.ones(6,bool),frame_of_proposal=np.array([1,2,3,4,5,6]))


def test_only_actual_assigned_past_proposals_enter_native_context():
    b=bank()
    tracks=[SimpleNamespace(identity_proposals=[(1,0),(3,2),(4,3)])]
    detections=[SimpleNamespace(current_proposal_index=5)]
    x=actual_model_inputs(tracks,detections,6,b,length=2)
    assert x['times'].tolist()==[[3,4]] and x['history_valid'].all()
    torch.testing.assert_close(x['history'][0,0],torch.from_numpy(b['local4'][2]))
    assert not any('target' in key or 'physical' in key or 'track_id' in key for key in x)
    tracks[0].identity_proposals.append((6,5))
    with pytest.raises(ValueError,match='Current/future'):
        actual_model_inputs(tracks,detections,6,b)


def test_timestamp_source_mismatch_and_future_current_reference_rejected():
    b=bank()
    with pytest.raises(ValueError,match='timestamp/source'):
        actual_model_inputs([SimpleNamespace(identity_proposals=[(2,0)])],[],5,b)
    with pytest.raises(ValueError,match='current frame'):
        actual_model_inputs([], [SimpleNamespace(current_proposal_index=5)],5,b)


def test_recent_control_reads_same_last_four_and_cannot_hallucinate_empty_history():
    x=torch.randn(2,8,12)
    times=torch.tensor([[1,2,3,4,5,6,7,8],[0,0,0,0,0,0,0,0]])
    valid=times>0;frame=torch.tensor([9,9])
    model=RecentIdentityControl('fifo',4)
    first,available=model(x,times,valid,frame)
    changed=x.clone();changed[:, :4]=1.e5
    torch.testing.assert_close(model(changed,times,valid,frame)[0],first,rtol=0,atol=0)
    assert available.tolist()==[True,False] and torch.count_nonzero(first[1])==0
