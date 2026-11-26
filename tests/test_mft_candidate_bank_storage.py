from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
import numpy as np
import pytest
from tools.mft_candidate_bank_storage import PayloadBudget,payload_bytes,quantize_frame


def test_descriptor_quantization_does_not_change_detection_packet():
    d=np.asarray([[0,0,20,20,.9],[30,0,55,20,.8]],np.float32)
    v=np.arange(1,1025,dtype=np.float32).reshape(2,512);v/=np.linalg.norm(v,axis=1,keepdims=True)
    packed,errors=quantize_frame(d,dict(rgb=v,ir=v[:,::-1].copy()))
    np.testing.assert_array_equal(packed['detections'],d)
    assert packed['rgb'].dtype==packed['ir'].dtype==np.float16
    assert all(x['maximum_unit_norm_error']<=1e-3 for x in errors.values())
    assert payload_bytes(2,1)==d.nbytes+packed['rgb'].nbytes+packed['ir'].nbytes+3*8


def test_hard_limit_and_monotonicity_reject_without_modifying_ledger(tmp_path):
    b=PayloadBudget(tmp_path,10);b.reserve('one',6);original=b.path.read_bytes()
    with pytest.raises(RuntimeError):b.reserve('two',5)
    assert b.path.read_bytes()==original
    with pytest.raises(AssertionError):b.reserve('one',5)
    assert b.path.read_bytes()==original
    assert b.reserve('two',4)['total_reserved_bytes']==10


def test_two_competing_writers_cannot_overreserve_global_budget(tmp_path):
    barrier=Barrier(2)
    def run(name):
        b=PayloadBudget(tmp_path,10);barrier.wait()
        try:b.reserve(name,7);return True
        except RuntimeError:return False
    with ThreadPoolExecutor(max_workers=2)as pool:
        results=list(pool.map(run,['one','two']))
    assert sum(results)==1
    state=json.loads((tmp_path/'payload_reservations.json').read_text())
    assert state['total_reserved_bytes']==7 and sum(state['sequences'].values())==7


def test_malformed_descriptor_or_proposal_is_refused():
    d=np.asarray([[0,0,20,20,.9]],np.float32)
    bad=np.zeros((1,512),np.float32)
    with pytest.raises(AssertionError):quantize_frame(d,dict(rgb=bad,ir=bad))
    with pytest.raises(AssertionError):quantize_frame(d.astype(np.float16),dict(rgb=bad,ir=bad))
