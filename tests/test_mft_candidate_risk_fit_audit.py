import hashlib
import numpy as np
import pytest
from tools.mft_candidate_risk_fit_audit import AuditAccumulator
from tools.mft_candidate_risk_training_stream import FrameEdgeBatch
from tools.mft_candidate_risk_model import RiskLogistic


def fixture():
    models={k:RiskLogistic(np.zeros(26),np.ones(26),k)for k in ['RGB','paired']}
    for m in models.values():m.coefficients[0]=.4
    a=AuditAccumulator('artificial',3,dict(count=4,positive=2,negative=2),models,1)
    x=np.zeros((2,26),np.float32);x[:,0]=[-1,1]
    return a,models,FrameEdgeBatch('artificial',2,x,np.asarray([0,1],np.float32))


def test_auditor_separates_boundary_calibration_without_updating_coefficients():
    a,models,b=fixture();before={k:m.coefficients.copy()for k,m in models.items()}
    a.observe(b);a.observe(FrameEdgeBatch(b.sequence,3,b.x,b.y))
    sums=a.sums.copy();derivative=a.objectives['paired'].gradient.copy()
    a.observe(FrameEdgeBatch(b.sequence,4,b.x+99,b.y))
    np.testing.assert_array_equal(a.sums,sums)
    np.testing.assert_array_equal(a.objectives['paired'].gradient,derivative)
    assert a.count==4 and a.calibration_counts==dict(known=2,positive=1,negative=1)
    for k,m in models.items():np.testing.assert_array_equal(m.coefficients,before[k])


def test_auditor_refuses_duplicate_batch_and_mismatched_early_supervision():
    a,models,b=fixture();a.observe(b)
    with pytest.raises(AssertionError):a.observe(b)
    reference=dict(time_cut_frame=3,count=2,positive=1,negative=1,
        mean=(a.sums/2).tolist(),second_moment=(a.squared/2).tolist(),
        feature_target_packet_sha256=a.signature.hexdigest())
    a.assert_early_matches(reference)
    reference['feature_target_packet_sha256']='0'*64
    with pytest.raises(AssertionError):a.assert_early_matches(reference)
