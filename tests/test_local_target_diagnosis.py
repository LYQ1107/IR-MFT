import torch
from cfp.local_dense_context_identity import ContextIdentityStudent
from cfp.local_dense_jepa import DenseLocalTarget
from cfp.local_target_diagnosis import TeacherLocalFIFO,FitMeanPrediction


def inputs():
    torch.manual_seed(23);n,l,k,d,m=2,3,4,12,3
    return dict(history=torch.randn(n,l,k,d),history_fraction=torch.ones(n,l,k),history_global=torch.randn(n,l,d),times=torch.tensor([[1,2,3],[1,2,3]]),history_valid=torch.ones(n,l,dtype=torch.bool),frame=4,candidates=torch.randn(m,k,d),candidate_fraction=torch.ones(m,k),candidate_global=torch.randn(m,d))


def test_teacher_diagnostic_uses_stopped_encoder_and_preserves_availability():
    student=ContextIdentityStudent(3,'full',4,12,3).eval();teacher=DenseLocalTarget(student);x=inputs();out=TeacherLocalFIFO(teacher)(**x)
    assert out['scores'].shape==(2,3) and out['history_available'].all()
    assert int(teacher.updates)==0 and all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    x['times'][0,-1]=4
    try:TeacherLocalFIFO(teacher)(**x)
    except ValueError:pass
    else:raise AssertionError('Current history must reject')


def test_fit_mean_control_has_no_query_identity_information_or_current_target_copy():
    student=ContextIdentityStudent(3,'full',4,12,3).eval();mean=torch.randn(4,12);control=FitMeanPrediction(student,mean);x=inputs();a=control(**x);x['candidates']=x['candidates']+40;b=control(**x)
    assert torch.equal(a['predicted'],b['predicted']) and torch.equal(a['predicted'][0],a['predicted'][1])
    x['history_valid'][1]=False;x['times'][1]=0;x['history_fraction'][1]=0;c=control(**x)
    assert not c['history_available'][1] and torch.equal(c['predicted'][1],torch.zeros_like(c['predicted'][1]))
