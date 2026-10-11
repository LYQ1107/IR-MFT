import importlib.util
from pathlib import Path

import numpy as np
import torch

from cfp.predictive import LatentIdentityPredictor,latent_identity_loss

spec=importlib.util.spec_from_file_location('cfp_predictor_training_cli',Path(__file__).parents[1]/'tools/cfp_train_predictor.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_stopped_batch_alignment_and_contaminated_vs_clean_inputs():
    features={'A':np.eye(4,dtype=np.float32)}
    row=dict(sequence='A',frame=5,identity=3,gap=1,positive=1,candidates=np.array([2,3]),
             clean_indices=np.array([0,1,-1,-1,-1,-1,-1,-1]),actual_indices=np.array([0,2,-1,-1,-1,-1,-1,-1]),
             clean_frames=np.array([1,4,0,0,0,0,0,0]),actual_frames=np.array([1,3,0,0,0,0,0,0]))
    clean=module.make_batch([row],features,'clean','cpu');actual=module.make_batch([row],features,'actual','cpu')
    assert torch.equal(clean[4],torch.tensor([[0.,0.,0.,1.]]))
    assert torch.equal(clean[4],actual[4]) and torch.equal(clean[5],actual[5])
    assert not torch.equal(clean[0],actual[0]) and torch.equal(clean[2],actual[2])
    assert not any(v.requires_grad for v in clean)
    model=LatentIdentityPredictor(4,'gru',hidden=8)
    prediction,_=model(*clean[:4]);loss,_=latent_identity_loss(prediction,*clean[4:])
    loss.backward()
    assert model.output.weight.grad is not None and torch.isfinite(model.output.weight.grad).all()
    assert not any(v.grad is not None for v in clean if v.is_floating_point())


def test_sampling_balances_videos_and_local_identities_without_cross_video_labels():
    rows=[dict(sequence='A',identity=3) for _ in range(100)]+[dict(sequence='A',identity=7)]+[dict(sequence='B',identity=3)]
    groups=module.sampling_groups(rows,['A','B'])
    first=module.sample_positions(groups,np.random.default_rng(9),10000)
    assert first==module.sample_positions(groups,np.random.default_rng(9),10000)
    selected=[rows[i] for i in first]
    fraction_b=sum(r['sequence']=='B' for r in selected)/len(selected)
    fraction_a7=sum(r['sequence']=='A' and r['identity']==7 for r in selected)/len(selected)
    assert .47<fraction_b<.53 and .22<fraction_a7<.28
