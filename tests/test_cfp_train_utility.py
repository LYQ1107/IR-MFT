import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cfp.utility import candidate_ce

spec=importlib.util.spec_from_file_location('cfp_train_utility_cli',Path(__file__).parents[1]/'tools/cfp_train_utility.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_ragged_candidate_ce_retains_true_candidate_and_null_without_padding_mass():
    rows=[dict(rgb=np.array([.1,.3],np.float32),ir=np.array([.2,.0],np.float32),target=0),
          dict(rgb=np.array([.1],np.float32),ir=np.array([.2],np.float32),target=1)]
    rgb,ir,mask,target=module.candidate_batch(rows,dict(null_distance=.35))
    logits=(-rgb/.1).masked_fill(~mask,-torch.inf)
    losses=F.cross_entropy(logits,target,reduction='none')
    expected=[candidate_ce(r['rgb'],r['target']) for r in rows]
    assert np.allclose(losses.numpy(),expected,atol=1e-6)
    assert mask.tolist()==[[True,True,True],[True,True,False]]
    assert rgb[1,1]==ir[1,1] and target.tolist()==[0,1]


def test_utility_metrics_report_uninformative_constant_predictions_honestly():
    result=module.utility_metrics([0,0,0,0],[-1,1,0,2])
    assert result['sign_auc_nonzero']==.5 and result['prediction_std']==0 and result['correlation'] is None
    result=module.utility_metrics([0,0],[0,0])
    assert result['sign_auc_nonzero'] is None and result['mae']==0
