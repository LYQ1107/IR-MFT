import json
import runpy
from pathlib import Path

import torch

gradient_norms = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/cfp_train_dinov2_fish_identity.py'))['gradient_norms']


def test_amp_overflow_is_explicit_json_null_and_large_finite_norm_is_stable():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([float('inf'), 1.])
    assert gradient_norms({'p': parameter}) == {'p': None}
    json.dumps(gradient_norms({'p': parameter}), allow_nan=False)
    parameter.grad = torch.tensor([3.e20, 4.e20])
    assert abs(gradient_norms({'p': parameter})['p']/5.e20-1) < 1.e-6
