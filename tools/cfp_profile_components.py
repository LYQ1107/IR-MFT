#!/usr/bin/env python3
"""Profile supported model operator FLOPs without saving tensor traces/weights."""
import argparse
import json
from pathlib import Path

import torch

from cfp.predictive import LatentIdentityPredictor
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.reid import FishReID
from cfp.utility import UtilityMLP, EVIDENCE_FIELDS
from yolox.exp import get_exp


def measure(model, inputs):
    model.eval().requires_grad_(False)
    with torch.inference_mode(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True, record_shapes=True) as profile:
        model(*inputs)
    operators = [{"operator": e.key, "flops": int(e.flops)} for e in profile.key_averages() if e.flops]
    return dict(parameters=sum(v.numel() for v in model.parameters()),
                supported_operator_flops=sum(v['flops'] for v in operators), operators=operators,
                coverage='PyTorch profiler supported conv/matmul FLOPs only; excludes unsupported nonlinear/normalization, generator, decode, crop and Hungarian operations. Not an exhaustive pipeline FLOP count.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Cost records are immutable')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    result = dict(status='RUNNING', torch=torch.__version__, device='CPU',
                  source_snapshots=snapshot_sources(args.output.with_suffix(''), [__file__, 'cfp/reid.py', 'cfp/predictive.py', 'cfp/utility.py']),
                  scope='Architectures/counts at declared shapes; random initialization does not change operator count. Never treat profiling CPU time as online GPU latency.', components={})
    write_json(args.output, result)
    model = get_exp('exps/yolox_mft26ir_rgbt_v3.py', None).get_model().eval()
    result['components']['detector_batch1_6x800x1440'] = measure(model, (torch.zeros(1, 6, 800, 1440),))
    del model
    write_json(args.output, result)
    result['components']['fish_encoder_one_crop_3x128x256'] = measure(FishReID(98, 512), (torch.zeros(1, 3, 128, 256),))
    features = torch.randn(1, 8, 512)
    frames = torch.arange(1, 9)[None]
    valid = torch.ones(1, 8, dtype=torch.bool)
    query = torch.tensor([9])
    for architecture in ('gru', 'transformer'):
        result['components'][architecture + '_one_track_history8'] = measure(LatentIdentityPredictor(512, architecture), (features, frames, valid, query))
    result['components']['utility_one_track'] = measure(UtilityMLP(64), (torch.zeros(1, len(EVIDENCE_FIELDS)),))
    result['status'] = 'COMPLETE'
    result['padding_cost'] = 'Detector batch8 repeats the current frame8 times. Online encoder pads each current crop chunk to128; encode calls and padding, not actual fish count alone, determine work.'
    write_json(args.output, result)
    print({name: {k: v[k] for k in ('parameters', 'supported_operator_flops')} for name, v in result['components'].items()})


if __name__ == '__main__':
    main()
