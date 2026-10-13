#!/usr/bin/env python3
"""Exact shape-based dense affine MACs, without images, weights or GPU execution."""
import argparse
import inspect
import json
from pathlib import Path

import torch
from torch import nn

from cfp.provenance import sha256, write_json
from cfp.reid import FishReID
from cfp.online import CurrentFrameFrontend
from yolox.exp import get_exp


def count(model, shape, **forward_args):
    total = dict(conv_macs=0, linear_macs=0, bias_adds=0)
    hooks = []

    def observe(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            ops = output.numel() * module.in_channels // module.groups
            ops *= module.kernel_size[0] * module.kernel_size[1]
            total['conv_macs'] += ops
        else:
            total['linear_macs'] += output.numel() * module.in_features
        if module.bias is not None:
            total['bias_adds'] += output.numel()

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(observe))
    try:
        with torch.inference_mode():
            model(torch.empty(shape, device='meta'), **forward_args)
    finally:
        for hook in hooks:
            hook.remove()
    total['affine_flops_without_bias'] = 2 * (total['conv_macs'] + total['linear_macs'])
    total['parameters'] = sum(p.numel() for p in model.parameters())
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Immutable operation accounting required')
    torch.set_num_threads(2)
    detector = get_exp('exps/yolox_mft26ir_rgbt_v3.py', None).get_model().eval().to('meta')
    detector.head.decode_in_inference = False
    rgb = FishReID().eval().to('meta')
    d = count(detector, (1, 6, 800, 1440))
    r = count(rgb, (1, 3, 128, 256), embedding_stage='post_neck')
    if d['parameters'] != 186199381 or r['parameters'] != 24607808:
        raise ValueError('Architecture parameter count differs from canonical deployment')
    encoder_batch = inspect.signature(CurrentFrameFrontend.__init__).parameters['encoder_batch'].default
    report = dict(status='COMPLETE', detector_one_item=d, rgb_one_crop=r,
        current_padding=dict(detector_items_per_frame=8, rgb_items_per_encoding_chunk=encoder_batch,
            detector_affine_flops_per_frame=8*d['affine_flops_without_bias'],
            rgb_affine_flops_per_padded_chunk=encoder_batch*r['affine_flops_without_bias']),
        scope='Exact dense Conv2d/Linear multiply-add shape accounting, two FLOPs/MAC. '
              'Architecture instantiated without checkpoint or image access; meta tensor execution uses no GPU. '
              'Includes RGB classification affine computation performed by current encoder forward. '
              'Excludes BN, activation, pooling, fusion elementwise operations, detector decoding/NMS, '
              'IR/codec/resize/crop/transfer, ordinary GRU/risk feature extraction and lifecycle/Hungarian. '
              'Not a complete pipeline FLOPs value or latency measurement. RGB chunk count varies with proposals.',
        source_hashes={path: sha256(path) for path in [__file__, 'cfp/reid.py', 'cfp/online.py', 'exps/yolox_mft26ir_rgbt_v3.py',
            'exps/yolox_mft26ir_rgbt_v2.py', 'yolox/models/yolo_pafpn_rgbt.py', 'yolox/models/yolo_head.py']})
    write_json(args.output, report)
    print(json.dumps(dict(detector=d['affine_flops_without_bias'], rgb=r['affine_flops_without_bias'], padding=report['current_padding'])))


if __name__ == '__main__':
    main()
