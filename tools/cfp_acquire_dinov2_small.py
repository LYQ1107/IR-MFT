#!/usr/bin/env python3
"""Acquire one licensed official small backbone outside Git, with size guards."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import torch

from cfp.provenance import sha256, write_json


URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    reference = Path('/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2')
    if subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('Exact licensed model source required')
    card = reference / 'MODEL_CARD.md'
    if '- **License:** Apache License 2.0' not in card.read_text() or URL not in (reference / 'README.md').read_text():
        raise ValueError('Official small-weight URL and explicit model license required')
    guard = dict(url=URL, reference_commit=COMMIT, model_card_sha256=sha256(card), license_sha256=sha256(reference / 'LICENSE'))
    if args.output.exists():
        if not args.resume:
            raise FileExistsError('Preserve existing backbone assets; explicit matching resume required')
        old = json.loads((args.output / 'manifest.json').read_text())
        if old['guard'] != guard or old['status'] == 'COMPLETE':
            raise ValueError('Only incomplete identical licensed download may resume')
    else:
        args.output.mkdir(parents=True)
    if shutil.disk_usage(args.output).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    temporary = args.output / 'dinov2_vits14_pretrain.partial.pth'
    final = args.output / 'dinov2_vits14_pretrain.pth'
    report = dict(status='RUNNING', guard=guard, source_sha256=sha256(__file__),
        scope='One official unfitted-MFT ViTS14 pretrained backbone only; no other weights or datasets acquired. LVD142M overlap cannot be excluded. No external project code executed by this acquisition.')
    write_json(args.output / 'manifest.json', report)
    command = ['curl', '--proxy', 'http://127.0.0.1:7890', '--fail', '--show-error', '--location',
        '--retry', '2', '--retry-all-errors', '--retry-delay', '1', '--connect-timeout', '10',
        '--max-time', '60', '--max-filesize', '100000000', '--continue-at', '-', '--output', str(temporary), URL]
    report['command'] = command
    try:
        subprocess.run(command, check=True, timeout=200)
        if not 1_000_000 <= temporary.stat().st_size <= 100_000_000:
            raise ValueError('Unexpected small-backbone asset size')
        state = torch.load(temporary, map_location='cpu', weights_only=True)
        for key, shape in [('cls_token', (1, 1, 384)), ('patch_embed.proj.weight', (384, 3, 14, 14)), ('norm.weight', (384,))]:
            if key not in state or tuple(state[key].shape) != shape or not torch.isfinite(state[key]).all():
                raise ValueError('Official ViTS14 checkpoint schema mismatch')
        if len({int(key.split('.')[1]) for key in state if key.startswith('blocks.')}) != 12:
            raise ValueError('Official small-backbone depth mismatch')
        if not all(torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor)):
            raise ValueError('Nonfinite pretrained backbone')
        if final.exists():
            raise FileExistsError('Never replace existing pretrained model')
        temporary.rename(final)
        report.update(status='COMPLETE', file_sha256=sha256(final), bytes=final.stat().st_size,
                      schema_checked=True, source_checksum_available=False,
                      checksum_scope='Recorded local SHA256 and official HTTPS origin; no publisher whole-file digest was supplied by the model card.')
    except Exception as error:
        report.update(status='ACQUISITION_FAILED', error=repr(error), partial_bytes=temporary.stat().st_size if temporary.exists() else 0)
        write_json(args.output / 'manifest.json', report)
        raise
    write_json(args.output / 'manifest.json', report)
    print('COMPLETE', final, report['bytes'], report['file_sha256'], flush=True)


if __name__ == '__main__':
    main()
