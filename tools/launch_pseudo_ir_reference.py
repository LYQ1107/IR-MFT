#!/usr/bin/env python3
"""Replay the preserved V3 model after the main new-input GPU3 queue exits.

This is the secondary reference in the frozen plan, not the matched old/new
training control. It uses the historical model's legacy normalization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from launch_pseudo_ir_campaign import run_job, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--after-pid', type=int)
    parser.add_argument('--result-root', type=Path,
        default=Path('/data3/liuyeqiang/IR-MFT_assets/pseudo_ir_comparison_20261120'))
    args = parser.parse_args()
    args.gpu = 3
    checkpoint = Path('/data3/liuyeqiang/IR-MFT_assets/checkpoints/mft26ir_rgbt_v3_best_single_gpu.pth.tar')
    h = hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    manifest = dict(checkpoint=str(checkpoint), sha256=h.hexdigest(), bytes=checkpoint.stat().st_size,
        role='preserved historical V3 checkpoint; different training history/budget from matched arms',
        auxiliary_input='archived legacy infrared JPEG', normalization='historical V2/V3 defaults',
        ir_means=[0.0651,0.0262,0.0581], ir_std=[0.1916,0.1015,0.1084],
        selection='preserved checkpoint named in the frozen secondary-reference plan, not Test-selected in this campaign')
    manifest_path = args.result_root / 'historical_v3_checkpoint_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('Preserved reference checkpoint changed')
    write(manifest_path, manifest)
    print('REFERENCE_CHECKPOINT_FROZEN', json.dumps(manifest), flush=True)
    if args.after_pid:
        process = Path('/proc') / str(args.after_pid)
        print('Waiting for the existing GPU3 main queue to exit.', flush=True)
        while process.exists() and process.joinpath('stat').read_text().split()[2] != 'Z':
            time.sleep(10)
    main_test = args.result_root / 'eval_new_diffv2ir_test/metrics.json'
    if not main_test.exists() or json.loads(main_test.read_text())['protocol']['frames'] != 15985:
        raise RuntimeError('Main new-input Test must complete before this secondary GPU3 replay')
    for split in ['dev','test']:
        run_job(args, 'reference_frozen_v3_' + split,
            ['infer','--variant','legacy','--ckpt',str(checkpoint),'--split',split,
             '--batch-size','4','--workers','4'])
    write(args.result_root / 'queue_gpu3.state.json',
          dict(status='COMPLETE', physical_gpu=3, includes_historical_reference=True, updated_unix=time.time()))


if __name__ == '__main__':
    main()
