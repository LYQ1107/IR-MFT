#!/usr/bin/env python3
"""Frozen V3 inference on a declared partition, with cache provenance and timing."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from cfp.provenance import config_hash, sha256, write_json
from yolox.exp import get_exp
from yolox.data import RGBTValTransform
from yolox.utils import postprocess


def keep_batch(batch):
    return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--partition', choices=['train', 'dev', 'historical_test'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--probe-frames', type=int, default=0, help='First N frames only; diagnostic, not an evaluation cache')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.output.exists() and list(args.output.glob('*.npz')):
        raise FileExistsError('Refuse to overwrite existing detection cache')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    source_split = 'test' if args.partition == 'historical_test' else 'train'
    exp = get_exp('exps/yolox_mft26ir_rgbt_v3.py', None)
    exp.data_dir = str(args.data_root)
    # Validation-size inference, including when reading original training JSON.
    exp.input_size = exp.test_size
    transform = RGBTValTransform(rgb_means=exp.rgb_means, std=exp.rgb_std,
                                 ir_means=exp.ir_means, ir_std=exp.ir_std)
    dataset = exp._dataset(source_split, source_split + '.json', transform)
    selected = [i for i, ann in enumerate(dataset.annotations) if ann[2].split('/')[0] in protocol[args.partition]]
    if args.probe_frames:
        selected = selected[:args.probe_frames]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = exp.get_model().cuda()
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state['model'], strict=True)
    del state
    if args.fp16:
        model.half()
    model.eval()
    config = dict(exp='exps/yolox_mft26ir_rgbt_v3.py', input_size=list(exp.test_size), fp16=args.fp16,
                  batch_size=args.batch_size, workers=args.workers, seed=args.seed, conf=.1, nms=.7,
                  partition=args.partition, source_split=source_split, probe_frames=args.probe_frames,
                  rgb_means=exp.rgb_means, rgb_std=exp.rgb_std, ir_means=exp.ir_means, ir_std=exp.ir_std)
    provenance = dict(schema_version=1, config=config, config_sha256=config_hash(config),
                      checkpoint_sha256=sha256(args.checkpoint), checkpoint_path=str(args.checkpoint),
                      protocol_sha256=sha256(args.protocol), annotation_sha256=sha256(args.data_root / 'annotations' / (source_split + '.json')),
                      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      inference_source_sha256=sha256(__file__), torch=torch.__version__,
                      cudnn=torch.backends.cudnn.version(), cuda=torch.version.cuda,
                      device=torch.cuda.get_device_name(), parameters=sum(x.numel() for x in model.parameters()),
                      probe=bool(args.probe_frames), status='RUNNING')
    write_json(args.output / 'manifest.json', provenance)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, selected),
                                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                        pin_memory=True, collate_fn=keep_batch, persistent_workers=args.workers > 0)
    entries = defaultdict(list)
    batch_latencies, input_hashes = [], []
    started_all = time.perf_counter()
    with torch.inference_mode():
        for batch_index, items in enumerate(loader):
            stacked = np.stack([item[0] for item in items])
            if args.probe_frames:
                import hashlib
                input_hashes.append(hashlib.sha256(stacked.tobytes()).hexdigest())
            tensor = torch.from_numpy(stacked).cuda(non_blocking=True)
            if args.fp16:
                tensor = tensor.half()
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = postprocess(model(tensor), 1, .1, .7)
            torch.cuda.synchronize()
            batch_latencies.append(time.perf_counter() - started)
            for item, output in zip(items, outputs):
                info = item[2]
                raw = np.empty((0, 7), dtype=np.float32) if output is None else output.detach().float().cpu().numpy()
                scale = min(exp.test_size[0] / info[0], exp.test_size[1] / info[1])
                entries[info[4].split('/')[0]].append((int(info[2]), raw[:, :4] / scale, raw[:, 4] * raw[:, 5], raw))
            if batch_index % 100 == 0:
                print(f'batch {batch_index}/{len(loader)} frames {min((batch_index + 1) * args.batch_size, len(selected))}', flush=True)
    reports = {}
    for seq, rows in entries.items():
        offsets = np.r_[0, np.cumsum([len(row[2]) for row in rows])].astype(np.int64)
        path = args.output / (seq + '.npz')
        np.savez_compressed(path, frame_ids=np.array([r[0] for r in rows], dtype=np.int32), offsets=offsets,
                            boxes=np.concatenate([r[1] for r in rows]).astype(np.float32),
                            scores=np.concatenate([r[2] for r in rows]).astype(np.float32),
                            raw_post_nms=np.concatenate([r[3] for r in rows]).astype(np.float32))
        reports[seq] = dict(frames=len(rows), detections=int(offsets[-1]), sha256=sha256(path))
        write_json(path.with_suffix('.provenance.json'), dict(provenance, cache=reports[seq], status='COMPLETE'))
    provenance.update(status='COMPLETE', caches=reports, total_frames=len(selected),
                      total_seconds=time.perf_counter() - started_all,
                      batch_detector_nms_latency_p50_p95_ms=(np.quantile(batch_latencies, [.5, .95]) * 1000).tolist(),
                      latency_warning='Batch timing is not strict-online single-frame latency; excludes decode and trackers',
                      peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(), input_batch_sha256=input_hashes)
    write_json(args.output / 'manifest.json', provenance)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
