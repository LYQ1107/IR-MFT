#!/usr/bin/env python3
"""Fixed-budget heldout visual/local adapters on stopped current crop features."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch

from cfp.identity_evidence import query_retrieval
from cfp.local_identity_encoder import LocalIdentityAdapter
from cfp.local_identity_training import load_local_features, SameFrameTemporalSampler, local_batch, local_identity_objective
from cfp.provenance import sha256, snapshot_sources, write_json


def atomic_checkpoint(path, value):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def diagnose(model, video, banks, device, count):
    bank = banks[video]
    arrays, records = bank['arrays'], bank['records']
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 256):
            stop = min(len(records), start + 256)
            inputs = [torch.from_numpy(np.array(arrays[name][start:stop])).float().to(device)
                      for name in ('imagenet_global', 'imagenet_local' + str(count), 'visibility' + str(count))]
            outputs.append(model(*inputs)['embedding'].cpu().numpy())
    embeddings = np.concatenate(outputs)
    result = {}
    for view in ('gt', 'detector'):
        indices = np.asarray([i for i, row in enumerate(records) if row['view'] == view])
        times = np.asarray([records[i]['frame'] for i in indices])
        labels = np.asarray([records[i]['identity'] for i in indices])
        valid = arrays['valid'][indices]
        query_indices = []
        for identity in np.unique(labels[labels >= 0]):
            available = np.flatnonzero((labels == identity) & valid)
            selected = np.unique(np.linspace(0, len(available) - 1, min(len(available), 64), dtype=int))
            if len(available):
                query_indices.extend(available[selected].tolist())
        query_indices.sort()
        gallery_features = torch.from_numpy(embeddings[indices]).to(device)
        metrics = {str(gap): [] for gap in (1, 25, 100)}
        for start in range(0, len(query_indices), 64):
            chosen = query_indices[start:start + 64]
            with torch.inference_mode():
                scores = (gallery_features[chosen] @ gallery_features.T).cpu().numpy()
            for j, query in enumerate(chosen):
                risk = np.where(labels < 0, -1, (labels != labels[query]).astype(np.int8)).astype(np.int8)
                for gap in (1, 25, 100):
                    selected = times <= times[query] - gap
                    metrics[str(gap)].append(query_retrieval(scores[j, selected], risk[selected]))
        result[view] = {}
        for gap, rows in metrics.items():
            supported = [r for r in rows if r['target_present']]
            result[view][gap] = dict(queries=len(rows), supported_queries=len(supported), missing_target=len(rows) - len(supported),
                rank1_percent=100 * np.mean([r['rank1'] for r in supported]) if supported else None,
                rank5_percent=100 * np.mean([r['rank5'] for r in supported]) if supported else None,
                mAP_percent=100 * np.mean([r['average_precision'] for r in supported]) if supported else None,
                unknown_selected=sum(r['unknown_selected'] for r in supported))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['global', 'dual_global', 'simple', 'learned'], required=True)
    parser.add_argument('--outer', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (config['state'] != 'FROZEN' or args.outer not in config['videos'] or args.seed not in config['seed_plan']
            or args.mode not in config['modes'] or not os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('Registered heldout adapter and explicit idle GPU required')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Never replace existing learned representation')
    if shutil.disk_usage(args.output.parent).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    inner = config['videos'][(config['videos'].index(args.outer) + 1) % len(config['videos'])]
    fit = [video for video in config['videos'] if video not in (args.outer, inner)]
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.2)
    banks, meta = load_local_features(args.features, config['local_count'])
    sampler = SameFrameTemporalSampler(banks, fit, args.seed, config['identities_per_frame'], config['observations_per_identity'], config['minimum_positive_gap'])
    video_map = {video: i for i, video in enumerate(fit)}
    model = LocalIdentityAdapter(len(sampler.labels), config['dimension'], config['local_count'], args.mode).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    sources = [__file__, 'cfp/local_identity_encoder.py', 'cfp/local_identity_training.py', 'cfp/reid.py', 'cfp/identity_evidence.py']
    guard = dict(config_sha256=sha256(args.config), feature_manifest_sha256=sha256(args.features / 'manifest.json'),
        mode=args.mode, outer=args.outer, inner=inner, fit=fit, seed=args.seed, source_sha256={p: sha256(p) for p in sources},
        physical_devices=os.environ['CUDA_VISIBLE_DEVICES'], torch=torch.__version__, imagenet_checkpoint_sha256=meta['imagenet_checkpoint_sha256'])
    report = dict(status='RUNNING', guard=guard, config=config, epochs=[], evaluations={},
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        classifier_identity_count=len(sampler.labels), classifier_keys=[[video, identity] for video, identity in sampler.labels],
        allocated_parameters=sum(p.numel() for p in model.parameters()),
        scope=config['scope'], training_seconds=0., input_source='Only stopped ImageNet columns; no fish-trained branch or heldout gradient/normalization statistics.')
    start_epoch = 0
    if args.resume:
        previous = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        if previous['guard'] != guard or json.loads((args.output / 'manifest.json').read_text())['status'] == 'COMPLETE':
            raise ValueError('Resume source/config/data/runtime mismatch or already complete')
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        sampler.rng.bit_generator.state = previous['rng_sampler']
        random.setstate(previous['rng_python'])
        np.random.set_state(previous['rng_numpy'])
        torch.set_rng_state(previous['rng_torch'])
        torch.cuda.set_rng_state_all(previous['rng_cuda'])
        start_epoch, report = previous['epoch'], previous['report']
    else:
        args.output.mkdir(parents=True)
        report['source_snapshots'] = snapshot_sources(args.output / 'sources', sources + [args.config])
    write_json(args.output / 'manifest.json', report)
    for epoch in range(start_epoch + 1, config['epochs'] + 1):
        if shutil.disk_usage(args.output).free < 40 * 2**30:
            raise OSError('40GiB reserve')
        model.train()
        parts, same_frame_anchors = [], 0
        start = time.monotonic()
        for step in range(config['steps_per_epoch']):
            batch = local_batch(sampler.sample(), banks, sampler.labels, video_map, 'cuda', config['local_count'])
            raw, local, visibility, labels, frames, videos = batch
            output = model(raw, local, visibility)
            loss, detail = local_identity_objective(output, labels, frames, videos, config['minimum_positive_gap'],
                config['triplet_weight'], config['same_frame_weight'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite identity training')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite identity gradient')
            optimizer.step()
            parts.append([float(loss.detach()), detail['ce'], detail['temporal'], detail['same_frame']])
            same_frame_anchors += detail['same_frame_anchors']
        scheduler.step()
        seconds = time.monotonic() - start
        report['training_seconds'] += seconds
        entry = dict(epoch=epoch, loss_ce_temporal_frame=np.mean(parts, axis=0).tolist(),
                     same_frame_supervised_anchors=same_frame_anchors, seconds=seconds)
        report['epochs'].append(entry)
        atomic_checkpoint(args.output / 'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            epoch=epoch, guard=guard, report=report, rng_sampler=sampler.rng.bit_generator.state,
            rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all()))
        write_json(args.output / 'manifest.json', report)
        print(json.dumps(entry), flush=True)
    report['optimizer_gradient_participating_parameters'] = sum(p.numel() for p in model.parameters() if p in optimizer.state)
    for video in (inner, args.outer):
        report['evaluations'][video] = diagnose(model, video, banks, 'cuda', config['local_count'])
        write_json(args.output / 'manifest.json', report)
        print(video, report['evaluations'][video], flush=True)
    atomic_checkpoint(args.output / 'epoch8.pt', dict(model=model.state_dict(), guard=guard, classifier_keys=report['classifier_keys'], epoch=config['epochs']))
    report.update(status='COMPLETE', checkpoint_sha256=sha256(args.output / 'epoch8.pt'),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
