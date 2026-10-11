#!/usr/bin/env python3
"""Train the declared RGB/causal-IR fish baseline without dev gradient updates."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from cfp.provenance import config_hash, sha256, write_json, snapshot_sources
from cfp.reid import FishReID, temporal_batch_hard


class IdentitySampler:
    def __init__(self, records, seed, identities_per_batch=8, observations_per_identity=8):
        self.rng = np.random.default_rng(seed)
        self.p, self.k = identities_per_batch, observations_per_identity
        self.by_identity = defaultdict(list)
        for i, row in enumerate(records):
            if row['partition'] == 'train':
                if row['train_label'] < 0:
                    raise ValueError('Missing training identity label')
                self.by_identity[row['train_label']].append(i)
        self.labels = sorted(self.by_identity)
        if len(self.labels) < self.p:
            raise ValueError('Insufficient train identities')
        for label in self.labels:
            self.by_identity[label].sort(key=lambda i: records[i]['frame_id'])

    def sample(self):
        positions = []
        for identity in self.rng.choice(self.labels, self.p, replace=False):
            observations = np.array(self.by_identity[int(identity)])
            if len(observations) < self.k:
                chosen = self.rng.choice(observations, self.k, replace=True)
            else:
                chosen = [self.rng.choice(part) for part in np.array_split(observations, self.k)]
            positions.extend(chosen)
        return np.array(positions, np.int64)


def augment(images, rng, config):
    flipped = torch.as_tensor(rng.random(len(images)) < config['horizontal_flip_probability'], device=images.device)
    images[flipped] = images[flipped].flip(-1)
    h, w = images.shape[-2:]
    for i in range(len(images)):
        if rng.random() < config['erase_probability']:
            area = rng.uniform(*config['erase_area_fraction']) * h * w
            aspect = rng.uniform(.5, 2.)
            eh, ew = min(h, max(1, round(math.sqrt(area / aspect)))), min(w, max(1, round(math.sqrt(area * aspect))))
            y, x = int(rng.integers(h - eh + 1)), int(rng.integers(w - ew + 1))
            images[i, :, y:y + eh, x:x + ew] = 0
    return images


def oracle_retrieval(records, embedding, minimum_gap=25):
    reports = {}
    for sequence in sorted({row['sequence'] for row in records if row['partition'] == 'dev'}):
        positions = [i for i, row in enumerate(records) if row['sequence'] == sequence]
        features = embedding[positions]
        times = np.array([records[i]['frame_id'] for i in positions])
        ids = np.array([records[i]['track_id'] for i in positions])
        identity_hits = defaultdict(list)
        all_hits = []
        for start in range(0, len(positions), 128):
            scores = features[start:start + 128] @ features.T
            for offset, row in enumerate(range(start, min(start + 128, len(positions)))):
                gallery = np.flatnonzero(times <= times[row] - minimum_gap)
                if len(gallery) == 0 or ids[row] not in ids[gallery]:
                    continue
                winner = gallery[np.argmax(scores[offset, gallery])]
                hit = bool(ids[row] == ids[winner])
                all_hits.append(hit)
                identity_hits[int(ids[row])].append(hit)
        reports[sequence] = dict(eligible_queries=len(all_hits), samples=len(positions),
                                recall1=float(np.mean(all_hits)),
                                identity_macro_recall1=float(np.mean([np.mean(h) for h in identity_hits.values()])))
    return reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--crop-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--pretrained', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='Resume this incomplete run exactly from its last completed epoch')
    p.add_argument('--keep-intermediate-checkpoints', action='store_true', help='Also retain epoch 1/4/8; default stores fixed selected epoch and resumable last.pt')
    args = p.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError('A training run never overwrites an earlier experiment')
    if args.resume and not (args.output / 'last.pt').is_file():
        raise ValueError('Resume requires this run last.pt')
    if args.resume and json.loads((args.output / 'manifest.json').read_text())['status'] == 'COMPLETE':
        raise ValueError('Completed runs are immutable; do not resume them')
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN' or args.seed not in config['seed_plan']:
        raise ValueError('Undeclared configuration/seed')
    crop_manifest = json.loads((args.crop_root / 'manifest.json').read_text())
    if crop_manifest['status'] != 'COMPLETE' or crop_manifest['protocol_sha256'] != sha256(args.protocol):
        raise ValueError('Incomplete crops or mismatched split protocol')
    if sha256(args.crop_root / 'crops.npy') != crop_manifest['crops_sha256']:
        raise ValueError('Crop cache has changed')
    if sha256(args.crop_root / 'index.json') != crop_manifest['index_sha256']:
        raise ValueError('Crop identity index has changed')
    index = json.loads((args.crop_root / 'index.json').read_text())
    records = index['records']
    protocol = json.loads(args.protocol.read_text())
    for row in records:
        if row['sequence'] not in protocol[row['partition']]:
            raise ValueError('Record partition disagrees with frozen video split')
        if row['partition'] == 'dev' and row['train_label'] != -1:
            raise ValueError('Development identity leaked into classifier')
    observed_labels = sorted({row['train_label'] for row in records if row['partition'] == 'train'})
    if observed_labels != list(range(index['train_identity_count'])):
        raise ValueError('Training classifier labels must cover exactly the declared local identities')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)
    sampler = IdentitySampler(records, args.seed, config['identities_per_batch'], config['observations_per_identity'])
    labels = np.array([row['train_label'] for row in records])
    frames = np.array([row['frame_id'] for row in records])
    sequences = {name: i for i, name in enumerate(sorted({row['sequence'] for row in records}))}
    sequence_labels = np.array([sequences[row['sequence']] for row in records])
    mapped = np.load(args.crop_root / 'crops.npy', mmap_mode='r')
    if mapped.shape[0] != len(records):
        raise ValueError('Crop-to-label alignment mismatch')
    model = FishReID(index['train_identity_count'], config['dimension'])
    model.initialize_imagenet(args.pretrained)
    model.cuda().train()
    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids and parameter.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=backbone_parameters, lr=config['backbone_lr']),
                                   dict(params=head_parameters, lr=config['head_lr'])], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    args.output.mkdir(parents=True, exist_ok=args.resume)
    sources = snapshot_sources(args.output, [__file__, 'cfp/reid.py', 'cfp/provenance.py', args.protocol, args.config])
    manifest = dict(status='RUNNING', config=config, config_sha256=config_hash(config), seed=args.seed,
                    modality=crop_manifest['modality'], crop_manifest_sha256=sha256(args.crop_root / 'manifest.json'),
                    pretrained_sha256=sha256(args.pretrained), index_sha256=crop_manifest['index_sha256'],
                    protocol_sha256=sha256(args.protocol), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256=sha256(__file__), model_source_sha256=sha256('cfp/reid.py'),
                    source_snapshots=sources,
                    torch=torch.__version__, cudnn=torch.backends.cudnn.version(), physical_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    parameters=sum(parameter.numel() for parameter in model.parameters()),
                    train_identity_count=index['train_identity_count'], training_videos=protocol['train'],
                    heldout_videos=protocol['dev'], protocol_role=protocol.get('role', 'canonical appearance baseline'),
                    parent_protocol_sha256=protocol.get('parent_protocol_sha256'),
                    checkpoint_retention='fixed selected epoch plus resumable last.pt' + (' and milestones 1/4/8' if args.keep_intermediate_checkpoints else ''),
                    selection=config['checkpoint_selection'], scope='GT crop fish ReID training, dev only for reporting; no tracking claim',
                    input_regime=crop_manifest['input_regime'], epochs=[])
    manifest['crop_variant'] = crop_manifest.get('crop_variant', 'original')
    manifest['crop_view_source_sha256'] = crop_manifest.get('crop_view_source_sha256')
    start_epoch, prior_seconds = 0, 0.
    if args.resume:
        previous = json.loads((args.output / 'manifest.json').read_text())
        if previous['status'] == 'COMPLETE':
            raise ValueError('Completed runs are immutable; do not resume them')
        checkpoint = torch.load(args.output / 'last.pt', map_location='cpu')
        if 'rng_state' not in checkpoint:
            raise ValueError('Legacy checkpoint lacks exact sampler/augmentation RNG state; cannot claim deterministic resume')
        for key in ['config_sha256', 'seed', 'crop_manifest_sha256', 'pretrained_sha256', 'index_sha256',
                    'protocol_sha256', 'source_sha256', 'model_source_sha256', 'torch', 'cudnn', 'physical_visible_devices']:
            if checkpoint['metadata'][key] != manifest[key]:
                raise ValueError(f'Resume provenance changed: {key}')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        rng.bit_generator.state = checkpoint['rng_state']['augmentation']
        sampler.rng.bit_generator.state = checkpoint['rng_state']['sampler']
        torch.set_rng_state(checkpoint['rng_state']['torch_cpu'])
        torch.cuda.set_rng_state_all(checkpoint['rng_state']['torch_cuda'])
        start_epoch = checkpoint['epoch']
        prior_seconds = checkpoint['elapsed_seconds_before_checkpoint']
        manifest = checkpoint['metadata']
        manifest['resumes'] = manifest.get('resumes', []) + [dict(checkpoint_sha256=sha256(args.output / 'last.pt'), start_epoch=start_epoch,
                                                               git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())]
    write_json(args.output / 'manifest.json', manifest)
    steps_per_epoch = math.ceil(sum(row['partition'] == 'train' for row in records) / (sampler.p * sampler.k))
    started_all = time.perf_counter()
    for epoch in range(start_epoch, config['epochs']):
        model.train()
        sums = np.zeros(4, np.float64)
        epoch_start = time.perf_counter()
        for step in range(steps_per_epoch):
            progress = epoch + step / steps_per_epoch
            if progress < config['warmup_epochs']:
                fraction = .1 + .9 * progress / config['warmup_epochs']
            else:
                phase = (progress - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])
                fraction = config['cosine_min_lr_fraction'] + (1 - config['cosine_min_lr_fraction']) * .5 * (1 + math.cos(math.pi * phase))
            for group, base_lr in zip(optimizer.param_groups, [config['backbone_lr'], config['head_lr']]):
                group['lr'] = base_lr * fraction
            positions = sampler.sample()
            images = torch.from_numpy(np.array(mapped[positions])).cuda().float()
            images = augment(images, rng, config['augmentation'])
            target = torch.as_tensor(labels[positions], device='cuda')
            timestamps = torch.as_tensor(frames[positions], device='cuda')
            video_labels = torch.as_tensor(sequence_labels[positions], device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config['amp']):
                embedding, logits = model(images)
                ce = F.cross_entropy(logits.float(), target, label_smoothing=config['ce_label_smoothing'])
            # cdist needs float32 and is explicitly outside autocast.
            triplet, coverage = temporal_batch_hard(embedding, target, timestamps, video_labels,
                                                    config['triplet_margin'], config['triplet_positive_minimum_frame_gap'])
            loss = ce + config['triplet_weight'] * triplet
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite ReID objective')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            scaler.step(optimizer)
            scaler.update()
            accuracy = (logits.argmax(dim=-1) == target).float().mean()
            sums += [float(ce.detach()), float(triplet.detach()), coverage, float(accuracy.detach())]
        torch.cuda.synchronize()
        report = dict(epoch=epoch + 1, ce=sums[0] / steps_per_epoch, triplet=sums[1] / steps_per_epoch,
                      valid_triplet_anchor_fraction=sums[2] / steps_per_epoch, train_accuracy=sums[3] / steps_per_epoch,
                      seconds=time.perf_counter() - epoch_start, steps=steps_per_epoch)
        manifest['epochs'].append(report)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     epoch=epoch + 1, config=config, seed=args.seed, metadata=manifest,
                     rng_state=dict(augmentation=rng.bit_generator.state, sampler=sampler.rng.bit_generator.state,
                                    torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all()),
                     elapsed_seconds_before_checkpoint=prior_seconds + time.perf_counter() - started_all)
        temporary = args.output / 'last.partial.pt'
        torch.save(state, temporary)
        temporary.replace(args.output / 'last.pt')
        if epoch + 1 == config['epochs'] or (args.keep_intermediate_checkpoints and epoch + 1 in [1, 4, 8]):
            torch.save(dict(model=model.state_dict(), epoch=epoch + 1, config=config, seed=args.seed, metadata=manifest),
                       args.output / f'epoch{epoch + 1}.pt')
        write_json(args.output / 'manifest.json', manifest)
        print(json.dumps(report), flush=True)
    model.eval()
    feature_rows = []
    latencies = []
    with torch.inference_mode():
        for start in range(0, len(records), 128):
            images = torch.from_numpy(np.array(mapped[start:start + 128])).cuda().float()
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=config['amp']):
                embedding, _ = model(images)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t)
            if not torch.isfinite(embedding).all() or not torch.allclose(embedding.norm(dim=-1), torch.ones(len(embedding), device='cuda'), atol=1e-4):
                raise ValueError('Invalid trained embeddings')
            feature_rows.append(embedding.cpu().numpy())
    all_features = np.concatenate(feature_rows)
    np.savez_compressed(args.output / 'features.npz', embedding=all_features)
    retrieval = oracle_retrieval(records, all_features, config['triplet_positive_minimum_frame_gap'])
    manifest.update(status='COMPLETE', checkpoint_sha256=sha256(args.output / f'epoch{config["epochs"]}.pt'),
                    features_sha256=sha256(args.output / 'features.npz'), dev_oracle_retrieval=retrieval,
                    total_seconds=prior_seconds + time.perf_counter() - started_all, peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                    crop_encoder_batch128_latency_p50_p95_ms=(np.quantile(latencies, [.5, .95]) * 1000).tolist(),
                    latency_scope='Cache-backed crop encoding; excludes original image decode, detector and tracker; not end-to-end FPS')
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['config', 'epochs']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
