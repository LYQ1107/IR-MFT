#!/usr/bin/env python3
"""Actual backbone-level fish fitting excludes both outer and inner videos."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch

from cfp.crop_retrieval import embedding_crop_retrieval
from cfp.local_identity_training import load_local_features, SameFrameTemporalSampler, local_identity_objective
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.trainable_local_identity import TrainableLocalFishReID


def atomic_checkpoint(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--crops', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['global', 'learned'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int,
        help='Checkpoint and exit early for controlled restart verification; never report COMPLETE')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch < config['epochs']:
        raise ValueError('Early stop must be before the registered final epoch')
    if (config['state'] != 'FROZEN' or args.mode not in config['modes'] or args.seed not in config['seed_plan']
            or not os.environ.get('CUDA_VISIBLE_DEVICES') or args.output.exists() and not args.resume):
        raise ValueError('Registered visual fitting, explicit idle GPU and new or matching incomplete output required')
    if (set(config['fit']) & {config['outer'], config['inner']} or config['outer'] == config['inner']):
        raise ValueError('Backbone fitting cannot include outer or inner calibration video')
    if shutil.disk_usage(args.output.parent).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.4)
    banks, fm = load_local_features(args.features, config['local_count'])
    crop_meta = json.loads((args.crops / 'manifest.json').read_text())
    if crop_meta['status'] != 'COMPLETE_ALIGNED_PILOT_CROPS' or crop_meta['feature_manifest_sha256'] != sha256(args.features / 'manifest.json'):
        raise ValueError('Exact same-pilot crop/feature pairing required')
    crops = {}
    for video, entry in crop_meta['videos'].items():
        p = args.crops / (video + '.npy')
        if sha256(p) != entry['sha256'] or entry['records_sha256'] != fm['videos'][video]['records_sha256']:
            raise ValueError('Current crop values/order changed')
        crops[video] = np.load(p, mmap_mode='r')
    sampler = SameFrameTemporalSampler(banks, config['fit'], args.seed, config['identities_per_frame'],
        config['observations_per_identity'], config['minimum_positive_gap'])
    pretrained = Path('/data3/liuyeqiang/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')
    if sha256(pretrained) != fm['imagenet_checkpoint_sha256']:
        raise ValueError('Fresh ImageNet initialization differs from frozen control')
    model = TrainableLocalFishReID(len(sampler.labels), config['dimension'], args.mode, config['local_count'])
    model.initialize_imagenet(pretrained)
    model.cuda()
    optimizer = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': config['backbone_lr']},
        {'params': model.head.parameters(), 'lr': config['head_lr']}], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.cuda.amp.GradScaler()
    augmentation_rng = np.random.default_rng(args.seed + 2000)
    spec = importlib.util.spec_from_file_location('declared_crop_augmentation', Path(__file__).with_name('cfp_train_reid.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sources = [__file__, 'cfp/trainable_local_identity.py', 'cfp/local_identity_encoder.py',
        'cfp/local_identity_training.py', 'cfp/crop_retrieval.py', 'cfp/reid.py', 'tools/cfp_train_reid.py']
    guard = dict(mode=args.mode, seed=args.seed, config_sha256=sha256(args.config),
        feature_manifest_sha256=sha256(args.features / 'manifest.json'), crop_manifest_sha256=sha256(args.crops / 'manifest.json'),
        initialization_sha256=sha256(pretrained), source_sha256={p: sha256(p) for p in sources},
        physical_devices=os.environ['CUDA_VISIBLE_DEVICES'], torch=torch.__version__, cudnn=torch.backends.cudnn.version())
    report = dict(status='RUNNING', guard=guard, config=config, epochs=[], evaluations={},
        classifier_keys=[[v, i] for v, i in sampler.labels], source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        parameters=sum(p.numel() for p in model.parameters()), training_seconds=0., scope=config['scope'],
        fitting_invariant='Actual backbone/head/BN updates use only4fit videos. Initialized fromImageNet only. Both outer and distinct inner excluded from gradients and BNstatistics. No oldfish pretrained parameters.')
    start_epoch = 0
    if args.resume:
        saved = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        if saved['guard'] != guard or json.loads((args.output / 'manifest.json').read_text())['status'] == 'COMPLETE':
            raise ValueError('Only identical incomplete visual run may resume')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        sampler.rng.bit_generator.state = saved['rng_sampler']
        augmentation_rng.bit_generator.state = saved['rng_augmentation']
        random.setstate(saved['rng_python'])
        np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch'])
        torch.cuda.set_rng_state_all(saved['rng_cuda'])
        start_epoch, report = saved['epoch'], saved['report']
    else:
        args.output.mkdir(parents=True)
        report['source_snapshots'] = snapshot_sources(args.output / 'sources', sources + [args.config])
    write_json(args.output / 'manifest.json', report)
    video_labels = {v: i for i, v in enumerate(config['fit'])}
    for epoch in range(start_epoch + 1, config['epochs'] + 1):
        if shutil.disk_usage(args.output).free < 40 * 2**30:
            raise OSError('40GiB reserve')
        model.train()
        losses = []
        overflow_skips = 0
        start = time.monotonic()
        for step in range(config['steps_per_epoch']):
            observations = sampler.sample()
            images = torch.from_numpy(np.array([crops[v][i] for v, i in observations])).cuda().float()
            visibility = torch.tensor(np.asarray([banks[v]['arrays']['visibility8'][i] for v, i in observations]), device='cuda', dtype=torch.float32)
            flips = augmentation_rng.random(len(images)) < config['augmentation']['horizontal_flip_probability']
            images[flips] = torch.flip(images[flips], (-1,))
            visibility[flips] = torch.flip(visibility[flips], (-1,))
            # Reuse declared erasing only; flips were already paired with masks.
            erase_config = dict(config['augmentation'], horizontal_flip_probability=0.)
            images = helper.augment(images, augmentation_rng, erase_config)
            labels = torch.tensor([sampler.labels[(v, banks[v]['records'][i]['identity'])] for v, i in observations], device='cuda')
            frames = torch.tensor([banks[v]['records'][i]['frame'] for v, i in observations], device='cuda')
            videos = torch.tensor([video_labels[v] for v, i in observations], device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                output = model(images, visibility)
            loss, parts = local_identity_objective(output, labels, frames, videos, config['minimum_positive_gap'],
                config['triplet_weight'], config['same_frame_weight'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite visual identity loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite_gradient = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if finite_gradient:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflow_skips += int(scaler.get_scale() < before)
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite visual parameter after AMP step')
            losses.append([float(loss.detach()), parts['ce'], parts['temporal'], parts['same_frame']])
        scheduler.step()
        seconds = time.monotonic() - start
        report['training_seconds'] += seconds
        entry = dict(epoch=epoch, loss_ce_temporal_frame=np.mean(losses, axis=0).tolist(), seconds=seconds,
                     attempted_steps=config['steps_per_epoch'], amp_overflow_skips=overflow_skips,
                     optimizer_updates=config['steps_per_epoch'] - overflow_skips)
        report['epochs'].append(entry)
        atomic_checkpoint(args.output / 'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            epoch=epoch, guard=guard, report=report, rng_sampler=sampler.rng.bit_generator.state,
            rng_augmentation=augmentation_rng.bit_generator.state, rng_python=random.getstate(), rng_numpy=np.random.get_state(),
            rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all()))
        write_json(args.output / 'manifest.json', report)
        print(json.dumps(entry), flush=True)
        if args.stop_after_epoch == epoch:
            report.update(status='CHECKPOINTED_EARLY_STOP', stopped_after_epoch=epoch)
            write_json(args.output / 'manifest.json', report)
            print(json.dumps({'status': report['status'], 'epoch': epoch}), flush=True)
            return
    model.eval()
    for video in (config['inner'], config['outer']):
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(crops[video]), 128):
                images = torch.from_numpy(np.array(crops[video][start:start + 128])).cuda().float()
                visible = torch.from_numpy(np.array(banks[video]['arrays']['visibility8'][start:start + 128])).cuda().float()
                with torch.cuda.amp.autocast():
                    out = model(images, visible)['embedding']
                embeddings.append(out.cpu().numpy())
        features = np.concatenate(embeddings)
        report['evaluations'][video] = embedding_crop_retrieval(features, banks[video]['records'], banks[video]['arrays']['valid'], 'cuda')
        write_json(args.output / 'manifest.json', report)
        print(video, report['evaluations'][video], flush=True)
    atomic_checkpoint(args.output / 'epoch8.pt', dict(model=model.state_dict(), guard=guard, config=config, classifier_keys=report['classifier_keys'], epoch=config['epochs']))
    report.update(status='COMPLETE', checkpoint_sha256=sha256(args.output / 'epoch8.pt'), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
