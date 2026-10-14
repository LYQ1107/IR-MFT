#!/usr/bin/env python3
"""Separate frozen original/MAE encoder identity probes, with exact resume and small head states."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch

from cfp.crop_retrieval import embedding_crop_retrieval
from cfp.dinov2_fish_data import load_crop_banks, pixel_batch, augment_observations
from cfp.mae_identity_probe import FrozenEncoderIdentityProbe, probe_identity_loss
from cfp.dinov2_identity import load_pinned_small
from cfp.local_identity_training import SameFrameTemporalSampler, local_identity_objective
from cfp.provenance import sha256, snapshot_sources, write_json


def checkpoint(path, payload):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.backbone.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_norms(trainable):
    """Overflow is explicit JSON null; finite gradients use stable FP64 norms."""
    result = {}
    for name, parameter in trainable.items():
        if parameter.grad is not None:
            value = float(parameter.grad.double().norm())
            result[name] = value if math.isfinite(value) else None
    return result


def check_gpu(physical):
    if physical not in ('0', '1', '2', '3'):
        raise ValueError('Exactly one approved idle physical GPU0-3 required')
    query = subprocess.check_output(['nvidia-smi', '-i', physical,
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    memory, utilization = [int(x.strip()) for x in query.split(',')]
    if memory > 64 or utilization:
        raise ValueError('Selected GPU is not idle; never interfere with other jobs')


def main():
    ap = argparse.ArgumentParser()
    for name in ('config', 'crops', 'features', 'index', 'reference', 'weights', 'source-proof', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--outer', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--local-count', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--encoder-init', type=Path)
    ap.add_argument('--loss-mode', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--stop-after-epoch', type=int)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    if (config['state'] != 'FROZEN' or args.mode not in config['modes'] or args.loss_mode not in config['loss_modes'] or args.seed not in config['seed_plan']
            or args.local_count not in config['local_counts'] or args.outer not in config['videos']
            or args.output.exists() and not args.resume or args.output.resolve().is_relative_to(Path.cwd().resolve())):
        raise ValueError('Registered config/fold/mode/seed/K and new outside-Git output or identical resume required')
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch < config['epochs']:
        raise ValueError('Deliberate restart stop must precede final epoch')
    cycle = config['videos']
    inner = cycle[(cycle.index(args.outer)+1) % len(cycle)]
    fit = [v for v in cycle if v not in (args.outer, inner)]
    if len(fit) != 4 or args.outer == inner:
        raise ValueError('Distinct heldout outer and calibration inner plus four fitting videos required')
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    check_gpu(physical)
    if shutil.disk_usage(args.output.parent).free < 40*2**30:
        raise OSError('40GiB reserve')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':raise ValueError('Declared deterministic CUDA workspace required')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    banks, dimensions, fm = load_crop_banks(args.crops, args.features, args.index)
    sampler = SameFrameTemporalSampler(banks, fit, args.seed, config['identities_per_frame'],
        config['observations_per_identity'], config['minimum_positive_gap'])
    assert not any(v in (args.outer, inner) for v, _ in sampler.labels)
    weight_manifest = json.loads((args.weights.parent/'manifest.json').read_text())
    if weight_manifest['status'] != 'COMPLETE' or sha256(args.weights) != fm['weight_sha256']:
        raise ValueError('Exact fresh official pretrained initialization required')
    proofs = json.loads(args.source_proof.read_text())['imported_source_proofs']
    backbone = load_pinned_small(args.reference, args.weights, proofs)
    init_manifest = None
    if args.mode != 'original':
        if args.encoder_init is None:raise ValueError('Exactmatchingseed pretrain initialization required')
        init_manifest=json.loads((args.encoder_init/'manifest.json').read_text());g=init_manifest['guard'];ic=init_manifest['config']
        if (init_manifest['status']!='COMPLETE_UNLABELED_VISIBLE_TOKEN_PRETRAINING' or g['mode']!=args.mode or g['seed']!=args.seed or set(ic['fit_videos'])!=set(fit) or set(ic['excluded_videos'])!=set([inner,args.outer]) or g['pretrained_sha256']!=sha256(args.weights) or sha256(args.encoder_init/'encoder_final.pt')!=init_manifest['encoder_checkpoint_sha256'] or any(sha256(p)!=h for p,h in g['source_sha256'].items())):raise ValueError('Exact4fit/2excluded/seed/source completedpretrain required')
        saved_encoder=torch.load(args.encoder_init/'encoder_final.pt',map_location='cpu',weights_only=False);backbone.load_state_dict(saved_encoder['backbone'],strict=True)
    elif args.encoder_init is not None:raise ValueError('OriginalDINO initialization mustnotconsume fishpretrain')
    model = FrozenEncoderIdentityProbe(backbone,len(sampler.labels),args.loss_mode)
    frozen_initial = frozen_digest(model)
    initial_tail = {n: p.detach().clone() for n, p in model.backbone.named_parameters() if p.requires_grad}
    check_gpu(physical)
    torch.cuda.set_per_process_memory_fraction(.4)
    model.cuda()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['head_lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.cuda.amp.GradScaler()
    augmentation_rng = np.random.default_rng(args.seed+2000)
    sources = [__file__, 'cfp/mae_identity_probe.py', 'cfp/dinov2_fish_data.py', 'cfp/dinov2_identity.py',
        'cfp/local_identity_encoder.py', 'cfp/local_identity_training.py', 'cfp/crop_retrieval.py', 'cfp/reid.py']
    guard = dict(mode=args.mode,loss_mode=args.loss_mode,encoder_init_sha256=init_manifest['encoder_checkpoint_sha256'] if init_manifest else sha256(args.weights),encoder_manifest_sha256=sha256(args.encoder_init/'manifest.json') if init_manifest else None, seed=args.seed, local_count=args.local_count, outer=args.outer, inner=inner,
        fit=fit, config_sha256=sha256(args.config), crops_sha256=sha256(args.crops/'manifest.json'),
        features_sha256=sha256(args.features/'manifest.json'), index_sha256=sha256(args.index/'manifest.json'),
        pretrained_sha256=sha256(args.weights), source_sha256={p: sha256(p) for p in sources},
        imported_source_proofs=proofs, physical_devices=physical, torch=torch.__version__,
        cudnn=torch.backends.cudnn.version(), frozen_prefix_sha256=frozen_initial)
    report = dict(status='RUNNING', guard=guard, config=config, epochs=[], evaluations={},
        classifier_keys=[[v, i] for v, i in sampler.labels],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in trainable.values()),
        trainable_backbone_parameters=sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
        trainable_head_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        training_seconds=0., scope=config['scope'], fitting_invariant='Separate4fit-video frozenbackbone identityprojection/classifier only; noheldoutgradient/state, encoderweights frozenall175keys. CE-only/Triplet-only exactloss; activeunusedclassifierfrozenandreported. Matchedoriginal/nearfull-MAE initialization provenance retained.')
    start_epoch = 0
    if args.resume:
        saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
        if saved['guard'] != guard or json.loads((args.output/'manifest.json').read_text())['status'] == 'COMPLETE':
            raise ValueError('Only identical incomplete training may resume')
        if json.loads((args.output/'manifest.json').read_text())['status']!='CHECKPOINTED_EARLY_STOP':raise ValueError('Completedassets immutable; onlydeliberatestop resumes')
        model.load_head(saved['head'])
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
        report['source_snapshots'] = snapshot_sources(args.output/'sources', sources+[args.config])
    write_json(args.output/'manifest.json', report)
    video_index = {v: i for i, v in enumerate(fit)}
    for epoch in range(start_epoch+1, config['epochs']+1):
        model.train()
        losses, coverage = [], []
        sampled_digest=hashlib.sha256()
        overflow_skips = 0
        started = time.monotonic()
        for step in range(config['steps_per_epoch']):
            observations = sampler.sample()
            images, masks = pixel_batch(observations, banks, dimensions)
            sampled_digest.update(json.dumps(dict(observations=observations,augmentation=augmentation_rng.bit_generator.state),sort_keys=True).encode())
            images, masks = augment_observations(images, masks, augmentation_rng, config['augmentation'])
            labels = torch.tensor([sampler.labels[(v, banks[v]['records'][i]['identity'])] for v, i in observations], device='cuda')
            frames = torch.tensor([banks[v]['records'][i]['frame'] for v, i in observations], device='cuda')
            videos = torch.tensor([video_index[v] for v, i in observations], device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                output = model(images, masks)
            valid = output['crop_valid']
            if not valid.any():
                raise ValueError('No observed content in entire sampled training batch')
            loss, parts = probe_identity_loss(output,labels,frames,videos,args.loss_mode,config['minimum_positive_gap'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite identity loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if any(p.grad is not None for p in model.backbone.parameters() if not p.requires_grad):
                raise ValueError('Frozen prefix received a fitting gradient')
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable.values())
            if epoch == 1 and step == 0:
                report['first_step_gradient_norms'] = gradient_norms(trainable)
                report['first_step_gradients_finite'] = bool(finite)
            if finite and 'first_finite_step_gradient_norms' not in report:
                report['first_finite_step_gradient_norms'] = gradient_norms(trainable)
                report['first_finite_step'] = dict(epoch=epoch, step=step)
                if not any(value is not None and value > 0 for n,value in report['first_finite_step_gradient_norms'].items() if n.startswith('projection.')):raise ValueError('Identity projection hasnoactualgradient')
            if finite:
                torch.nn.utils.clip_grad_norm_(trainable.values(), 5.)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflow_skips += int(scaler.get_scale() < before)
            if not all(torch.isfinite(p).all() for p in trainable.values()):
                raise ValueError('Nonfinite trained parameter')
            losses.append([float(loss.detach()), parts['ce'], parts['triplet'], 0.])
            coverage.append([int(valid.sum()), 0, parts['temporal_coverage']])
        scheduler.step()
        if 'first_finite_step_gradient_norms' not in report:
            raise ValueError('No finite visual gradient during first epoch')
        if frozen_digest(model) != frozen_initial:
            raise ValueError('Frozen pretrained prefix changed')
        seconds = time.monotonic()-started
        report['training_seconds'] += seconds
        entry = dict(epoch=epoch, sampled_input_sha256=sampled_digest.hexdigest(), loss_ce_temporal_frame=np.mean(losses, axis=0).tolist(),
            sample_sameframe_temporal_coverage=np.mean(coverage, axis=0).tolist(), seconds=seconds,
            attempted_steps=config['steps_per_epoch'], amp_overflow_skips=overflow_skips,
            optimizer_updates=config['steps_per_epoch']-overflow_skips)
        report['epochs'].append(entry)
        checkpoint(args.output/'last.pt', dict(epoch=epoch, guard=guard, report=report,
            head=model.head_state(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(), rng_sampler=sampler.rng.bit_generator.state,
            rng_augmentation=augmentation_rng.bit_generator.state, rng_python=random.getstate(),
            rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all()))
        write_json(args.output/'manifest.json', report)
        print(json.dumps(entry), flush=True)
        if shutil.disk_usage(args.output).free < 40*2**30:
            raise OSError('40GiB reserve')
        if args.stop_after_epoch == epoch:
            report.update(status='CHECKPOINTED_EARLY_STOP', stopped_after_epoch=epoch)
            write_json(args.output/'manifest.json', report)
            print(json.dumps({'status': report['status'], 'epoch': epoch}), flush=True)
            return
    report['frozen_prefix_unchanged'] = True
    report['trained_backbone_l2_change'] = 0.0
    if frozen_digest(model)!=frozen_initial:raise ValueError('Frozenencoderchanged duringidentityprobe')
    model.eval()
    for video in (inner, args.outer):
        directory = args.output/video
        directory.mkdir(exist_ok=args.resume)
        shutil.copyfile(args.features/video/'records.json', directory/'records.json')
        valid = banks[video]['arrays']['valid']
        records = banks[video]['records']
        shapes = {'embedding':(384,), 'raw_global':(384,)}
        arrays = {n:np.zeros((len(records),*shape),np.float32) for n,shape in shapes.items()}
        with torch.inference_mode():
            for start in range(0, len(records), 128):
                observations = [(video, i) for i in range(start, min(start+128, len(records)))]
                images, masks = pixel_batch(observations, banks, dimensions)
                output = model(images, masks)  # FP32 matches frozen diagnostic.
                for name in shapes:
                    arrays[name][start:start+len(observations)] = output[name].cpu().numpy()
        artifacts = {n:dict(canonical_tensor_bytes_sha256=hashlib.sha256(value.tobytes()).hexdigest(),shape=list(value.shape),dtype=str(value.dtype),storage='TransientRAMonly, reproduciblefromencoder/head/data SHA; nofeaturecache retained') for n,value in arrays.items()}
        report['evaluations'][video] = dict(
            retrieval=embedding_crop_retrieval(arrays['embedding'], records, valid, device='cuda'),
            raw_global_retrieval=embedding_crop_retrieval(arrays['raw_global'], records, valid, device='cuda'),
            artifacts=artifacts, records_sha256=sha256(directory/'records.json'))
        write_json(args.output/'manifest.json', report)
        print(video, report['evaluations'][video]['retrieval'], flush=True)
    checkpoint(args.output/'final.pt', dict(head=model.head_state(), guard=guard, config=config,
        classifier_keys=report['classifier_keys'], epoch=config['epochs']))
    report.update(status='COMPLETE', checkpoint_sha256=sha256(args.output/'final.pt'),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(args.output/'manifest.json', report)


if __name__ == '__main__':
    main()
