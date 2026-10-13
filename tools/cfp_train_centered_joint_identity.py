#!/usr/bin/env python3
"""Separate conditional ranking / balancedNULL cost repair with matched whole sets."""
import argparse
import hashlib
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch

from cfp.joint_identity_decoder import decode_joint_assignment
from cfp.centered_joint_identity import CenteredJointIdentity as JointFishIDDecoder, centered_joint_cost as bounded_joint_cost
from cfp.centered_joint_training import balanced_identity_loss, conditional_identity_ranking_loss
from cfp.joint_identity_training_data import load_joint_training, joint_frame_batch
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.matching import linear_assignment


def atomic_checkpoint(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def evaluate(model, records, banks, device, config, seed):
    model.eval()
    rng = np.random.default_rng(seed + 1000)
    counts = defaultdict(lambda: defaultdict(int))
    with torch.inference_mode():
        for record in records:
            inputs, supervision = joint_frame_batch(record, banks, device, rng, config['capacity'])
            q, i = banks[record['video']]['queries'], record['position']
            el, eh = q['edge_offsets'][i:i + 2]
            baseline = torch.from_numpy(q['baseline_cost'][el:eh].reshape(inputs['geometry'].shape).copy()).to(device)
            output = model(**inputs)
            residual = bounded_joint_cost(baseline, output['residual'], output['legal'], config['maximum_delta'])
            solutions = dict(full_id=decode_joint_assignment(output['logits'], inputs['detection_valid']))
            for name, costs in (('baseline_gru', baseline), ('centered_joint', residual)):
                matched, _, _ = linear_assignment(costs.cpu().numpy(), thresh=config['unmatched_cost'])
                assigned = torch.full((len(inputs['detections']),), -1, dtype=torch.int64, device=device)
                for track, detection in matched:
                    assigned[int(detection)] = int(track)
                solutions[name] = assigned
            target = supervision['target']
            known = target >= 0
            null = target == len(inputs['history'])
            physical_target = torch.where(null, -torch.ones_like(target), target)
            for name, assigned in solutions.items():
                c = counts[name]
                c['frames'] += 1
                c['detections'] += len(target)
                c['supervised'] += int(known.sum())
                c['existing'] += int((known & ~null).sum())
                c['null'] += int((known & null).sum())
                c['correct'] += int((known & (assigned == physical_target)).sum())
                c['existing_correct'] += int((known & ~null & (assigned == physical_target)).sum())
                c['null_correct'] += int((known & null & (assigned == -1)).sum())
                c['false_old_match_on_null'] += int((known & null & (assigned >= 0)).sum())
                c['unmatched_existing'] += int((known & ~null & (assigned == -1)).sum())
    return {name: dict(c, accuracy_percent=100 * c['correct'] / c['supervised'] if c['supervised'] else None,
                      existing_accuracy_percent=100 * c['existing_correct'] / c['existing'] if c['existing'] else None,
                      null_recall_percent=100 * c['null_correct'] / c['null'] if c['null'] else None)
            for name, c in counts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--architecture', choices=['joint', 'pair', 'cross_transformer'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN' or config['epochs'] < 1 or config['steps_per_epoch'] < 1:
        raise ValueError('Frozen finite training configuration required')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Existing experimental assets must be preserved')
    if shutil.disk_usage(args.output.parent).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    if args.device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Explicit idle physical GPU selection required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or args.seed not in config['seed_plan'] or args.architecture not in config['architectures']:
        raise ValueError('Registered CPU-only seed/model required')
    if args.stop_after_epoch is not None and not 1<=args.stop_after_epoch<config['epochs']:
        raise ValueError('Early checkpoint epoch must precede finalepoch')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.25)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(args.seed)
    records, banks, manifest = load_joint_training(args.queries)
    partitions = [config[name] for name in ('fit_videos', 'calibration_videos', 'outer_videos')]
    if len(set(sum(partitions, []))) != len(sum(partitions, [])) or set(sum(partitions, [])) != set(manifest['videos']):
        raise ValueError('Disjoint complete video partitions required')
    groups = {video: [r for r in records if r['video'] == video] for video in config['fit_videos']}
    model = JointFishIDDecoder(dimension=config['dimension'], hidden=config['hidden'], heads=config['heads'],
                              capacity=config['capacity'], architecture=args.architecture).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    source_files = [__file__, 'cfp/joint_identity_decoder.py', 'cfp/joint_identity_supervision.py',
                    'cfp/joint_identity_training_data.py', 'cfp/predictive.py', 'cfp/provenance.py',
                    'cfp/centered_joint_identity.py','cfp/centered_joint_training.py',str(args.config)]
    guard = dict(config_sha256=sha256(args.config), query_manifest_sha256=sha256(args.queries / 'manifest.json'),
        source_sha256={path: sha256(path) for path in source_files}, architecture=args.architecture, seed=args.seed,
        torch=torch.__version__, device=args.device, physical_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    start_epoch = 0
    report = dict(status='RUNNING', guard=guard, config=config, epochs=[], evaluations={},
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        parameters=sum(p.numel() for p in model.parameters()), scope='Fixed-state frozen-feature full high-stage set conditional head probe. Inherited encoder/teacher exposures disclosed. This is not a full-video/endogenous MOT or complete online runtime result.',
        queries_scope=manifest['scope'], supervision=manifest['supervision'], elapsed_training_seconds=0.)
    if args.resume:
        checkpoint = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        if checkpoint['guard'] != guard or json.loads((args.output / 'manifest.json').read_text())['status'] == 'COMPLETE':
            raise ValueError('Only incomplete identical-source/data/config/runtime checkpoints may resume')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        rng.bit_generator.state = checkpoint['rng_numpy_generator']
        np.random.set_state(checkpoint['rng_numpy'])
        random.setstate(checkpoint['rng_python'])
        torch.set_rng_state(checkpoint['rng_torch'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
        start_epoch, report = checkpoint['epoch'], checkpoint['report']
    else:
        args.output.mkdir(parents=True)
        report['source_snapshots'] = snapshot_sources(args.output / 'sources', source_files)
    write_json(args.output / 'manifest.json', report)
    for epoch in range(start_epoch + 1, config['epochs'] + 1):
        if shutil.disk_usage(args.output).free < 40 * 2**30:
            raise OSError('40GiB reserve')
        if any(sha256(p)!=value for p,value in guard['source_sha256'].items()):
            raise ValueError('Active registered source changed')
        model.train()
        digest=hashlib.sha256()
        losses, selected_counts = [], 0
        updates=0
        separate_losses=defaultdict(list)
        start = time.monotonic()
        for step in range(config['steps_per_epoch']):
            video = config['fit_videos'][int(rng.integers(len(groups)))]
            record = groups[video][int(rng.integers(len(groups[video])))]
            inputs, supervision = joint_frame_batch(record, banks, args.device, rng, config['capacity'])
            digest.update((video+':'+str(record['position'])+';').encode())
            digest.update(inputs['codes'].cpu().numpy().tobytes())
            q, i = banks[video]['queries'], record['position']
            el, eh = q['edge_offsets'][i:i + 2]
            baseline = torch.from_numpy(q['baseline_cost'][el:eh].reshape(inputs['geometry'].shape).copy()).to(args.device)
            output = model(**inputs)
            if not len(inputs['detections']):
                continue
            identity_loss, parts = balanced_identity_loss(output['logits'], supervision, output['legal'],
                ranking_weight=config['ranking_weight'], consistency_weight=config['consistency_weight'])
            costs = bounded_joint_cost(baseline, output['residual'], output['legal'], config['maximum_delta'])
            residual_loss, residual_parts = conditional_identity_ranking_loss(costs, supervision, output['legal'],
                temperature=config['residual_temperature'], ranking_weight=config['ranking_weight'],
                consistency_weight=config['consistency_weight'])
            loss = identity_loss + config['residual_loss_weight'] * residual_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite joint loss')
            optimizer.zero_grad(set_to_none=True)
            if loss.requires_grad:
                loss.backward()
                if not report.get('first_gradient_norms'):
                    report['first_gradient_norms']={name:float(p.grad.detach().double().norm()) for name,p in model.named_parameters() if p.grad is not None}
                    report['first_gradient_parameters']=sum(p.numel() for p in model.parameters() if p.grad is not None)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite joint gradients')
                optimizer.step()
                updates+=1
            losses.append(float(loss.detach()))
            for name,value in [('full_balanced_identity',float(identity_loss.detach())),('conditional_centered_ranking',float(residual_loss.detach())),('full_existing',parts['existing']),('full_null',parts['null']),('ranking_existing',residual_parts['existing']),('ranking_null',residual_parts['null'])]:
                separate_losses[name].append(value)
            selected_counts += parts['supervised_detections']
        scheduler.step()
        seconds = time.monotonic() - start
        report['elapsed_training_seconds'] += seconds
        entry = dict(epoch=epoch, mean_loss=float(np.mean(losses)), sampled_frames=len(losses),
                     supervised_detections=selected_counts, seconds=seconds, optimizer_updates=updates,
                     sampled_input_sha256=digest.hexdigest(),objective_means={k:float(np.mean(v)) for k,v in separate_losses.items()})
        report['epochs'].append(entry)
        atomic_checkpoint(args.output / 'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), epoch=epoch, guard=guard, report=report,
            rng_numpy_generator=rng.bit_generator.state, rng_numpy=np.random.get_state(), rng_python=random.getstate(),
            rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if args.device == 'cuda' else []))
        write_json(args.output / 'manifest.json', report)
        print(json.dumps(entry), flush=True)
        if args.stop_after_epoch==epoch:
            report['status']='CHECKPOINTED_EARLY_STOP';write_json(args.output/'manifest.json',report)
            print(json.dumps(dict(status='CHECKPOINTED_EARLY_STOP',epoch=epoch)),flush=True)
            return
    for video in config['calibration_videos'] + config['outer_videos']:
        report['evaluations'][video] = evaluate(model, [r for r in records if r['video'] == video], banks, args.device, config, args.seed)
        write_json(args.output / 'manifest.json', report)
        print(video, json.dumps(report['evaluations'][video]), flush=True)
    atomic_checkpoint(args.output / 'epoch8.pt', dict(model=model.state_dict(), guard=guard, epoch=config['epochs']))
    report.update(status='COMPLETE', checkpoint_sha256=sha256(args.output / 'epoch8.pt'),
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
