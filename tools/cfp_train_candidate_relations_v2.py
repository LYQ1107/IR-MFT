#!/usr/bin/env python3
"""Task-aligned raw-relation predictive association, matched ordinary and EMA controls.

Unknown negatives are excluded; candidate-missing NULL examples remain explicit.
All current candidates and past accepted histories are genuine recorded inputs.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from cfp.candidate_relation_jepa import CandidateRelationStudent, CandidateRelationTarget, candidate_relations
from cfp.candidate_training_data import load_candidate_training, candidate_batch
from cfp.provenance import sha256, snapshot_sources, write_json


def save_checkpoint(path, payload):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def diagnose(model, rows, banks, device):
    counts = dict(queries=len(rows), known_positive=0, null_labelled=0,
        known_rank_correct=0, unknown_predicted=0, null_predicted=0, null_correct=0,
        N01=0, N10=0, no_known_correct_past=0)
    variance, scores_margin = [], []
    model.eval()
    with torch.inference_mode():
        for lo in range(0, len(rows), 128):
            selected = rows[lo:lo + 128]
            batch = candidate_batch(selected, banks, device)
            output = model(*batch[:7])
            scores = output['scores'].cpu().numpy()
            latent = output['predicted_latent'].cpu().numpy()
            width = scores.shape[1] - 1
            for i, row in enumerate(selected):
                m = len(row['candidates'])
                predicted = int(np.argmax(scores[i]))
                counts['no_known_correct_past'] += int(row['correct_history'] == 0)
                if predicted == width:
                    counts['null_predicted'] += 1
                if row['positive'] < m:
                    counts['known_positive'] += 1
                    good = predicted < m and row['edge_labels'][predicted] == 0
                    known = predicted == width or predicted < m and row['edge_labels'][predicted] >= 0
                    counts['known_rank_correct'] += int(good)
                    counts['unknown_predicted'] += int(not known)
                    original = row['assigned']
                    if known and original >= 0 and row['edge_labels'][original] >= 0:
                        old_good = row['edge_labels'][original] == 0
                        counts['N01'] += int(not old_good and good)
                        counts['N10'] += int(old_good and not good)
                    variance.append(latent[i, row['positive']])
                elif row['null_known']:
                    counts['null_labelled'] += 1
                    counts['null_correct'] += int(predicted == width)
                if m >= 2:
                    ordered = np.sort(scores[i, :m])
                    scores_margin.append(float(ordered[-1] - ordered[-2]))
    counts.update(rank1_percent=100 * counts['known_rank_correct'] / max(counts['known_positive'], 1),
        net_correction=counts['N01'] - counts['N10'],
        positive_prediction_variance_sum=float(np.stack(variance).var(0).sum()) if variance else None,
        margin_mean=float(np.mean(scores_margin)) if scores_margin else None,
        scope='Fixed actual-state local candidate decisions, not Hungarian/MOT gains. NULL and unknown prediction counts retained.')
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-epoch', type=int)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN' or args.output.exists() and not args.resume:
        raise ValueError('Frozen config and immutable output required')
    if shutil.disk_usage(args.output.parent).free < 40 * 2**30:
        raise OSError('40 GiB reserve')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if config['device'] == 'cuda' and (os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '1') or torch.cuda.device_count() != 1):
        raise ValueError('One explicit idle physical GPU 0/1 required')
    device = torch.device(config['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.1)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(config['seed']); random.seed(config['seed'])
    query_rng, perturb_rng = np.random.default_rng(config['seed']), np.random.default_rng(config['seed'] + 971)
    rows, banks, meta = load_candidate_training(config['queries'], config['encoder_seed'])
    fit_videos = config['fit_videos']
    excluded = [config['calibration_video'], config['validation_video']]
    if len(set(excluded)) != 2 or set(fit_videos) & set(excluded) or set(fit_videos + excluded) != set(meta['videos']):
        raise ValueError('Complete-video data isolation violated')
    if set(meta.get('producer_excluded_videos', [])) != set(excluded):
        raise ValueError('Relation trial requires genuinely video-excluded producer states')
    fit = [r for r in rows if r['sequence'] in fit_videos and r['null_known'] and len(r['candidates'])]
    val = [r for r in rows if r['sequence'] == config['validation_video']]
    cal = [r for r in rows if r['sequence'] == config['calibration_video']]
    groups = [np.array([i for i, r in enumerate(fit) if r['sequence'] == seq], np.int64) for seq in fit_videos]
    if not fit or any(not len(group) for group in groups):
        raise ValueError('Every fit video needs known candidate/NULL supervision')
    model = CandidateRelationStudent(latent=config['latent']).to(device)
    teacher = CandidateRelationTarget(model, config['ema_decay']).to(device) if config['ema'] else None
    drop_rng = np.random.default_rng(config['seed'] + 2461)
    missing_augmented = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    guard = dict(config_sha256=sha256(args.config), queries_manifest_sha256=sha256(Path(config['queries']) / 'manifest.json'),
        feature_manifest_sha256=meta['feature_manifest_sha256'], physical_manifest_sha256=meta['physical_manifest_sha256'],
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                     device=str(device), physical_device=os.environ.get('CUDA_VISIBLE_DEVICES')),
        sources=snapshot_sources(args.output, [__file__, 'cfp/candidate_relation_jepa.py', 'cfp/candidate_training_data.py',
            'cfp/predictive.py', 'cfp/recoverability.py', args.config]))
    initial_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    report = dict(status='RUNNING', config=config, guard=guard, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        parameters=sum(p.numel() for p in model.parameters()), initial_tensor_sha256=initial_hash,
        fit_queries=len(fit), calibration_queries=len(cal), validation_queries=len(val), epochs=[],
        rgb_encoder=meta['rgb_encoder'], optimizer_steps=0, ema_updates=0,
        evidence_mode='Visible raw RGB cosine/margin relations plus time/early-group/current geometry; cached past scalars ignored. Clean current/future identity-to-candidate relations are teacher-only.',
        producer_excluded_videos=meta['producer_excluded_videos'],
        scope='Actual ordinary producer excludes outer/calibration videos; frozen detector/encoder pretraining exposure remains. Fixed final epoch, no historic-dev selection. Teacher never enters student forward/export. Synthetic candidate-missing augmentation removes known positive and every unknown current label, leaving real known negatives; natural unknowns never become wrong.')
    start_epoch = 0
    if args.resume:
        saved = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        if saved['guard'] != guard or saved['epoch'] >= config['epochs']:
            raise ValueError('Exact resume changed input/config/source/runtime or already complete')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        if teacher:
            teacher.load_state_dict(saved['teacher']); teacher.updates = saved['ema_updates']
        query_rng.bit_generator.state = saved['query_rng']; perturb_rng.bit_generator.state = saved['perturb_rng']
        torch.set_rng_state(saved['torch_rng']); random.setstate(saved['python_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        drop_rng.bit_generator.state = saved['drop_rng']
        missing_augmented = saved['missing_augmented']
        report, start_epoch = saved['report'], saved['epoch']
    write_json(args.output / 'manifest.json', report)
    for epoch in range(start_epoch + 1, config['epochs'] + 1):
        losses, predictions, classification, order = [], [], [], []
        model.train()
        for step in range(config['steps_per_epoch']):
            positions = [int(query_rng.choice(groups[int(query_rng.integers(len(groups)))])) for _ in range(config['batch_size'])]
            order.extend(positions)
            selected = [fit[i] for i in positions]
            batch = candidate_batch(selected, banks, device, perturb_rng,
                masking=config['masking'], corruption=config['corruption'], target_time=config['target_time'])
            history, times, valid, frame, candidates, legal, evidence, supervised, positive, clean_target, target_valid = batch
            early_times = torch.tensor(np.array([r['times'][:3] for r in selected]), device=device)
            early = (times[:, :, None] == early_times[:, None]).any(-1) & valid
            if config.get('known_negative_missing_probability', 0.) > 0:
                for i, row in enumerate(selected):
                    negative = np.flatnonzero(row['edge_labels'] == 1)
                    if target_valid[i] and len(negative) and drop_rng.random() < config['known_negative_missing_probability']:
                        old_candidates, old_evidence = candidates[i].clone(), evidence[i].clone()
                        candidates[i].zero_(); evidence[i].zero_(); legal[i].fill_(False); supervised[i].fill_(False)
                        ids = torch.tensor(negative, device=device, dtype=torch.int64)
                        n = len(negative); candidates[i, :n] = old_candidates[ids]; evidence[i, :n] = old_evidence[ids]
                        legal[i, :n] = True; supervised[i, :n] = True; supervised[i, -1] = True; positive[i] = candidates.shape[1]
                        boxes = banks[row['sequence']]['boxes'][row['candidates'][negative]]
                        size = (boxes[:, 2:] - boxes[:, :2]).prod(1)
                        intersection = np.maximum(np.minimum(boxes[:, None, 2:], boxes[None, :, 2:]) - np.maximum(boxes[:, None, :2], boxes[None, :, :2]), 0).prod(-1)
                        overlap = intersection / np.maximum(size[:, None] + size[None] - intersection, 1e-8)
                        np.fill_diagonal(overlap, 0)
                        evidence[i, :n, 18] = torch.tensor(overlap.max(1), device=device)
                        missing_augmented += 1
            output = model(history, times, valid, frame, candidates, legal, evidence, early)
            ce = F.cross_entropy(output['scores'].masked_fill(~supervised, -torch.inf), positive)
            latent_loss = output['predicted_latent'].sum() * 0
            if config['prediction_weight'] > 0 and target_valid.any():
                target = teacher(clean_target, candidates, legal) if teacher else F.normalize(model.pair_encoder(candidate_relations(clean_target[:, None], candidates, legal).squeeze(2)).detach(), dim=-1)
                distances = 1 - (output['predicted_latent'] * target).sum(-1)
                supervised_prediction = legal & target_valid[:, None]
                latent_loss = distances[supervised_prediction].mean()
            loss = ce + config['prediction_weight'] * latent_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite objective, no optimizer/EMA update')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step(); report['optimizer_steps'] += 1
            if teacher:
                teacher.update(model, successful=True)
                report['ema_updates'] = teacher.updates
            losses.append(float(loss.detach())); classification.append(float(ce.detach())); predictions.append(float(latent_loss.detach()))
        report['epochs'].append(dict(epoch=epoch, missing_augmented_cumulative=missing_augmented, steps=len(losses), loss=float(np.mean(losses)),
            candidate_ce=float(np.mean(classification)), latent_prediction=float(np.mean(predictions)),
            sample_order_sha256=hashlib.sha256(np.asarray(order, np.int64).tobytes()).hexdigest()))
        payload = dict(model=copy.deepcopy(model.state_dict()), teacher=copy.deepcopy(teacher.state_dict()) if teacher else None,
            ema_updates=teacher.updates if teacher else 0, optimizer=optimizer.state_dict(), epoch=epoch,
            query_rng=query_rng.bit_generator.state, perturb_rng=perturb_rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
            python_rng=random.getstate(), drop_rng=drop_rng.bit_generator.state, missing_augmented=missing_augmented, guard=guard, report=report)
        save_checkpoint(args.output / 'last.pt', payload)
        write_json(args.output / 'manifest.json', report)
        print(report['epochs'][-1], flush=True)
        if args.stop_after_epoch == epoch and epoch < config['epochs']:
            report['status'] = 'INTERRUPTED_AT_EPOCH_BOUNDARY'
            write_json(args.output / 'manifest.json', report)
            return
    report.update(status='COMPLETE', calibration=diagnose(model, cal, banks, device),
        validation=diagnose(model, val, banks, device), checkpoint_sha256=sha256(args.output / 'last.pt'))
    save_checkpoint(args.output / 'inference.pt', dict(model=model.cpu().state_dict(), config=config,
        guard=guard, rgb_encoder=meta['rgb_encoder'], teacher_removed=True))
    report['inference_sha256'] = sha256(args.output / 'inference.pt')
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
