#!/usr/bin/env python3
"""Fit a light reliability state on train-only actual/clean/corrupted histories."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
from scipy.stats import rankdata
import torch
from torch.nn import functional as F

from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recoverable_identity import RecoverableIdentityState, reliability_supervision_loss
from cfp.recovery_training_data import load_recovery_data, sampling_groups, sample_indices, make_batch


def auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    n, m = int(labels.sum()), int((~labels).sum())
    if not n or not m:
        return None
    return float((rankdata(scores)[labels].sum() - n * (n + 1) / 2) / (n * m))


def diagnostics(model, rows, features, labels, device, seed):
    result = {}
    model.eval()
    for view, noisy in [('clean', False), ('actual', False), ('actual_corrupted', True)]:
        counts = dict(queries=0, model_correct=0, mean_correct=0, N01=0, N10=0, no_correct_history=0,
                      no_correct_history_model_correct=0)
        token_truth, token_score, support_truth, support_score = [], [], [], []
        cosine, norms, predictions, purity, margins = [], [], [], [], []
        rng = np.random.default_rng(7000 + seed)
        with torch.inference_mode():
            for start in range(0, len(rows), 128):
                batch = make_batch(rows[start:start + 128], features, labels, device, rng,
                                   'actual' if noisy else view, perturb=noisy, masking=noisy)
                past, times, valid, frame, candidates, candidate_valid, positive, correct, known = batch
                prediction, available, token_logits, confidence = model.details(past, times, valid, frame)
                candidates = F.normalize(candidates, dim=-1)
                score = torch.einsum('bd,bkd->bk', prediction, candidates).masked_fill(~candidate_valid, -2.)
                mean = F.normalize((F.normalize(past, dim=-1) * valid[..., None]).sum(1), dim=-1)
                baseline = torch.einsum('bd,bkd->bk', mean, candidates).masked_fill(~candidate_valid, -2.)
                good, ordinary = score.argmax(1) == positive, baseline.argmax(1) == positive
                support = (correct & known & valid).any(1)
                counts['queries'] += len(past)
                counts['model_correct'] += int(good.sum())
                counts['mean_correct'] += int(ordinary.sum())
                counts['N01'] += int((good & ~ordinary).sum())
                counts['N10'] += int((~good & ordinary).sum())
                counts['no_correct_history'] += int((~support).sum())
                counts['no_correct_history_model_correct'] += int((~support & good).sum())
                selected = known & valid
                token_truth.extend(correct[selected].cpu().tolist())
                token_score.extend(torch.sigmoid(token_logits[selected]).cpu().tolist())
                support_truth.extend(support.cpu().tolist())
                support_score.extend(torch.sigmoid(confidence).cpu().tolist())
                target = candidates[torch.arange(len(past), device=device), positive]
                cosine.extend((prediction * target).sum(-1).cpu().tolist())
                alternatives = score.clone()
                alternatives[torch.arange(len(past), device=device), positive] = -2.
                margins.extend((score[torch.arange(len(past), device=device), positive] - alternatives.max(1).values).cpu().tolist())
                predictions.append(prediction.cpu().numpy())
        predicted = np.concatenate(predictions)
        counts.update(rank1=counts['model_correct'] / counts['queries'], mean_rank1=counts['mean_correct'] / counts['queries'],
                      net_corrections=counts['N01'] - counts['N10'], token_reliability_auc=auc(token_truth, token_score),
                      support_confidence_auc=auc(support_truth, support_score), target_cosine_mean=float(np.mean(cosine)),
                      positive_negative_margin_mean=float(np.mean(margins)), feature_variance_sum=float(predicted.var(0).sum()),
                      mean_prediction_vector_norm=float(np.linalg.norm(predicted.mean(0))),
                      interpretation='Candidate-rank query corrections, not online IDSW or proven restoration when no true past clue remains.')
        result[view] = counts
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--objective', choices=['reliability', 'association_only'], required=True)
    p.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh model directory or explicit completed-epoch resume required')
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN' or config['query_manifest_sha256'] != sha256(args.queries / 'manifest.json'):
        raise ValueError('Frozen query/config pair required')
    if shutil.disk_usage(args.queries).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows, features, labels, data_manifest = load_recovery_data(args.queries, args.seed)
    fit = [row for row in rows if row['sequence'] in config['fit_videos']]
    calibration = [row for row in rows if row['sequence'] in config['calibration_videos']]
    groups = sampling_groups(fit, config['fit_videos'])
    steps = math.ceil(len(fit) / config['batch_size'])
    device = torch.device('cuda')
    model = RecoverableIdentityState(config['hidden']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    rng = np.random.default_rng(args.seed)
    guard = dict(config_sha256=sha256(args.config), query_manifest_sha256=sha256(args.queries / 'manifest.json'),
                 feature_manifest_sha256=data_manifest['feature_manifest_sha256'], objective=args.objective, seed=args.seed,
                 torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                 sources={name: sha256(name) for name in [__file__, 'cfp/recoverable_identity.py', 'cfp/recovery_training_data.py']})
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(status='RUNNING', guard=guard, config=config, feature_root=data_manifest['feature_root'], rgb_encoder=data_manifest['rgb_encoder'],
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_snapshots=snapshot_sources(args.output, [__file__, 'cfp/recoverable_identity.py', 'cfp/recovery_training_data.py', args.config]),
                  trainable_parameters=sum(value.numel() for value in model.parameters()), fit_queries=len(fit), calibration_queries=len(calibration),
                  steps_per_epoch=steps, epochs=[], scope='Light token reliability/association state; frozen fish encoder, no dev fitting. Confidence diagnostic only, never a GT-driven inference gate.')
    first_epoch = 1
    if args.resume:
        old = json.loads((args.output / 'manifest.json').read_text())
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if old['status'] == 'COMPLETE' or checkpoint['guard'] != guard or checkpoint['epoch'] >= config['epochs']:
            raise ValueError('Resume guard or completed run differs')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        rng.bit_generator.state = checkpoint['numpy_rng']
        torch.set_rng_state(checkpoint['cpu_rng'].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint['cuda_rng']])
        random.setstate(checkpoint['python_rng'])
        record['epochs'] = checkpoint['epochs']
        first_epoch = checkpoint['epoch'] + 1
    write_json(args.output / 'manifest.json', record)
    for epoch in range(first_epoch, config['epochs'] + 1):
        model.train()
        sums = dict(loss=0., association=0., reliability=0., confidence=0.)
        started = time.perf_counter()
        for _ in range(steps):
            selected = sample_indices(groups, rng, config['batch_size'])
            batch = make_batch([fit[index] for index in selected], features, labels, device, rng, view='mixed', perturb=True, masking=True)
            past, times, valid, frame, candidates, candidate_valid, positive, correct, known = batch
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
                prediction, available, token_logits, confidence = model.details(past, times, valid, frame)
                candidate_norm = F.normalize(candidates, dim=-1)
                logits = (torch.einsum('bd,bkd->bk', prediction, candidate_norm) / .1).masked_fill(~candidate_valid, -1e4)
                association = F.cross_entropy(logits.float(), positive)
                reliability = reliability_supervision_loss(token_logits.float(), correct, known & valid)
                support = (correct & known & valid).any(1)
                confidence_loss = F.binary_cross_entropy_with_logits(confidence.float(), support.float())
                loss = association
                if args.objective == 'reliability':
                    loss = loss + config['loss_weights']['token_reliability'] * reliability + config['loss_weights']['recovery_confidence'] * confidence_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite recovery loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            scaler.step(optimizer)
            scaler.update()
            for key, value in [('loss', loss), ('association', association), ('reliability', reliability), ('confidence', confidence_loss)]:
                sums[key] += float(value.detach())
        record['epochs'].append(dict(epoch=epoch, **{key: value / steps for key, value in sums.items()}, seconds=time.perf_counter() - started))
        checkpoint = dict(epoch=epoch, guard=guard, model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                          numpy_rng=rng.bit_generator.state, cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(), epochs=record['epochs'])
        temp = args.output / 'last.pt.tmp'
        torch.save(checkpoint, temp)
        temp.replace(args.output / 'last.pt')
        write_json(args.output / 'manifest.json', record)
        print('Epoch', epoch, record['epochs'][-1], flush=True)
        if shutil.disk_usage(args.queries).free < 40 * 2**30:
            raise OSError('40GiB reserve after completed checkpoint')
    final = args.output / 'epoch8.pt'
    shutil.copyfile(args.output / 'last.pt', final)
    record.update(status='COMPLETE', checkpoint_sha256=sha256(final), diagnostics=diagnostics(model, calibration, features, labels, device, args.seed),
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    write_json(args.output / 'manifest.json', record)
    print('Complete:', args.output, 'parameters', record['trainable_parameters'], flush=True)


if __name__ == '__main__':
    main()
