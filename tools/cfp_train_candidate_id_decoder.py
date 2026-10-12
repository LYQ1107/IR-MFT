#!/usr/bin/env python3
"""Train a small MOTIP-inspired temporary-ID control on actual frame graphs.

Each step consumes64 track histories; whole-frame groups are sampled uniformly
by video and the last group is truncated to the remaining track budget. This
changes grouping/loss orientation, explicitly, relative to independent row CE.
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

from cfp.candidate_id_decoder import CandidateIDDecoder
from cfp.candidate_training_data import load_candidate_training, candidate_batch
from cfp.provenance import sha256, snapshot_sources, write_json


def frame_groups(rows):
    groups = {}
    for row in rows:
        if len(row['candidates']):
            groups.setdefault((row['sequence'], row['frame']), []).append(row)
    result = list(groups.values())
    for group in result:
        first = group[0]['candidates']
        if any(not np.array_equal(row['candidates'], first) for row in group):
            raise ValueError('All track histories must share the actual current candidate pool')
    return result


def graph_batch(rows, banks, device, perturb_rng=None, masking=False, corruption=False):
    batch = candidate_batch(rows, banks, device, perturb_rng, masking, corruption)
    history, times, valid, frame, candidates, _, _, *_ = batch
    labels = np.full(len(rows[0]['candidates']), -1, np.int64)
    observed = valid.any(1).cpu().numpy()
    identities = np.array([r['identity'] for r in rows])
    physical = banks[rows[0]['sequence']]['labels'][rows[0]['candidates']]
    for j, identity in enumerate(physical):
        if identity < 0:
            continue
        matches = np.flatnonzero(identities == identity)
        if len(matches) == 1 and observed[matches[0]]:
            labels[j] = matches[0]
        elif not len(matches) and np.all(identities >= 0):
            labels[j] = len(rows)  # Known current identity absent from supplied ID dictionary.
    return history, times, valid, frame, candidates[0, :len(labels)], torch.tensor(labels, device=device)


def diagnose(model, groups, banks, device):
    counts = dict(frame_graphs=len(groups), known_detections=0, correct_id=0,
                  known_positive=0, rank_correct=0, N01=0, N10=0, unknown_predicted=0)
    model.eval()
    with torch.inference_mode():
        for rows in groups:
            h, t, v, f, c, labels = graph_batch(rows, banks, device)
            probability = model(h, t, v, f, c).softmax(-1)
            known = labels >= 0
            counts['known_detections'] += int(known.sum())
            counts['correct_id'] += int((probability.argmax(1)[known] == labels[known]).sum())
            for i, row in enumerate(rows):
                if row['positive'] >= len(c):
                    continue
                counts['known_positive'] += 1
                chosen = int(probability[:, i].argmax())
                good = row['edge_labels'][chosen] == 0
                counts['rank_correct'] += int(good)
                counts['unknown_predicted'] += int(row['edge_labels'][chosen] < 0)
                original = row['assigned']
                if original >= 0 and row['edge_labels'][original] >= 0 and row['edge_labels'][chosen] >= 0:
                    old_good = row['edge_labels'][original] == 0
                    counts['N01'] += int(not old_good and good)
                    counts['N10'] += int(old_good and not good)
    counts.update(rank1_percent=100 * counts['rank_correct'] / max(counts['known_positive'], 1),
                  known_detection_id_accuracy=100 * counts['correct_id'] / max(counts['known_detections'], 1),
                  net_correction=counts['N01'] - counts['N10'],
                  scope='Pure per-track candidate ranking has no added NULL threshold. Frame ID accuracy includes explicit newborn class. Not Hungarian/MOT metrics.')
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['state'] != 'FROZEN' or args.output.exists() and not args.resume:
        raise ValueError('Frozen config/new immutable output required')
    if shutil.disk_usage(args.output.parent).free < 40 * 2**30:
        raise OSError('40 GiB reserve')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Only the reserved physical GPU0 for this control')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    device = torch.device('cuda')
    query_rng = np.random.default_rng(cfg['seed'])
    perturb_rng = np.random.default_rng(cfg['seed'] + 971)
    code_rng = np.random.default_rng(cfg['seed'] + 1973)
    rows, banks, metadata = load_candidate_training(cfg['queries'], cfg['encoder_seed'])
    fit_videos = cfg['fit_videos']
    excluded = [cfg['validation_video'], cfg['calibration_video']]
    if len(set(excluded)) != 2 or set(fit_videos) & set(excluded) or set(fit_videos + excluded) != set(metadata['videos']):
        raise ValueError('Whole-video fit/calibration/validation separation required')
    fit = frame_groups([r for r in rows if r['sequence'] in fit_videos])
    cal = frame_groups([r for r in rows if r['sequence'] == cfg['calibration_video']])
    val = frame_groups([r for r in rows if r['sequence'] == cfg['validation_video']])
    by_video = [np.array([i for i, g in enumerate(fit) if g[0]['sequence'] == s]) for s in fit_videos]
    if any(not len(g) for g in by_video):
        raise ValueError('All fit videos need frame graphs')
    model = CandidateIDDecoder(hidden=cfg['hidden'], capacity=cfg['capacity']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    guard = dict(config_sha256=sha256(args.config), query_manifest_sha256=sha256(Path(cfg['queries']) / 'manifest.json'),
                 feature_manifest_sha256=metadata['feature_manifest_sha256'], physical_manifest_sha256=metadata['physical_manifest_sha256'],
                 runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), physical_device='0'),
                 sources=snapshot_sources(args.output, [__file__, args.config, 'cfp/candidate_id_decoder.py', 'cfp/candidate_training_data.py', 'cfp/predictive.py']))
    initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    report = dict(status='RUNNING', config=cfg, guard=guard, parameters=sum(p.numel() for p in model.parameters()),
                  initial_tensor_sha256=initial, epochs=[], optimizer_steps=0, rgb_encoder=metadata['rgb_encoder'],
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  scope='MOTIP-inspired ID dictionary decoder, not official MOTIP. Warm actual states/backbone exposure. '
                        'Each step exactly64 supplied track histories, whole groups/truncated last group; '
                        'current detections classify among supplied IDs/newborn. Additional other-track raw histories '
                        'and group-oriented loss differ from per-row candidate JEPA; parameter/data budgets measured.')
    start = 0
    if args.resume:
        saved = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        if saved['guard'] != guard or saved['epoch'] >= cfg['epochs']:
            raise ValueError('Exact resume source/input/runtime changed or already complete')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        query_rng.bit_generator.state = saved['query_rng']; perturb_rng.bit_generator.state = saved['perturb_rng']
        code_rng.bit_generator.state = saved['code_rng']; torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng']); random.setstate(saved['python_rng'])
        report, start = saved['report'], saved['epoch']
    write_json(args.output / 'manifest.json', report)
    for epoch in range(start + 1, cfg['epochs'] + 1):
        losses, order, supervised_edges = [], [], 0
        model.train()
        for _ in range(cfg['steps_per_epoch']):
            remaining = cfg['batch_size']; optimizer.zero_grad(set_to_none=True); step_loss = 0.
            while remaining:
                index = int(query_rng.choice(by_video[int(query_rng.integers(len(by_video)))]))
                selected = fit[index][:remaining]; n = len(selected); remaining -= n; order.extend([index, n])
                h, t, v, f, c, labels = graph_batch(selected, banks, device, perturb_rng, cfg['masking'], cfg['corruption'])
                codes = torch.tensor(code_rng.choice(cfg['capacity'], n, replace=False), device=device)
                logits = model(h, t, v, f, c, codes)
                known = labels >= 0
                if known.any():
                    loss = F.cross_entropy(logits[known], labels[known]) * n / cfg['batch_size']
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite supervised ID loss')
                    loss.backward(); step_loss += float(loss.detach()); supervised_edges += int(known.sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step(); report['optimizer_steps'] += 1; losses.append(step_loss)
        report['epochs'].append(dict(epoch=epoch, loss=float(np.mean(losses)), steps=len(losses),
            track_histories=len(losses) * cfg['batch_size'], known_detection_targets=supervised_edges,
            sample_order_sha256=hashlib.sha256(np.array(order, np.int64).tobytes()).hexdigest()))
        payload = dict(model=copy.deepcopy(model.state_dict()), optimizer=optimizer.state_dict(), epoch=epoch,
                       guard=guard, query_rng=query_rng.bit_generator.state, perturb_rng=perturb_rng.bit_generator.state,
                       code_rng=code_rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(), report=report)
        temporary = args.output / 'last.tmp'; torch.save(payload, temporary); temporary.replace(args.output / 'last.pt')
        write_json(args.output / 'manifest.json', report); print(report['epochs'][-1], flush=True)
        if args.stop_after_epoch == epoch and epoch < cfg['epochs']:
            report['status'] = 'INTERRUPTED_AT_EPOCH_BOUNDARY'; write_json(args.output / 'manifest.json', report); return
    report.update(status='COMPLETE', calibration=diagnose(model, cal, banks, device), validation=diagnose(model, val, banks, device),
                  checkpoint_sha256=sha256(args.output / 'last.pt'))
    torch.save(dict(model=model.cpu().state_dict(), config=cfg, guard=guard, rgb_encoder=metadata['rgb_encoder'], teacher_removed=True), args.output / 'inference.pt')
    report['inference_sha256'] = sha256(args.output / 'inference.pt'); write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
