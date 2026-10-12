#!/usr/bin/env python3
"""Whole-video outer validation, inner video calibration, exact epoch resume.

This initial warm-start experiment is not a whole-tracker OOF validation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recoverability import FEATURE_NAMES
from cfp.recoverability_ablation import risk_model_from_config
from cfp.selective_metrics import risk_metrics


def load_data(root):
    meta = json.loads((root / 'manifest.json').read_text())
    if meta['status'] != 'COMPLETE' or meta['partition'] != 'train':
        raise ValueError('Complete training-only candidate graphs required')
    rows = []
    for seq, spec in meta['videos'].items():
        path = root / (seq + '.npz')
        if sha256(path) != spec['sha256']:
            raise ValueError('Candidate supervision drift')
        with np.load(path, allow_pickle=False) as data:
            # NPZ member access decompresses on every lookup. Materialize once,
            # otherwise per-row slicing repeatedly decompresses the whole bank.
            arrays = {key: data[key] for key in data.files}
            for i, frame in enumerate(arrays['frame']):
                lo, hi = arrays['candidate_offsets'][i:i + 2]
                rows.append(dict(sequence=seq, frame=int(frame), track_id=int(arrays['track_id'][i]),
                    features=arrays['features'][lo:hi].copy(), labels=arrays['risk_labels'][lo:hi].copy(),
                    assigned=int(arrays['assigned'][i]), positive=int(arrays['positive'][i]),
                    null_known=bool(arrays['null_known'][i]),
                    correct_history=int(arrays['correct_history'][i]), unknown_history=int(arrays['unknown_history'][i])))
    return rows, meta


def batch(rows, device):
    width = max(max(len(r['labels']) for r in rows), 1)
    features = np.zeros((len(rows), width, len(FEATURE_NAMES)), np.float32)
    legal = np.zeros((len(rows), width), bool)
    labels = np.full((len(rows), width), -1, np.float32)
    for i, row in enumerate(rows):
        size = len(row['labels'])
        features[i, :size], legal[i, :size], labels[i, :size] = row['features'], True, row['labels']
    return [torch.from_numpy(a).to(device) for a in (features, legal, labels)]


def predict(model, rows, device):
    values = []
    model.eval()
    with torch.inference_mode():
        for lo in range(0, len(rows), 256):
            selected = rows[lo:lo + 256]
            x, legal, _ = batch(selected, device)
            out = model(x, legal).cpu().numpy()
            values.extend(out[i, :len(r['labels'])].copy() for i, r in enumerate(selected))
    return values


def calibrate(rows, logits):
    x, y = [], []
    for r, score in zip(rows, logits):
        chosen = r['assigned']
        if chosen >= 0 and r['labels'][chosen] >= 0:
            x.append(score[chosen]); y.append(r['labels'][chosen])
    if not x or not 0 < sum(y) < len(y):
        return dict(temperature=1., bias=0., known=len(y), fitted=False)
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    log_temperature = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature, bias], max_iter=60, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x / log_temperature.clamp(-3, 3).exp() + bias, y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return dict(temperature=float(log_temperature.detach().clamp(-3, 3).exp()),
        bias=float(bias.detach()), known=len(y), fitted=True)


def diagnose(rows, logits, calibration):
    edge_p, edge_y, chosen_p, chosen_y = [], [], [], []
    rank = []
    n01 = n10 = 0
    for row, score in zip(rows, logits):
        scaled = score / calibration['temperature'] + calibration['bias']
        p = torch.tensor(scaled).sigmoid().numpy()
        edge_p.extend(p.tolist()); edge_y.extend(row['labels'].tolist())
        selected = row['assigned']
        if selected >= 0:
            chosen_p.append(p[selected]); chosen_y.append(row['labels'][selected])
        if len(score) and row['positive'] < len(score):
            predicted = int(np.argmin(score))
            if row['labels'][predicted] < 0:
                continue
            correct = row['labels'][predicted] == 0
            rank.append(correct)
            if selected >= 0 and row['labels'][selected] >= 0:
                base_correct = row['labels'][selected] == 0
                n01 += int(not base_correct and correct)
                n10 += int(base_correct and not correct)
    return dict(candidate_edges=risk_metrics(edge_p, edge_y), accepted_edges=risk_metrics(chosen_p, chosen_y),
        known_positive_rank_queries=len(rank), known_positive_rank1_percent=100 * float(np.mean(rank)) if rank else None,
        N01=n01, N10=n10, net_correction=n01 - n10,
        scope='Paired local candidate reranking, not changed Hungarian decisions or online ID corrections.')


def checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-epoch', type=int)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN' or config['validation_video'] == config['calibration_video']:
        raise ValueError('Frozen distinct video fitting/calibration/validation required')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Immutable training output')
    if shutil.disk_usage(args.output.parent).free < 40 * 2**30:
        raise OSError('40 GiB reserve')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device(config['device'])
    if device.type == 'cuda' and (os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '1') or torch.cuda.device_count() != 1):
        raise ValueError('Initially only one explicit physical GPU 0 or 1')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(config['seed']); torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    rows, meta = load_data(Path(config['queries']))
    fit_videos = config['fit_videos']
    excluded = [config['calibration_video'], config['validation_video']]
    if set(fit_videos) & set(excluded) or set(fit_videos + excluded) != set(meta['videos']):
        raise ValueError('Fit/calibration/outer-video isolation violated')
    fit = [r for r in rows if r['sequence'] in fit_videos and len(r['labels']) and np.any(r['labels'] >= 0)]
    cal = [r for r in rows if r['sequence'] == config['calibration_video']]
    val = [r for r in rows if r['sequence'] == config['validation_video']]
    groups = [np.array([i for i, r in enumerate(fit) if r['sequence'] == seq], dtype=np.int64) for seq in fit_videos]
    model = risk_model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    guard = dict(config_sha256=sha256(args.config), query_manifest_sha256=sha256(Path(config['queries']) / 'manifest.json'),
        runtime=dict(torch=torch.__version__, cudnn=torch.backends.cudnn.version(), device=str(device)),
        sources=snapshot_sources(args.output, [__file__, 'cfp/recoverability.py', 'cfp/recoverability_ablation.py', 'cfp/selective_metrics.py', args.config]))
    report = dict(status='RUNNING', config=config, guard=guard, fit_queries=len(fit), cal_queries=len(cal), val_queries=len(val),
        parameters=sum(v.numel() for v in model.parameters()), epochs=[],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        scope='New-module whole-video heldout on warm-start teacher histories; not genuine whole-system OOF. Unknowns excluded from risk supervision.')
    start_epoch = 0
    if args.resume:
        saved = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        if saved['guard'] != guard:
            raise ValueError('Exact resume source/config/input/runtime changed')
        if saved['epoch'] >= config['epochs']:
            raise ValueError('Completed training cannot resume')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['rng_numpy']; random.setstate(saved['rng_python'])
        torch.set_rng_state(saved['rng_torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['rng_cuda'])
        start_epoch = saved['epoch']; report = saved['report']
    write_json(args.output / 'manifest.json', report)
    if config['epochs'] == 0:
        checkpoint(args.output / 'last.pt', dict(model=copy.deepcopy(model.state_dict()), optimizer=optimizer.state_dict(), epoch=0, guard=guard, report=report, rng_numpy=rng.bit_generator.state, rng_python=random.getstate(), rng_torch=torch.get_rng_state(), rng_cuda=None))
    for epoch in range(start_epoch + 1, config['epochs'] + 1):
        losses, order = [], []
        model.train()
        for step in range(config['steps_per_epoch']):
            positions = [int(rng.choice(groups[int(rng.integers(len(groups)))])) for _ in range(config['batch_size'])]
            order.extend(positions)
            x, legal, y = batch([fit[i] for i in positions], device)
            logits = model(x, legal)
            known = (y >= 0) & legal
            loss = F.binary_cross_entropy_with_logits(logits[known], y[known])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            losses.append(float(loss.detach()))
        import hashlib
        report['epochs'].append(dict(epoch=epoch, loss=float(np.mean(losses)), steps=len(losses),
            sample_order_sha256=hashlib.sha256(np.asarray(order, np.int64).tobytes()).hexdigest()))
        checkpoint(args.output / 'last.pt', dict(model=copy.deepcopy(model.state_dict()),
            optimizer=optimizer.state_dict(), epoch=epoch, guard=guard, report=report,
            rng_numpy=rng.bit_generator.state, rng_python=random.getstate(), rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
        write_json(args.output / 'manifest.json', report)
        print(report['epochs'][-1], flush=True)
        if args.stop_after_epoch == epoch and epoch < config['epochs']:
            report['status'] = 'INTERRUPTED_AT_EPOCH_BOUNDARY'
            write_json(args.output / 'manifest.json', report)
            return
    cal_logits = predict(model, cal, device)
    calibration = calibrate(cal, cal_logits)
    val_logits = predict(model, val, device)
    report.update(status='COMPLETE', calibration=calibration,
        validation_calibrated=diagnose(val, val_logits, calibration),
        validation_uncalibrated=diagnose(val, val_logits, dict(temperature=1., bias=0.)),
        checkpoint_sha256=sha256(args.output / 'last.pt'))
    checkpoint(args.output / 'inference.pt', dict(model=model.cpu().state_dict(), config=config, guard=guard, calibration=calibration,
        teacher_removed=True, feature_names=FEATURE_NAMES))
    report['inference_sha256'] = sha256(args.output / 'inference.pt')
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
