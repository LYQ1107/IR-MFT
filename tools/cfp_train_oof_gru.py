#!/usr/bin/env python3
"""Video-excluded ordinary raw-coordinate GRU bootstrap, exact epoch resume.

Training GT-clean histories are explicit supervision, never labeled online traces.
The exported ordinary model produces current/past-only accepted states afterwards.
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

from cfp.predictive import LatentIdentityPredictor
from cfp.provenance import sha256, snapshot_sources, write_json


def load_bootstrap(root):
    root = Path(root)
    meta = json.loads((root / 'manifest.json').read_text())
    if meta['status'] != 'COMPLETE' or meta['partition'] != 'train':
        raise ValueError('Complete training-only clean bootstrap indices required')
    features_root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase2/causal_train_detection_features_postneck_seed0')
    fm = json.loads((features_root / 'manifest.json').read_text())
    rows, features = [], {}
    for seq, entry in meta['videos'].items():
        fp, qp = features_root / (seq + '.npz'), root / (seq + '.npz')
        if sha256(fp) != entry['feature_sha256'] or sha256(qp) != entry['sha256']:
            raise ValueError('Frozen bootstrap inputs changed')
        with np.load(fp, allow_pickle=False) as archive:
            features[seq] = archive['rgb']
        with np.load(qp, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        for i, frame in enumerate(data['frame']):
            lo, hi = data['candidate_offsets'][i:i + 2]
            indices, times = data['history_indices'][i], data['history_frames'][i]
            valid = indices >= 0
            if np.any(times[valid] >= frame) or not valid.any():
                raise ValueError('GT-clean training histories must still be strictly past')
            rows.append(dict(sequence=seq, frame=int(frame), identity=int(data['identity'][i]),
                indices=indices.copy(), times=times.copy(), candidates=data['candidates'][lo:hi].copy(),
                candidate_known=data['candidate_known'][lo:hi].copy(), positive=int(data['positive'][i])))
    return rows, features, fm


def batch(rows, features, device):
    width = max(len(r['candidates']) for r in rows)
    past, current = np.zeros((len(rows), 8, 512), np.float32), np.zeros((len(rows), width, 512), np.float32)
    times, valid, known = np.zeros((len(rows), 8), np.int64), np.zeros((len(rows), 8), bool), np.zeros((len(rows), width), bool)
    for i, row in enumerate(rows):
        valid[i] = row['indices'] >= 0
        past[i, valid[i]] = features[row['sequence']][row['indices'][valid[i]]]
        times[i] = row['times']
        n = len(row['candidates'])
        current[i, :n] = features[row['sequence']][row['candidates']]
        known[i, :n] = row['candidate_known']
    arrays = (past, times, valid, np.array([r['frame'] for r in rows]), current, known, np.array([r['positive'] for r in rows]))
    return tuple(torch.from_numpy(v).to(device) for v in arrays)


def diagnose(model, rows, features, device):
    correct, cosines = [], []
    model.eval()
    with torch.inference_mode():
        for lo in range(0, len(rows), 128):
            past, times, valid, frame, candidates, known, positive = batch(rows[lo:lo + 128], features, device)
            prototype, available = model(past, times, valid, frame)
            logits = torch.einsum('bd,bmd->bm', prototype, F.normalize(candidates, dim=-1)).masked_fill(~known, -torch.inf)
            correct.extend((logits.argmax(1) == positive).cpu().tolist())
            cosines.extend(logits[torch.arange(len(past), device=device), positive].cpu().tolist())
    return dict(clean_queries=len(rows), known_candidate_rank1_percent=100 * float(np.mean(correct)),
        mean_positive_cosine=float(np.mean(cosines)), scope='GT-clean diagnostic only; actual online history performance not established.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN' or args.output.exists() and not args.resume:
        raise ValueError('Frozen config and unique output required')
    if shutil.disk_usage(args.output.parent).free < 40 * 2**30:
        raise OSError('40 GiB reserve')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device(config['device'])
    if device.type == 'cuda':
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '1') or torch.cuda.device_count() != 1:
            raise ValueError('One explicitly reserved idle physical GPU 0/1')
        torch.cuda.set_per_process_memory_fraction(.1)
    torch.set_num_threads(2); torch.manual_seed(config['seed']); random.seed(config['seed'])
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(config['seed'])
    rows, features, fm = load_bootstrap(config['queries'])
    fit_videos, excluded = config['fit_videos'], [config['calibration_video'], config['validation_video']]
    if len(set(excluded)) != 2 or set(fit_videos) & set(excluded) or set(fit_videos + excluded) != set(features):
        raise ValueError('OOF video isolation violated')
    fit = [r for r in rows if r['sequence'] in fit_videos]
    groups = [np.array([i for i, r in enumerate(fit) if r['sequence'] == v], np.int64) for v in fit_videos]
    model = LatentIdentityPredictor(hidden=config['hidden']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    guard = dict(config_sha256=sha256(args.config), query_manifest_sha256=sha256(Path(config['queries']) / 'manifest.json'),
        feature_manifest_sha256=sha256('/data3/liuyeqiang/IR-MFT_assets/cfp/phase2/causal_train_detection_features_postneck_seed0/manifest.json'),
        runtime=dict(torch=torch.__version__, cudnn=torch.backends.cudnn.version(), device=str(device), physical_device=os.environ.get('CUDA_VISIBLE_DEVICES')),
        sources=snapshot_sources(args.output, [__file__, 'cfp/predictive.py', args.config]))
    report = dict(status='RUNNING', config=config, guard=guard, epochs=[], rgb_encoder=fm['rgb_encoder'],
        parameters=sum(p.numel() for p in model.parameters()), fit_queries=len(fit),
        features_root='/data3/liuyeqiang/IR-MFT_assets/cfp/phase2/causal_train_detection_features_postneck_seed0',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        scope='Ordinary raw512 residual GRU, clean training bootstrap from GT on fit videos only. '
              'No learned old teacher states. New actual states must come from subsequent causal replay. '
              'OOF conditional on frozen encoders/detector exposed to six/eight videos, not whole-system unseen data.')
    start_epoch = 0
    if args.resume:
        saved = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        if saved['guard'] != guard or saved['epoch'] >= config['epochs']:
            raise ValueError('Exact resume mismatch or complete run')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']; torch.set_rng_state(saved['torch_rng']); random.setstate(saved['python_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        report, start_epoch = saved['report'], saved['epoch']
    write_json(args.output / 'manifest.json', report)
    for epoch in range(start_epoch + 1, config['epochs'] + 1):
        model.train(); losses, order = [], []
        for step in range(config['steps_per_epoch']):
            positions = [int(rng.choice(groups[int(rng.integers(len(groups)))])) for _ in range(config['batch_size'])]
            order.extend(positions)
            past, times, valid, frame, candidates, known, positive = batch([fit[i] for i in positions], features, device)
            predicted, available = model(past, times, valid, frame)
            logits = torch.einsum('bd,bmd->bm', predicted, F.normalize(candidates, dim=-1)) / .1
            loss = F.cross_entropy(logits.masked_fill(~known, -torch.inf), positive)
            if not available.all() or not torch.isfinite(loss):
                raise ValueError('Invalid clean ordinary training batch')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach()))
        report['epochs'].append(dict(epoch=epoch, loss=float(np.mean(losses)), steps=len(losses),
            query_order_sha256=hashlib.sha256(np.asarray(order, np.int64).tobytes()).hexdigest()))
        payload = dict(model=copy.deepcopy(model.state_dict()), optimizer=optimizer.state_dict(), epoch=epoch,
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None, guard=guard, report=report)
        temporary = args.output / 'last.tmp'; torch.save(payload, temporary); os.replace(temporary, args.output / 'last.pt')
        write_json(args.output / 'manifest.json', report); print(report['epochs'][-1], flush=True)
    report.update(status='COMPLETE', calibration=diagnose(model, [r for r in rows if r['sequence'] == excluded[0]], features, device),
        validation=diagnose(model, [r for r in rows if r['sequence'] == excluded[1]], features, device),
        checkpoint_sha256=sha256(args.output / 'last.pt'))
    torch.save(dict(model=model.cpu().state_dict(), epoch=config['epochs'], config=config, guard=guard,
        teacher_removed=True, rgb_encoder=fm['rgb_encoder']), args.output / 'inference.pt')
    report['inference_sha256'] = sha256(args.output / 'inference.pt')
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
