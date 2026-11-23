#!/usr/bin/env python3
"""Two ordinary Train-only confidence maps; no auxiliary pixels or detector fit."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.models.yolo_head import YOLOXHead
from yolox.tracker.matching import ious
from yolox.utils import postprocess


def main():
    torch.set_num_threads(2)
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text()); work = Path(plan['work_root'])
    config_path = ROOT/'configs/perceptual_RGB_score_calibration_20261123.json'
    spec = json.loads(config_path.read_text()); parent_path = ROOT/'configs/perceptual_correlation_repair_20261123.json'
    assert sha256(parent_path) == spec['parent_sha256']
    parent = json.loads(parent_path.read_text()); protocol = json.loads((ROOT/plan['protocol']).read_text())
    out = work/'RGB_score_calibration'; out.mkdir(exist_ok=False)
    checkpoint = Path(parent['source_RGB_checkpoint']); assert sha256(checkpoint) == parent['source_RGB_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    head = YOLOXHead(1, width=1.25)
    head.load_state_dict({k.removeprefix('head.'): v for k, v in saved['model'].items() if k.startswith('head.')}, strict=True)
    del saved
    head = head.float().eval().requires_grad_(False); sources = {}; started = time.perf_counter()
    dump(out/'state.json', dict(status='PRODUCING_TRAIN_RGB_HEAD_PREDICTIONS_NO_GT', Test_read=False))
    with torch.inference_mode():
        for sequence in protocol['train']:
            directory = work/'correlation_cache'/sequence/'features'
            meta = json.loads((directory/'complete.json').read_text())
            assert meta['config_sha256'] == sha256(parent_path) and meta['Dev_rows'] == meta['Test_rows'] == 0
            for level in [3, 4, 5]:
                assert sha256(directory/f'rgb_P{level}.npy') == meta['outputs'][f'rgb_P{level}']['sha256']
            banks = [np.load(directory/f'rgb_P{level}.npy', mmap_mode='r') for level in [3, 4, 5]]
            results = []
            for first in range(0, len(banks[0]), 4):
                features = [torch.from_numpy(np.array(x[first:first+4])).float() for x in banks]
                prediction = head(features)
                values = postprocess(prediction, 1, .1, .7)
                for value in values:
                    if value is None:
                        results.append(np.empty((0, 5), np.float32))
                    else:
                        z = value.numpy(); results.append(np.c_[z[:, :4], z[:, 4]*z[:, 5]])
            path = out/(sequence+'.npz')
            np.savez_compressed(path, boxes=np.concatenate(results), offsets=np.r_[0, np.cumsum([len(v) for v in results])])
            sources[sequence] = dict(path=str(path), sha256=sha256(path), source_manifest_sha256=sha256(directory/'complete.json'))
            print(json.dumps(dict(sequence=sequence, frames=len(results), frozen_predictions=True, seconds=time.perf_counter()-started)), flush=True)
    dump(out/'predictions_frozen.json', dict(status='ALL1152_RGB_TRAIN_PREDICTIONS_FROZEN_BEFORE_GT', sources=sources,
        config_sha256=sha256(config_path), source_RGB_sha256=sha256(checkpoint), GT_read=False, source_sha256=sha256(__file__)))
    logits = []; labels = []; weights = []; counts = {}
    for sequence, source in sources.items():
        assert sha256(source['path']) == source['sha256']
        directory = work/'correlation_cache'/sequence/'features'; meta = json.loads((directory/'complete.json').read_text())
        path = Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        assert sha256(path) == protocol['derived_gt_sha256']['train/'+sequence]
        gt = np.loadtxt(path, delimiter=',', ndmin=2); xs = []; ys = []
        with np.load(source['path']) as data:
            for i, frame in enumerate(meta['frame_ids']):
                lo, hi = data['offsets'][i:i+2]; detections = data['boxes'][lo:hi]
                truth = gt[gt[:, 0] == frame]
                positive = truth[(truth[:, 6] > 0) & (truth[:, 7] == 1), 2:6].copy()
                ignored = truth[truth[:, 7] == 8, 2:6].copy()
                for boxes in [positive, ignored]:
                    boxes[:, 2:] += boxes[:, :2]; boxes *= meta['ratios'][i]
                good = ious(detections[:, :4], positive).max(1) if len(positive) else np.zeros(len(detections))
                ignore = ious(detections[:, :4], ignored).max(1) if len(ignored) else np.zeros(len(detections))
                y = np.where(good >= .5, 1, np.where((good <= .2) & (ignore <= .2), 0, -1))
                selected = y >= 0; score = np.clip(detections[selected, 4], 1e-6, 1-1e-6)
                xs.append(np.log(score/(1-score))); ys.append(y[selected])
        x = np.concatenate(xs); y = np.concatenate(ys); assert len(y)
        logits.append(x); labels.append(y); weights.append(np.full(len(y), 1/len(y)))
        counts[sequence] = dict(rows=len(y), positives=int(y.sum()), negatives=int((y == 0).sum()))
    x, y, weight = np.concatenate(logits), np.concatenate(labels), np.concatenate(weights)
    weight *= len(weight)/weight.sum()
    def loss(log_temperature):
        value = x*np.exp(-log_temperature)
        return float(np.average(np.logaddexp(0, value)-y*value, weights=weight))
    fit = minimize_scalar(loss, bounds=(-4., 4.), method='bounded', options=dict(xatol=1e-9))
    assert fit.success
    platt = LogisticRegression(C=1., max_iter=1000, tol=1e-8, random_state=0)
    platt.fit(x[:, None], y, sample_weight=weight)
    assert platt.n_iter_[0] < 1000 and platt.coef_[0, 0] > 0
    models = dict(identity=dict(slope=1., intercept=0.),
                  temperature=dict(slope=float(np.exp(-fit.x)), intercept=0., temperature=float(np.exp(fit.x))),
                  platt=dict(slope=float(platt.coef_[0, 0]), intercept=float(platt.intercept_[0])))
    report = dict(status='COMPLETE_BOTH_RGB_TRAIN_CALIBRATIONS_BEFORE_DEV', models=models, counts=counts,
        Train_rows=len(y), Train_videos=protocol['train'], Dev_fit_rows=0, Test_read=False,
        config_sha256=sha256(config_path), source_sha256=sha256(__file__),
        predictions_frozen_sha256=sha256(out/'predictions_frozen.json'), seconds=time.perf_counter()-started,
        limitation='Train head FP32 on frozen FP16FPN, no backbone forward. Dev uses canonical FP16-inference NMS packets. No new below0.1 proposals admitted, no calibrated detector retraining or IR input.')
    dump(out/'models.json', report); dump(out/'state.json', dict(status='COMPLETE', Test_read=False))
    print(json.dumps(dict(status=report['status'], rows=len(y), models=models)), flush=True)


if __name__ == '__main__':
    main()
