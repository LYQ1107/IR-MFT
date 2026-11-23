#!/usr/bin/env python3
"""Train-only ordinary objectness probes, complete held-out Dev diagnostics."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.matching import ious


def read(path, key):
    with np.load(path) as z:
        return z[key].astype(np.float32)


def main():
    main = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    config_path = ROOT/'configs/perceptual_objectness_audit_20261123.json'
    config = json.loads(config_path.read_text())
    protocol = json.loads((ROOT/main['protocol']).read_text())
    work = Path(main['work_root'])
    out = work/'native_objectness_audit'
    out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        raise ValueError('Objectness diagnostic already complete')
    sequences = protocol['train']
    labels, weights, reference, sources = [], [], [], {}
    for sequence in sequences:
        index = work/'native_candidate_index'/sequence/'index.npz'
        y = (read(index, 'label') >= 0).astype(np.int64)
        if set(np.unique(y)) != {0,1}:
            raise ValueError('Both physical Train classes required')
        sample_weight = np.empty(len(y), np.float64)
        for value in [0,1]:
            sample_weight[y == value] = 1/(2*len(sequences)*(y == value).sum())
        feature = work/'native_candidate_features'/sequence/'rgb/features.npz'
        labels.append(y)
        weights.append(sample_weight)
        reference.append(read(feature, 'reference_post'))
        sources[sequence] = dict(index_sha256=sha256(index), reference_sha256=sha256(feature))
    y = np.concatenate(labels)
    sample_weight = np.concatenate(weights)*len(y)
    reference = np.concatenate(reference)
    # Finish and serialize every Train fit before reading any Dev annotation.
    models = {}
    for cue, seed in [('reference',0)] + [(cue, seed) for cue in ['rgb','ir'] for seed in range(3)]:
        started = time.perf_counter()
        if cue == 'reference':
            x = reference
        else:
            values = []
            for sequence in sequences:
                path = work/'native_candidate_features'/sequence/cue/'features.npz'
                values.append(read(path, f'native_s{seed}'))
                sources[sequence][cue+'_sha256'] = sha256(path)
            x = np.concatenate([reference, np.concatenate(values)], axis=1)
        mean = np.average(x, axis=0, weights=sample_weight).astype(np.float32)
        std = np.maximum(np.sqrt(np.average((x-mean)**2, axis=0, weights=sample_weight)), .001).astype(np.float32)
        x = (x-mean)/std
        model = LogisticRegression(C=config['C'], max_iter=config['max_iter'], tol=config['tolerance'],
                                   solver='lbfgs', random_state=0)
        model.fit(x, y, sample_weight=sample_weight)
        if int(model.n_iter_[0]) >= config['max_iter']:
            raise RuntimeError('Probe did not converge; do not report an unfair fit')
        key = f'{cue}_s{seed}'
        target = out/(key+'.npz')
        np.savez_compressed(target, mean=mean, std=std, coefficient=model.coef_.ravel(), intercept=model.intercept_)
        models[key] = dict(cue=cue, seed=seed, path=str(target), sha256=sha256(target), iterations=int(model.n_iter_[0]),
                           fit_seconds=time.perf_counter()-started)
        print(json.dumps(dict(Train_model=key, rows=len(y), iterations=int(model.n_iter_[0]), seconds=models[key]['fit_seconds'])), flush=True)
        del x
    dump(out/'train_complete.json', dict(status='ALL_TRAIN_FITS_COMPLETE_BEFORE_DEV_ANNOTATIONS', rows=len(y),
        class_counts=np.bincount(y).tolist(), sources=sources, models=models, config_sha256=sha256(config_path),
        Dev_fit_rows=0, Test_read=False))
    rows = []
    for sequence in protocol['dev']:
        gt_path = Path(main['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        gt = np.loadtxt(gt_path, delimiter=',', ndmin=2)
        for front in ['new','rgb']:
            reference_path = work/'native_eval_features/dev'/sequence/'rgb'/(front+'.npz')
            with np.load(reference_path) as z:
                boxes, offsets, frames = z['boxes'], z['offsets'], z['frame_ids']
                ref = z['reference_post']
                rgb = {f'native_s{seed}':z[f'native_s{seed}'] for seed in range(3)}
            labels = np.full(len(boxes), -1, np.int8)
            for position, frame in enumerate(frames):
                lo,hi = offsets[position:position+2]
                truth = gt[gt[:,0] == frame]
                positive = truth[(truth[:,6] > 0)&(truth[:,7] == 1), 2:6].copy()
                ignored = truth[truth[:,7] == 8, 2:6].copy()
                for values in [positive,ignored]:
                    values[:,2:] += values[:,:2]
                maximum = ious(boxes[lo:hi], positive).max(1) if len(positive) else np.zeros(hi-lo)
                overlap_ignore = ious(boxes[lo:hi], ignored).max(1) if len(ignored) else np.zeros(hi-lo)
                labels[lo:hi] = np.where(maximum >= .5, 1, np.where((maximum <= .2)&(overlap_ignore <= .2),0,-1))
            selected = labels >= 0
            for key, info in models.items():
                cue,seed = info['cue'],info['seed']
                if cue == 'reference':
                    x = ref
                elif cue == 'rgb':
                    x = np.c_[ref,rgb[f'native_s{seed}']]
                else:
                    path = work/'native_eval_features/dev'/sequence/'ir'/(front+'.npz')
                    x = np.c_[ref,read(path, f'native_s{seed}')]
                with np.load(info['path']) as model:
                    logit = ((x-model['mean'])/model['std'])@model['coefficient'] + model['intercept'][0]
                probability = 1/(1+np.exp(-np.clip(logit,-60,60)))
                auc = roc_auc_score(labels[selected], logit[selected])
                ap = average_precision_score(labels[selected], logit[selected])
                rows.append(dict(sequence=sequence, frontend=front, model=key, cue=cue, seed=seed,
                    frames=len(frames), evaluated_rows=int(selected.sum()), positives=int((labels[selected] == 1).sum()),
                    negatives=int((labels[selected] == 0).sum()), AUC=float(auc), AP=float(ap),
                    foreground_mean=float(probability[labels == 1].mean()), background_mean=float(probability[labels == 0].mean())))
    report = dict(status='COMPLETE_TRAIN_FITTED_DEV_DIAGNOSTIC', config_sha256=sha256(config_path),
        Train_rows=len(y), Train_videos=sequences, Dev_fit_rows=0, Test_read=False, source_sha256=sha256(__file__),
        models=models, rows=rows, supported_multimodal_innovations=0, goal_met=False,
        scope='Ordinary foreground classification probes, not tracking HOTA. Dev GT is posthoc diagnostic only; all models completed before any Dev labels were read.')
    dump(out/'complete.json', report)
    dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_OBJECTNESS_DEV_AUDIT.json', report)
    print(json.dumps(dict(status=report['status'], rows=len(rows), Test_read=False)), flush=True)


if __name__ == '__main__':
    main()
