#!/usr/bin/env python3
"""Registered small Train-only probe, serialized before GT diagnosis."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from tools.build_correlation_cache import auxiliary_input, read_pair
from tools.extract_correlation_policies import detections
from tools.photometric_compatibility import photometric_views
from tools.audit_correlation_readout import support
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.exp import get_exp
from yolox.tracker.foveated_regions import exact_half_preproc
from yolox.utils import postprocess


def cosine_maps(query, key):
    query = F.normalize(query.float(), dim=1, eps=1e-6)
    key = F.pad(F.normalize(key.float(), dim=1, eps=1e-6), (2, 2, 2, 2))
    h, w = query.shape[-2:]
    valid = F.pad(torch.ones_like(query[:, 0]), (2, 2, 2, 2))
    best = torch.full_like(query[:, 0], -2.)
    for dy in range(5):
        for dx in range(5):
            value = (query * key[:, :, dy:dy+h, dx:dx+w]).sum(1)
            value = value.masked_fill(valid[:, dy:dy+h, dx:dx+w] == 0, -2.)
            best = torch.maximum(best, value)
    return best.cpu().numpy()


def overlap(boxes, targets):
    boxes, targets = np.asarray(boxes), np.asarray(targets)
    if not len(boxes) or not len(targets):
        return np.zeros((len(boxes), len(targets)))
    top = np.maximum(boxes[:, None, :2], targets[None, :, :2])
    bottom = np.minimum(boxes[:, None, 2:4], targets[None, :, 2:4])
    intersection = np.maximum(bottom-top, 0).prod(-1)
    return intersection / np.maximum((boxes[:, 2:4]-boxes[:, :2]).prod(-1)[:, None] +
                                     (targets[:, 2:4]-targets[:, :2]).prod(-1)[None] - intersection, 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-staged', action='store_true')
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical != '3':
        raise ValueError('This queued probe is assigned only to physical GPU3')
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    parent_path = ROOT/'configs/perceptual_correlation_repair_20261123.json'
    spec_path = ROOT/'configs/perceptual_photometric_compatibility_20261123.json'
    plan, parent, spec = [json.loads(p.read_text()) for p in [plan_path, parent_path, spec_path]]
    assert sha256(plan_path) == spec['main_sha256'] and sha256(parent_path) == spec['parent_sha256']
    work = Path(plan['work_root']); out = work/'photometric_compatibility'
    out.mkdir(exist_ok=False)
    state_path = out/'state.json'
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/'tools/photometric_compatibility.py',
               ROOT/'tools/build_correlation_cache.py', ROOT/'tools/audit_correlation_readout.py',
               ROOT/'yolox/tracker/foveated_regions.py']}
    state = dict(status='WAITING_FOR_STAGED_REMOVAL_PRODUCERS', specification_sha256=sha256(spec_path),
                 sources=sources, physical_gpu=physical, Test_read=False)
    dump(state_path, state)
    if args.wait_staged:
        while True:
            staged = json.loads((work/'correlation_recoverability_staged_dev_state.json').read_text())
            if staged['status'] == 'FAILED':
                raise RuntimeError('Staged dependency failed')
            if staged['status'] == 'COMPLETE':
                break
            time.sleep(5)
    assert all(sha256(ROOT/p) == digest for p, digest in sources.items())
    assert sha256(spec_path) == state['specification_sha256']
    state['status'] = 'PRODUCING_ALL24_TRAIN_PREDICTIONS_NO_GT'
    dump(state_path, state)
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.35, 0)
    checkpoint = Path(parent['source_RGB_checkpoint'])
    assert sha256(checkpoint) == parent['source_RGB_sha256']
    model = get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'), None).get_model()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True); del saved
    model = model.cuda().half().eval().requires_grad_(False)
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    calibration = json.loads((work/'correlation_cache/calibration.json').read_text())['statistics']
    packets = {}; metadata = {}; permutation = [2, 3, 0, 1]
    with torch.inference_mode():
        for sequence in protocol['train']:
            cache = json.loads((work/'correlation_cache'/sequence/'features/complete.json').read_text())
            assert cache['config_sha256'] == sha256(parent_path) and cache['Dev_rows'] == cache['Test_rows'] == 0
            frame_ids = [cache['frame_ids'][i] for i in spec['Train_cached_indices']]
            pairs = [read_pair(plan, sequence, frame) for frame in frame_ids]
            prepared = [exact_half_preproc(rgb, spec['input_hw'], (.485, .456, .406), (.229, .224, .225)) for rgb, ir in pairs]
            features = {'rgb': model.backbone(torch.from_numpy(np.stack([x[0] for x in prepared])).cuda())}
            teacher = model.head(features['rgb'])
            rgb_boxes = [detections(p, item[1]) for p, item in zip(postprocess(teacher.clone(), 1, .1, .7), prepared)]
            view_inputs = {name: [] for name in spec['views'] if name != 'rgb'}
            diagnostics = []
            for (rgb, ir), boxes in zip(pairs, rgb_boxes):
                views, diag = photometric_views(rgb, ir, boxes[:, :4], boxes[:, 4]); diagnostics.append(diag)
                for name, image in views.items():
                    norm = calibration['ir' if name == 'ir_raw' else 'gray']
                    value, _ = auxiliary_input(image, spec['input_hw'], norm['mean'], norm['std'])
                    view_inputs[name].append(value)
            outputs = {'rgb': rgb_boxes}
            for name, values in view_inputs.items():
                features[name] = model.backbone(torch.from_numpy(np.stack(values)).cuda())
                outputs[name] = [detections(p, item[1]) for p, item in zip(postprocess(model.head(features[name]), 1, .1, .7), prepared)]
            arrays = {}
            for name in spec['views']:
                values = outputs[name]
                arrays[name+'_boxes'] = np.concatenate(values)
                arrays[name+'_offsets'] = np.r_[0, np.cumsum([len(x) for x in values])]
                for level, (q, key) in enumerate(zip(features['rgb'], features[name])):
                    arrays[f'{name}_P{level+3}_current'] = cosine_maps(q, key)
                    arrays[f'{name}_P{level+3}_wrong'] = cosine_maps(q, key[permutation])
            path = out/(sequence+'.npz'); np.savez_compressed(path, **arrays)
            packets[sequence] = dict(path=str(path), sha256=sha256(path), frame_ids=frame_ids,
                                     cache_manifest_sha256=sha256(work/'correlation_cache'/sequence/'features/complete.json'))
            metadata[sequence] = diagnostics
            print(json.dumps(dict(sequence=sequence, frames=len(frame_ids), predictions_frozen=True)), flush=True)
            del features, arrays
    frozen = dict(status='ALL24_TRAIN_PREDICTIONS_FROZEN_BEFORE_GT', packets=packets, metadata=metadata,
                  specification_sha256=sha256(spec_path), source_RGB_sha256=sha256(checkpoint), sources=sources, GT_read=False)
    dump(out/'predictions_frozen.json', frozen)
    rows = []; cosine_rows = []
    from scipy.optimize import linear_sum_assignment
    for sequence, packet in packets.items():
        assert sha256(packet['path']) == packet['sha256']
        gt_path = Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        assert sha256(gt_path) == protocol['derived_gt_sha256']['train/'+sequence]
        gt = np.loadtxt(gt_path, delimiter=',', ndmin=2)
        gt = gt[(gt[:, 6] > 0) & (gt[:, 7] == 1)]
        target_cache = np.load(work/'correlation_cache'/sequence/'features/targets.npy', mmap_mode='r')
        targets = target_cache[spec['Train_cached_indices']]
        with np.load(packet['path']) as arrays:
            for name in spec['views']:
                for threshold in [.1, .6, .7]:
                    tp = fp = total = 0
                    for i, frame in enumerate(packet['frame_ids']):
                        labels = gt[gt[:, 0] == frame, 2:6].copy(); labels[:, 2:] += labels[:, :2]
                        lo, hi = arrays[name+'_offsets'][i:i+2]
                        boxes = arrays[name+'_boxes'][lo:hi]; boxes = boxes[boxes[:, 4] >= threshold]
                        iou = overlap(boxes[:, :4], labels)
                        a, b = linear_sum_assignment(-(iou >= .5).astype(float)-iou*1e-3)
                        hits = int((iou[a, b] >= .5).sum())
                        tp += hits; fp += len(boxes)-hits; total += len(labels)
                    rows.append(dict(sequence=sequence, view=name, threshold=threshold, TP=tp, FP=fp, GT=total,
                                     recall=tp/max(total, 1), precision=tp/max(tp+fp, 1)))
                numerator = 0.; cells = 0
                for level in [3, 4, 5]:
                    current, wrong = arrays[f'{name}_P{level}_current'], arrays[f'{name}_P{level}_wrong']
                    for i in range(4):
                        h, width = current[i].shape
                        mask = support(targets[i], h, width, spec['input_hw']) & ~support(targets[permutation[i]], h, width, spec['input_hw'])
                        numerator += float((current[i]-wrong[i])[mask].sum()); cells += int(mask.sum())
                cosine_rows.append(dict(sequence=sequence, view=name, cells=cells, current_minus_wrong=numerator/max(cells, 1)))
    recall_wins = []; cosine_wins = []
    for sequence in protocol['train']:
        rr = {r['view']: r['recall'] for r in rows if r['sequence'] == sequence and r['threshold'] == .6}
        cc = {r['view']: r['current_minus_wrong'] for r in cosine_rows if r['sequence'] == sequence}
        recall_wins.append(rr['ir_polarity_cdf'] > rr['ir_raw'])
        cosine_wins.append(cc['ir_polarity_cdf'] > cc['ir_raw'])
    report = dict(status='COMPLETE_SMALL_TRAIN_COMPATIBILITY_DIAGNOSTIC', rows=rows, cosine_rows=cosine_rows,
                  recall_video_wins=sum(recall_wins), cosine_video_wins=sum(cosine_wins),
                  prerequisite_for_further_registration=sum(recall_wins) >= 4 and sum(cosine_wins) >= 4,
                  predictions_frozen_sha256=sha256(out/'predictions_frozen.json'), source=state,
                  scope='Only24 Train images, not heldout accuracy or tracking HOTA. Unprojected frozen-FPN local-cosine diagnostic. Precision is raw class1 foreground precision without class8 ignore suppression. No new fitting/cache/Test authorized by this report.',
                  Test_read=False, goal_met=False, supported_innovations=0)
    dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_PHOTOMETRIC_TRAIN_AUDIT.json', report)
    state.update(status='COMPLETE', prerequisite=report['prerequisite_for_further_registration'])
    dump(state_path, state)
    print(json.dumps({k: report[k] for k in ['status', 'recall_video_wins', 'cosine_video_wins', 'prerequisite_for_further_registration']}))


if __name__ == '__main__':
    main()
