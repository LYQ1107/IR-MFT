#!/usr/bin/env python3
"""Isolated cu117 V3 cache-tail batch shape probe, not whole online profiling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import cv2
import numpy as np
import torch
from cfp.detector_runtime_device_pad import DevicePadDetectorFrontend
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.utils import postprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if a.output.exists() or physical not in ['0', '1', '2', '3']:
        raise ValueError('New probe and exactly one approved idle GPU required')
    idle = subprocess.check_output(['nvidia-smi', '-i', physical, '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    memory, utilization = [int(x.strip()) for x in idle.split(',')]
    if memory > 64 or utilization:
        raise ValueError('Never interfere with an active GPU')
    if torch.__version__ != '2.0.0+cu117' or torch.version.cuda != '11.7' or torch.backends.cudnn.version() != 8500:
        raise ValueError('Exact original cu117 detector runtime required')
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.manual_seed(0); torch.cuda.set_per_process_memory_fraction(.6)
    config = json.loads(Path('configs/cfp/phase_vii_whole_raw_current_profile_v1.json').read_text())
    detections = Path(config['detections']); dm = json.loads((detections/'manifest.json').read_text())
    total = sum(v['frames'] for v in dm['caches'].values()); tail = total % dm['config']['batch_size']
    if tail != 3 or dm['config']['batch_size'] != 8:
        raise ValueError('Inspected original global unpadded3-tail contract changed')
    paths = json.loads((a.root/'identity_two_crop_index_v1/SN-015.frames.json').read_text())
    with np.load(detections/'SN-015.npz', allow_pickle=False) as bank:
        frames, offsets, boxes, scores = [bank[k].copy() for k in ['frame_ids', 'offsets', 'boxes', 'scores']]
    frontend = DevicePadDetectorFrontend(config['detector_checkpoint'], config['scripts_root'], detector_batch=8,
                                         input_regime='causal_precomputed', preprocessing='lookup_uint8')
    results = []
    with torch.inference_mode():
        for frame in range(int(frames[-1])-tail+1, int(frames[-1])+1):
            row = paths[frame-1]
            rgb = cv2.imread(str(Path(config['data_root'])/'train'/row['rgb_file']))
            ir = cv2.imread(str(Path(config['data_root'])/'train'/row['ir_file']))
            current, _ = frontend.transform(rgb, ir, None, frontend.exp.test_size)
            tensor = torch.from_numpy(current)[None].cuda().half(); position = frame-(int(frames[-1])-tail+1)
            outputs = {}
            for kind in ['repeat8_current', 'repeat3_current', 'zeros3_current_at_original_position']:
                batch = tensor.repeat(8 if kind=='repeat8_current' else 3, 1, 1, 1)
                index = 0
                if kind == 'zeros3_current_at_original_position':
                    batch.zero_(); batch[position] = tensor[0]; index = position
                predicted = postprocess(frontend.detector(batch), 1, .1, .7)[index]
                raw = np.empty((0, 7), np.float32) if predicted is None else predicted.float().cpu().numpy()
                scale = min(frontend.exp.test_size[0]/rgb.shape[0], frontend.exp.test_size[1]/rgb.shape[1])
                actual_boxes = (raw[:, :4]/scale).astype(np.float32); actual_scores = (raw[:, 4]*raw[:, 5]).astype(np.float32)
                lo, hi = offsets[frame-1:frame+1]
                same_shape = actual_boxes.shape == boxes[lo:hi].shape
                outputs[kind] = dict(rows=len(raw), reference_rows=int(hi-lo),
                    proposal_byte_equal=np.array_equal(actual_boxes, boxes[lo:hi]) and np.array_equal(actual_scores, scores[lo:hi]),
                    maximum_box_difference=float(np.max(np.abs(actual_boxes-boxes[lo:hi]))) if same_shape and len(raw) else None,
                    maximum_score_difference=float(np.max(np.abs(actual_scores-scores[lo:hi]))) if same_shape and len(raw) else None,
                    boxes_sha256=__import__('hashlib').sha256(actual_boxes.tobytes()).hexdigest(), scores_sha256=__import__('hashlib').sha256(actual_scores.tobytes()).hexdigest())
                np.savez_compressed(a.output.with_name(a.output.stem+f'_{frame}_{kind}.npz'), boxes=actual_boxes, scores=actual_scores, raw_post_nms=raw)
            results.append(dict(frame=frame, original_tail_position=position, results=outputs))
    verified = all(v['results']['zeros3_current_at_original_position']['proposal_byte_equal'] for v in results)
    write_json(a.output, dict(status='VERIFIED_CURRENT_ONLY_DETECTOR_CACHE_TAIL_BATCH_SHAPE' if verified else 'DETECTOR_CACHE_TAIL_NOT_EXPLAINED',
        global_selected_frames=total, original_batch_size=8, final_cache_batch_size=tail, checkpoint_sha256=sha256(config['detector_checkpoint']),
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), results=results,
        sources=snapshot_sources(a.output.with_suffix('.sources'), [__file__, 'tools/cfp_cache_detections.py', 'cfp/detector_runtime_device_pad.py']),
        scope='Three current-frame-only cu117 V3 inputs, cached currentIR used only for this isolated numerical diagnosis. No future images in any probe; zeros at all other originaltail positions. Not whole raw generation/timing, no tracker update, no cached/tail-validation outputs fed into inference. Normal declared online detector remains padded8. Originalcache finalbatch3 numerical difference must be reported, not concealed or forced into onlinebatch3.'))
    print(json.dumps(dict(verified=verified, results=results)))
    if not verified:
        raise RuntimeError('Actual tail batch probe did not explain cache discrepancy')


if __name__=='__main__':
    main()
