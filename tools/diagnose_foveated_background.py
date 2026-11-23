#!/usr/bin/env python3
"""Unlabeled current-image audit of causal background-conditioned attention."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.foveated_regions import coarse_attention, working_region, attention_windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sequence', choices=['BT-005', 'SN-013'], required=True)
    p.add_argument('--frames', type=int, default=300)
    a = p.parse_args()
    cv2.setNumThreads(1)
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    source = Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(a.sequence+'.npz')
    with np.load(source) as z:
        boxes, scores, offsets = z['boxes'], z['scores'], z['offsets']
    backgrounds = {cue: cv2.createBackgroundSubtractorMOG2(history=256, varThreshold=16,
                                                         detectShadows=False) for cue in ['rgb', 'ir']}
    records, panels = [], []
    for frame in range(1, a.frames+1):
        rgb_path = Path(plan['data_root'])/'train'/a.sequence/'visible'/f'{frame:06d}.jpg'
        ir_path = Path(plan['generated_ir_root'])/'dev'/a.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
        rgb, ir = cv2.imread(str(rgb_path)), cv2.imread(str(ir_path), -1)
        lo, hi = offsets[frame-1:frame+1]
        b, s = boxes[lo:hi], scores[lo:hi]
        support = working_region(rgb, b, s)
        model_inputs = dict(rgb=cv2.cvtColor(cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA),
                                           cv2.COLOR_BGR2LAB),
                            ir=cv2.cvtColor(cv2.resize(ir, (320, 180), interpolation=cv2.INTER_AREA),
                                            cv2.COLOR_GRAY2BGR))
        current = []
        for cue, image in [('rgb', rgb), ('ir', ir)]:
            mask = backgrounds[cue].apply(model_inputs[cue], learningRate=-1)/255.
            mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.)
            raw = coarse_attention(image, b, s, rgb.shape, support=support)
            # Diagnosis only: current contrast gated by past-only scene novelty.
            heat = raw*mask
            if frame not in {150, a.frames}:
                continue
            windows = attention_windows(heat, rgb.shape, 4, fraction=.25)
            shown = rgb.copy() if cue == 'rgb' else cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
            for x1,y1,x2,y2 in windows:
                cv2.rectangle(shown, (x1,y1), (x2,y2), (0,255,255), 4)
            canvas = cv2.resize(shown, (640,360))
            cv2.putText(canvas, f'{a.sequence} f{frame} {cue} causal background', (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            current += [canvas, cv2.resize(cv2.applyColorMap((np.clip(heat/2,0,1)*255).astype(np.uint8),
                                                            cv2.COLORMAP_TURBO), (640,360))]
            records.append(dict(frame=frame, cue=cue, foreground_fraction=float(mask.mean()), windows=windows,
                                rgb_sha256=sha256(rgb_path), IR_sha256=sha256(ir_path)))
        if current:
            panels.append(np.concatenate(current, axis=1))
    out = work/'foveated_background_diagnosis'/a.sequence
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out/'audit.png'), np.concatenate(panels, axis=0))
    dump(out/'audit.json', dict(status='UNLABELED_DIAGNOSIS_ONLY', GT_read=False, Test_read=False,
                               past_frame_inputs=True, future_frame_inputs=False, rows=records))
    print(json.dumps(dict(sequence=a.sequence, frames=a.frames, GT_read=False, image=str(out/'audit.png'))))


if __name__ == '__main__':
    main()
