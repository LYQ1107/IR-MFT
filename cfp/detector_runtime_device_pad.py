"""Identical detector input: current tensor transfers once, pads on device.

No weight/precision/batch-size/IR/output change. The model still executes the same
contiguous eight-frame FP16 tensor. Only repeated CPU allocation/H2D traffic falls.
"""
from pathlib import Path
import os
import socket
import subprocess
import time

import numpy as np
import torch

from .detector_runtime import DetectorRuntime, receive_arrays
from .online import CurrentFrameFrontend
from yolox.utils import postprocess


def current_device_padding(image, batch, device='cuda'):
    if image.dtype != np.float32 or image.ndim != 3 or batch not in (1, 8):
        raise ValueError('Same float32 CHW current image and declared1/8batch required')
    return torch.from_numpy(image)[None].to(device).half().repeat(batch, 1, 1, 1)


class DevicePadDetectorRuntime(DetectorRuntime):
    def __init__(self, python, checkpoint, scripts_root, batch, preprocessing):
        self.channel, peer = socket.socketpair()
        self.channel.settimeout(120)
        command = [str(python), '-u', 'tools/cfp_detector_worker_device_pad.py', '--fd', str(peer.fileno()),
            '--checkpoint', str(checkpoint), '--scripts-root', str(scripts_root), '--batch', str(batch), '--preprocessing', preprocessing]
        self.process_handle = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, PYTHONPATH='.'), pass_fds=(peer.fileno(),))
        peer.close()
        _, self.metadata = receive_arrays(self.channel)
        expected = dict(torch='2.0.0+cu117', cuda='11.7', cudnn=8500)
        if any(self.metadata[k] != value for k, value in expected.items()):
            self.close()
            raise ValueError('Device-padding worker runtime mismatch')


class DevicePadDetectorFrontend(CurrentFrameFrontend):
    """Worker-only detector; no identity encoder or IR generation shortcuts."""
    def process(self, rgb, provided_ir=None, **kwargs):
        if self.input_regime != 'causal_precomputed' or self.detector_worker is not None or self.rgb_encoder is not None:
            raise ValueError('This worker only scores provided current pairs')
        if rgb.dtype != np.uint8 or provided_ir is None or provided_ir.shape != rgb.shape:
            raise ValueError('Current aligned uint8 pair required')
        start = time.perf_counter()
        image, _ = self.transform(rgb, provided_ir, None, self.exp.test_size)
        tensor = current_device_padding(image, self.detector_batch)
        torch.cuda.synchronize()
        timings = dict(provided_ir_validation=0., detector_preprocess_transfer=time.perf_counter() - start)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = postprocess(self.detector(tensor), 1, .1, .7)
        torch.cuda.synchronize()
        timings['detector_nms'] = time.perf_counter() - start
        start = time.perf_counter()
        raw = np.empty((0, 7), np.float32) if outputs[0] is None else outputs[0].float().cpu().numpy()
        scale = min(self.exp.test_size[0] / rgb.shape[0], self.exp.test_size[1] / rgb.shape[1])
        boxes, scores = (raw[:, :4] / scale).astype(np.float32), (raw[:, 4] * raw[:, 5]).astype(np.float32)
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError('Nonfinite detector output')
        timings['proposal_transfer'] = time.perf_counter() - start
        return dict(boxes=boxes, scores=scores, raw_post_nms=raw, stage_seconds=timings)

