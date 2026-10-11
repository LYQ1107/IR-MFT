"""Causal adapters for the recovered, content-pinned RGB response generators.

Original scripts stay outside the repository and their guarded mains never run.
Every stream owns its state; no full-video sampling or strategy selection occurs.
"""
import contextlib
import importlib.util
import io
from pathlib import Path

import cv2
import numpy as np

from .provenance import sha256


SOURCES = {
    'local': ('trans_tir_final.py', '31289e713bf13d34530e0310dff038c18eb150c9c5a310e9f3341fa5f6662437'),
    'unified_v2': ('trans-tir0611.py', '74da8fce8619f3eaafb3aa451d6e123ccb6790b692e7aeb1f70b4fb834826800'),
    'temporal_ema': ('trans-sn0612.py', '03d578453f7cc37f08b9ea44102f0fc74891bcee132dc7d3f64ad3adbc41913d'),
}


def load_generator(scripts_root, mode):
    filename, expected = SOURCES[mode]
    path = Path(scripts_root) / filename
    if sha256(path) != expected:
        raise ValueError(f'Generator source differs from pinned original: {path}')
    spec = importlib.util.spec_from_file_location('cfp_external_' + mode, path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


class CausalPseudoIR:
    def __init__(self, scripts_root, mode='temporal_ema', alpha=.01, optimize=True):
        if mode not in SOURCES or not 0 < alpha <= 1:
            raise ValueError('Invalid fixed causal generator configuration')
        self.mode, self.alpha = mode, alpha
        self.source = load_generator(scripts_root, mode)
        self.background = None
        self.frames_seen = 0
        self.border_masks = {}
        if optimize and mode == 'temporal_ema':
            # Replace only two pure functions in this private imported module.
            # No original file, shared cv2 function, or response parameter changes.
            self.source.suppress_border_only = self.suppress_border
            self.source.filter_components = self.filter_components

    def advance_background(self, frame):
        """Advance prefix state without a response, for offline chunk preparation."""
        if self.mode != 'temporal_ema':
            raise ValueError('Only causal EMA has a background state')
        if self.background is None:
            self.background = frame.astype(np.float32)
        if self.background.shape != frame.shape:
            raise ValueError('Changing image dimensions within a stream')
        cv2.accumulateWeighted(frame, self.background, self.alpha)

    def suppress_border(self, gray):
        h, w = gray.shape
        source = self.source
        if (h, w) not in self.border_masks:
            mask = np.ones((h, w), np.float32)
            mask[:, :int(w * source.left_border_ratio)] *= source.border_keep
            mask[:, w - int(w * source.right_border_ratio):] *= source.border_keep
            if source.use_ellipse_suppression:
                yy, xx = np.mgrid[0:h, 0:w]
                ellipse = ((xx - w / 2.) / (w * .465)) ** 2 + ((yy - h / 2.) / (h * .485)) ** 2
                ellipse_mask = np.ones((h, w), np.float32)
                ellipse_mask[ellipse >= source.ellipse_outer] = source.ellipse_outside_keep
                ring = (ellipse >= source.ellipse_inner) & (ellipse < source.ellipse_outer)
                if np.any(ring):
                    t = (ellipse[ring] - source.ellipse_inner) / (source.ellipse_outer - source.ellipse_inner)
                    ellipse_mask[ring] = 1. - t * (1. - source.ellipse_outside_keep)
                mask *= cv2.GaussianBlur(ellipse_mask, (51, 51), 0)
            self.border_masks[(h, w)] = np.clip(cv2.GaussianBlur(mask, (21, 21), 0), 0., 1.)
        return np.clip(gray.astype(np.float32) * self.border_masks[(h, w)], 0, 255).astype(np.uint8)

    def filter_components(self, mask):
        h, w = mask.shape
        source = self.source
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        accepted = np.zeros(count, dtype=np.uint8)
        for i in range(1, count):
            _, _, bw, bh, area = stats[i]
            if area < max(10, int(h * w * source.min_area_ratio)) or area > int(h * w * source.max_area_ratio):
                continue
            aspect = bw / float(bh + 1e-6)
            if aspect > 10. or aspect < .10 or bw > .65 * w or bh > .65 * h:
                continue
            accepted[i] = 255
        # One lookup replaces one full-image comparison per accepted component.
        keep = accepted[labels]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.GaussianBlur(cv2.dilate(keep, kernel, iterations=1), (17, 17), 0)

    def __call__(self, frame):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected uint8 BGR image')
        if self.mode == 'temporal_ema':
            if self.background is None:
                # First frame uses its own image as background; there is no warmup
                # lookahead, future-dependent initialization or discarded prefix.
                self.background = frame.astype(np.float32)
            if self.background.shape != frame.shape:
                raise ValueError('Changing image dimensions within a stream')
            background = np.clip(self.background, 0, 255).astype(np.uint8)
            gray = self.source.rgb_to_pseudo_gray_temporal(frame, background)
            # The response is computed BEFORE writing the current observation.
            cv2.accumulateWeighted(frame, self.background, self.alpha)
        else:
            gray = self.source.rgb_to_pseudo_gray(frame)
            gray = self.source.udwt_bayes_denoise_gray(gray)
            gray = self.source.suppress_border_only(gray)
            if self.mode == 'local':
                gray = self.source.percentile_enhance(gray)
            gray[gray < self.source.background_threshold] = 0
            if self.mode == 'unified_v2':
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
        self.frames_seen += 1
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
