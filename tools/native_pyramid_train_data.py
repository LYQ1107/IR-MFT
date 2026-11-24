"""Ordered, complete Train observations for the native-pyramid study.

There is no frame/FPN disk cache. Clips only group optimizer inputs; temporal
state continues across adjacent clips and is reset at the video boundary.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.build_correlation_cache import auxiliary_input
from tools.run_pseudo_ir_comparison import sha256
from yolox.models.native_pyramid_fusion import NativeTemporalState
from yolox.tracker.foveated_regions import exact_half_preproc


@dataclass(frozen=True)
class TrainClip:
    sequence: str
    frames: tuple


def epoch_clips(lengths, epoch, clip_size=4, seed=0):
    """Shuffle videos, never their frames; a short tail is not duplicated."""
    if seed != 0 or not isinstance(epoch, int) or epoch < 1 or clip_size < 1:
        raise ValueError('Positive epoch/clip size and fixed seed0 required')
    if not lengths or any(not isinstance(n, int) or n < 1 for n in lengths.values()):
        raise ValueError('Positive integer Train video lengths required')
    sequences = sorted(lengths)
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(sequences))
    for index in order:
        sequence = sequences[int(index)]
        for start in range(1, lengths[sequence] + 1, clip_size):
            yield TrainClip(sequence, tuple(range(start, min(start + clip_size, lengths[sequence] + 1))))


def address_bytes(clip):
    return json.dumps([clip.sequence, list(clip.frames)], separators=(',', ':')).encode() + b'\n'


def schedule_proof(lengths, epoch, clip_size=4):
    digest, count, order = hashlib.sha256(), 0, []
    for clip in epoch_clips(lengths, epoch, clip_size):
        digest.update(address_bytes(clip))
        count += 1
        if not order or order[-1] != clip.sequence:
            order.append(clip.sequence)
    return dict(epoch=epoch, video_order=order, clips=count, frames=sum(lengths.values()),
                repeated_frames=0, address_sha256=digest.hexdigest(),
                per_video_frames=dict(lengths))


def detach_state(state, device=None):
    """Classical online memory has no gradient through past observations."""
    if state is None:
        return None
    def values(rows):
        return tuple(v.detach().to(device=device) if device is not None else v.detach() for v in rows)
    return NativeTemporalState(state.sequence, state.frame, values(state.latent),
                               values(state.covariance), values(state.valid))


@dataclass
class PixelClip:
    address: TrainClip
    rgb: torch.Tensor
    cue: torch.Tensor
    targets: torch.Tensor
    valid: torch.Tensor
    available: tuple
    flow_gray: tuple


def preprocess_observation(rgb, cue_image, cue, input_hw, calibration):
    """Same fixed half-precision input lookup as the source detector."""
    if cue not in ['rgb', 'gray', 'ir'] or rgb.dtype != np.uint8 or rgb.ndim != 3:
        raise ValueError('Registered cue and decoded uint8 RGB required')
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    value, ratio = exact_half_preproc(rgb, input_hw, (.485, .456, .406), (.229, .224, .225))
    h, w = int(rgb.shape[0] * ratio), int(rgb.shape[1] * ratio)
    valid = np.zeros((1, *input_hw), bool)
    valid[:, :h, :w] = True
    padded = np.zeros(input_hw, np.uint8)
    padded[:h, :w] = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
    flow_gray = cv2.resize(padded, (input_hw[1] // 4, input_hw[0] // 4), interpolation=cv2.INTER_AREA)
    if cue == 'rgb':
        auxiliary, available = value.astype(np.float32), True
    else:
        image = gray if cue == 'gray' else cue_image
        if image is None or image.dtype != np.uint8 or image.ndim != 2 or image.shape != rgb.shape[:2]:
            raise ValueError('Aligned scalar uint8 actual cue image required')
        norm = calibration['statistics'][cue]
        auxiliary, _ = auxiliary_input(image, input_hw, norm['mean'], norm['std'])
        auxiliary = auxiliary.astype(np.float32)
        available = bool(image.max() != image.min())
    return value, auxiliary, valid, flow_gray, ratio, available


def current_to_previous_flow(current, previous):
    """Both arrays contain only current/past RGB, at quarter input resolution."""
    if previous is None:
        return None
    if current.shape != previous.shape or current.dtype != np.uint8 or previous.dtype != np.uint8:
        raise ValueError('Same-size uint8 current and previous RGB grayscale required')
    flow = cv2.calcOpticalFlowFarneback(current, previous, None, .5, 3, 15, 3, 5, 1.2, 0)
    return torch.from_numpy(flow.transpose(2, 0, 1)[None].copy())


class OrderedTrainPixels:
    def __init__(self, plan, protocol, lengths, calibration, image_reader, input_hw, device):
        if set(lengths) != set(protocol['train']) or image_reader.sequences != frozenset(protocol['train']):
            raise ValueError('Only the fixed six Train videos are allowed')
        if calibration['Dev_rows'] != 0 or calibration['Test_rows'] != 0:
            raise ValueError('Train-only calibration required')
        self.lengths, self.reader = dict(lengths), image_reader
        self.calibration, self.input_hw, self.device = calibration, tuple(input_hw), torch.device(device)
        self.gt_hashes, self.targets, self.expected_next = {}, {}, None
        self.epoch, self.prepared, self.digest, self.used = None, 0, None, None
        for sequence in sorted(lengths):
            path = Path(plan['evaluation']['GT_root']) / 'train' / sequence / 'gt/gt.txt'
            digest = sha256(path)
            if digest != protocol['derived_gt_sha256']['train/' + sequence]:
                raise ValueError('Fixed derived Train GT changed')
            gt = np.loadtxt(path, delimiter=',', ndmin=2)
            gt = gt[(gt[:, 6] > 0) & (gt[:, 7] == 1)]
            self.targets[sequence], self.gt_hashes[sequence] = gt, digest

    def begin_epoch(self, epoch):
        if self.epoch is not None:
            raise ValueError('Previous epoch must be completely verified first')
        self.epoch, self.prepared = epoch, 0
        self.digest = hashlib.sha256()
        self.used = {s: [] for s in self.lengths}
        self.expected_next = iter(epoch_clips(self.lengths, epoch))

    def prepare(self, address, cue):
        if cue not in ['rgb', 'gray', 'ir'] or self.epoch is None:
            raise ValueError('Begin an epoch with a registered cue first')
        expected = next(self.expected_next, None)
        if address != expected:
            raise ValueError('Train clip is out of the registered chronological schedule')
        rgb_rows, cue_rows, targets, valid, available, gray_rows = [], [], [], [], [], []
        for frame in address.frames:
            rgb = self.reader.read(address.sequence, frame, 'rgb')
            ir = self.reader.read(address.sequence, frame, 'ir') if cue == 'ir' else None
            r, a, mask, gray, ratio, present = preprocess_observation(
                rgb, ir, cue, self.input_hw, self.calibration)
            target = np.zeros((500, 5), np.float32)
            gt = self.targets[address.sequence]
            boxes = gt[gt[:, 0] == frame, 2:6].astype(np.float32).copy()
            boxes[:, :2] += boxes[:, 2:] / 2
            boxes *= ratio
            if len(boxes) > 500:
                raise ValueError('Original maximum target count exceeded')
            target[:len(boxes), 1:] = boxes
            rgb_rows.append(r); cue_rows.append(a); targets.append(target)
            valid.append(mask); available.append(present); gray_rows.append(gray)
        def tensor(rows):
            return torch.from_numpy(np.stack(rows)).to(self.device)
        self.digest.update(address_bytes(address))
        self.used[address.sequence].extend(address.frames)
        self.prepared += 1
        return PixelClip(address, tensor(rgb_rows), tensor(cue_rows), tensor(targets),
                         tensor(valid), tuple(available), tuple(gray_rows))

    def finish_epoch(self):
        if self.epoch is None or next(self.expected_next, None) is not None:
            raise ValueError('The complete Train traversal has not been delivered')
        expected = schedule_proof(self.lengths, self.epoch)
        if (self.digest.hexdigest() != expected['address_sha256'] or self.prepared != expected['clips'] or
                any(self.used[s] != list(range(1, n + 1)) for s, n in self.lengths.items())):
            raise ValueError('Actual full-Train delivery differs from the registered schedule')
        result = dict(expected, actual_address_sha256=self.digest.hexdigest(),
                      delivered_clips=self.prepared, complete_all_Train_frames=True)
        self.epoch = self.expected_next = None
        return result
