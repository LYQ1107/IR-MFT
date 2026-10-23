#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Continuous RGB/auxiliary clips for causal temporal fusion.

The ordinary MOT dataset is intentionally frame indexed and can be sampled in
arbitrary order.  That is correct for the detector baseline, but it cannot
train a recurrent fusion module without accidentally connecting unrelated
frames.  This wrapper groups an already validated :class:`MOTDatasetRGBT`
instance by sequence and returns fixed-length, time-ordered clips.  A clip
never crosses a sequence boundary; the caller must reset the temporal state
between clips.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .mot_rgbt import MOTDatasetRGBT


def _make_clip_starts(length: int, clip_len: int, stride: int) -> List[int]:
    """Return starts that cover a sequence with fixed-length clips.

    The final window is shifted to the end when the regular stride leaves a
    short uncovered tail.  This can overlap the preceding window, but keeps
    every returned sample length ``clip_len`` and ensures every frame is seen
    by at least one clip.  MFT26-IR sequences are all longer than four frames.
    """

    length = int(length)
    clip_len = int(clip_len)
    stride = int(stride)
    if length <= 0:
        return []
    if clip_len <= 0 or stride <= 0:
        raise ValueError("clip_len and stride must be positive")
    if length < clip_len:
        raise ValueError(
            f"sequence has {length} frames, shorter than clip_len={clip_len}"
        )
    last_start = length - clip_len
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


class MOTDatasetRGBTClip(Dataset):
    """A fixed-length, causal clip view over :class:`MOTDatasetRGBT`.

    Parameters other than ``clip_len`` and ``stride`` are forwarded to the
    frame dataset.  ``auxiliary_mode='causal_cache'`` is recommended for
    shuffled clip loading: the cache has already been verified against the
    online causal generator, while the raw ``causal`` mode depends on access
    order and therefore must use a sequential, single-worker loader.
    """

    def __init__(
        self,
        *args,
        clip_len: int = 4,
        stride: int | None = None,
        **kwargs,
    ):
        self.clip_len = int(clip_len)
        self.stride = self.clip_len if stride is None else int(stride)
        if self.clip_len <= 0 or self.stride <= 0:
            raise ValueError("clip_len and stride must be positive")
        self.frame_dataset = MOTDatasetRGBT(*args, **kwargs)

        grouped: "OrderedDict[str, List[Tuple[int, int]]]" = OrderedDict()
        for index, annotation in enumerate(self.frame_dataset.annotations):
            # MOTDatasetRGBT stores (target, img_info, visible_path,
            # auxiliary_path); img_info[2] is the integer frame id.
            sequence = annotation[2].split("/", 1)[0]
            frame_id = int(annotation[1][2])
            grouped.setdefault(sequence, []).append((frame_id, index))

        self.sequence_indices = OrderedDict()
        self.clips: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
        for sequence, entries in grouped.items():
            entries.sort(key=lambda item: item[0])
            frame_ids = [item[0] for item in entries]
            frame_indices = [item[1] for item in entries]
            self.sequence_indices[sequence] = tuple(frame_indices)
            for start in _make_clip_starts(
                len(frame_indices), self.clip_len, self.stride
            ):
                stop = start + self.clip_len
                self.clips.append(
                    (
                        sequence,
                        tuple(frame_indices[start:stop]),
                        tuple(frame_ids[start:stop]),
                    )
                )

        if not self.clips:
            raise RuntimeError("MOTDatasetRGBTClip found no complete clips")

    @property
    def input_dim(self):
        return self.frame_dataset.input_dim

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index):
        sequence, frame_indices, frame_ids = self.clips[index]
        frames = [self.frame_dataset[frame_index] for frame_index in frame_indices]
        images = torch.stack([torch.from_numpy(frame[0]) for frame in frames], dim=0)
        targets = torch.stack([torch.from_numpy(frame[1]) for frame in frames], dim=0)
        infos = [frame[2] for frame in frames]
        return images, targets, {
            "sequence_id": sequence,
            "frame_ids": tuple(int(frame_id) for frame_id in frame_ids),
            "infos": infos,
        }


def clip_collate(batch):
    """Collate fixed-size clips while retaining frame metadata."""

    if not batch:
        raise ValueError("cannot collate an empty clip batch")
    images, targets, metadata = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(targets, dim=0), list(metadata)


__all__ = ["MOTDatasetRGBTClip", "clip_collate", "_make_clip_starts"]
