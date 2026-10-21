#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""COCO/MOT reader for paired visible and infrared MFT26-IR frames."""

import os

import cv2
import numpy as np

from .datasets_wrapper import Dataset
from .mot import MOTDataset
from ..causal_auxiliary import CausalAuxiliaryGenerator, grayscale_three_channel


class MOTDatasetRGBT(MOTDataset):
    """MOTDataset variant that returns a six-channel RGB-IR image."""

    AUXILIARY_MODES = {
        "legacy",
        "causal",
        "zero",
        "shuffle",
        "repeat_rgb",
        "grayscale_rgb",
        "causal_cache",
    }

    def __init__(
        self,
        *args,
        auxiliary_mode="legacy",
        causal_alpha=0.01,
        sequence_filter=None,
        **kwargs,
    ):
        if auxiliary_mode not in self.AUXILIARY_MODES:
            raise ValueError(
                f"unknown auxiliary_mode={auxiliary_mode!r}; "
                f"expected one of {sorted(self.AUXILIARY_MODES)}"
            )
        self.auxiliary_mode = auxiliary_mode
        self.causal_alpha = float(causal_alpha)
        self.causal_cache_root = kwargs.pop("causal_cache_root", None)
        self.sequence_filter = (
            set(sequence_filter) if sequence_filter is not None else None
        )
        self._causal_auxiliary = CausalAuxiliaryGenerator(alpha=self.causal_alpha)
        super().__init__(*args, **kwargs)
        if self.sequence_filter is not None:
            keep = [
                index
                for index, annotation in enumerate(self.annotations)
                if annotation[2].split("/", 1)[0] in self.sequence_filter
            ]
            self.ids = [self.ids[index] for index in keep]
            self.annotations = [self.annotations[index] for index in keep]
        self._sequence_frame_paths = {}
        for annotation in self.annotations:
            vi_path = annotation[2]
            sequence = vi_path.split("/", 1)[0]
            self._sequence_frame_paths.setdefault(sequence, []).append(vi_path)
        self._last_sequence = None
        self._last_frame_id = None

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objects = []
        for obj in annotations:
            x1, y1, box_w, box_h = obj["bbox"]
            x2 = x1 + box_w
            y2 = y1 + box_h
            if obj.get("area", box_w * box_h) > 0 and x2 >= x1 and y2 >= y1:
                objects.append((obj, [x1, y1, x2, y2]))

        result = np.zeros((len(objects), 6), dtype=np.float32)
        for index, (obj, clean_bbox) in enumerate(objects):
            result[index, :4] = clean_bbox
            result[index, 4] = self.class_ids.index(obj["category_id"])
            result[index, 5] = obj["track_id"]

        vi_file_name = im_ann["vi_file_name"]
        ir_file_name = im_ann["ir_file_name"]
        img_info = (
            height,
            width,
            frame_id,
            video_id,
            vi_file_name,
            ir_file_name,
        )
        return result, img_info, vi_file_name, ir_file_name

    def pull_item(self, index):
        id_ = self.ids[index]
        result, img_info, vi_file_name, ir_file_name = self.annotations[index]

        vi_path = os.path.join(self.data_dir, self.name, vi_file_name)
        ir_path = os.path.join(self.data_dir, self.name, ir_file_name)
        visible = cv2.imread(vi_path, cv2.IMREAD_COLOR)
        if visible is None:
            raise FileNotFoundError(f"Unable to read visible image: {vi_path}")
        if self.auxiliary_mode == "causal":
            infrared = self._causal_auxiliary(
                visible,
                sequence_id=vi_file_name.split("/", 1)[0],
                frame_id=int(img_info[2]),
            )
        elif self.auxiliary_mode == "causal_cache":
            if not self.causal_cache_root:
                raise ValueError("causal_cache mode requires causal_cache_root")
            cache_path = os.path.join(
                self.causal_cache_root,
                vi_file_name.rsplit("/", 1)[0],
                os.path.basename(vi_file_name).rsplit(".", 1)[0] + ".webp",
            )
            infrared = cv2.imread(cache_path, cv2.IMREAD_COLOR)
            if infrared is None:
                raise FileNotFoundError(
                    f"Unable to read causal cache image: {cache_path}"
                )
        elif self.auxiliary_mode == "zero":
            infrared = np.zeros_like(visible)
        elif self.auxiliary_mode == "repeat_rgb":
            infrared = visible.copy()
        elif self.auxiliary_mode == "grayscale_rgb":
            infrared = grayscale_three_channel(visible)
        elif self.auxiliary_mode == "shuffle":
            sequence = vi_file_name.split("/", 1)[0]
            paths = self._sequence_frame_paths[sequence]
            current = paths.index(vi_file_name)
            shuffled_path = os.path.join(self.data_dir, self.name, paths[(current + 1) % len(paths)])
            infrared = cv2.imread(shuffled_path, cv2.IMREAD_COLOR)
        else:
            infrared = cv2.imread(ir_path, cv2.IMREAD_COLOR)
        if infrared is None:
            raise FileNotFoundError(
                f"Unable to read auxiliary image: {ir_path} (mode={self.auxiliary_mode})"
            )
        if visible.shape[:2] != infrared.shape[:2]:
            raise ValueError(
                "RGB and IR image sizes differ for "
                f"{vi_file_name}: {visible.shape[:2]} vs {infrared.shape[:2]}"
            )

        return visible, infrared, result.copy(), img_info, np.array([id_])

    @Dataset.resize_getitem
    def __getitem__(self, index):
        visible, infrared, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            image, target = self.preproc(
                visible, infrared, target, self.input_dim
            )
        else:
            image = np.concatenate(
                (visible.transpose(2, 0, 1), infrared.transpose(2, 0, 1)), axis=0
            )
        return image, target, img_info, img_id
