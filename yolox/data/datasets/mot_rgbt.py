#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""COCO/MOT reader for paired visible and infrared MFT26-IR frames."""

import os

import cv2
import numpy as np

from .datasets_wrapper import Dataset
from .mot import MOTDataset


class MOTDatasetRGBT(MOTDataset):
    """MOTDataset variant that returns a six-channel RGB-IR image."""

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
        infrared = cv2.imread(ir_path, cv2.IMREAD_COLOR)
        if visible is None:
            raise FileNotFoundError(f"Unable to read visible image: {vi_path}")
        if infrared is None:
            raise FileNotFoundError(f"Unable to read infrared image: {ir_path}")
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
