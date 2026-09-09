#!/usr/bin/env python3
import os
import random
import unittest

import numpy as np

from yolox.data.data_augment_rgbt import RGBTTrainTransform
from yolox.data.datasets.mot_rgbt import MOTDatasetRGBT


DATA_ROOT = os.environ.get(
    "MFT26IR_ROOT", "/data1/LWR/vranlee/CELL/PFTrack/data/mft26-ir"
)


class RGBTDataPairingTest(unittest.TestCase):
    def test_fifty_train_pairs_and_shared_transform(self):
        dataset = MOTDatasetRGBT(
            data_dir=DATA_ROOT,
            json_file="train.json",
            name="train",
            img_size=(320, 576),
            preproc=None,
        )
        rng = random.Random(20260910)
        indices = rng.sample(range(len(dataset)), 50)
        transform = RGBTTrainTransform(
            p=1.0,
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_labels=500,
        )

        for index in indices:
            visible, infrared, target, info, _ = dataset.pull_item(index)
            self.assertEqual(visible.shape[:2], infrared.shape[:2])
            vi_name, ir_name = info[4], info[5]
            self.assertEqual(vi_name.split("/")[0], ir_name.split("/")[0])
            self.assertEqual(vi_name.split("/")[-1], ir_name.split("/")[-1])
            image, padded = transform(
                visible, infrared, target, (320, 576)
            )
            self.assertEqual(image.shape, (6, 320, 576))
            self.assertEqual(padded.shape, (500, 6))
            self.assertTrue(np.isfinite(image).all())
            self.assertTrue(np.isfinite(padded).all())


if __name__ == "__main__":
    unittest.main()
