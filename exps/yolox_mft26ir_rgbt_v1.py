# encoding: utf-8
"""V1: independent RGB/IR YOLOPAFPN branches with average fusion."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from yolox.data import (
    DataLoader,
    InfiniteSampler,
    MOTDatasetRGBT,
    RGBTTrainTransform,
    RGBTValTransform,
    YoloBatchSampler,
)
from yolox.evaluators import COCOEvaluator
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25

        self.exp_name = "v1_avg"
        self.output_dir = "./results"
        self.data_dir = os.environ.get(
            "MFT26IR_ROOT",
            "/data1/LWR/vranlee/CELL/PFTrack/data/mft26-ir",
        )
        self.train_ann = "train.json"
        self.val_ann = "test.json"
        # The historical visible-only checkpoint was trained with multi-scale
        # YOLOX inputs. Use a smaller fixed training size for the first paired
        # development run so two independent YOLOPAFPN branches fit with a
        # useful distributed batch size. Evaluation remains 800x1440 below.
        self.input_size = (640, 1152)
        self.test_size = (800, 1440)

        self.random_size = None
        self.max_epoch = 3
        self.print_interval = 20
        # Formal MFT26-IR evaluation is run by tools/eval_mft26ir.py after
        # training. Avoid an invalid COCO-only test pass after every epoch.
        self.eval_interval = 100
        self.no_aug_epochs = 0
        self.enable_mixup = False
        self.data_num_workers = 0

        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.fusion_mode = "avg"

    def get_model(self):
        from yolox.models import YOLOPAFPNRGBT, YOLOX, YOLOXHead

        def init_yolo(module):
            for submodule in module.modules():
                if isinstance(submodule, nn.BatchNorm2d):
                    submodule.eps = 1e-3
                    submodule.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPNRGBT(
                self.depth,
                self.width,
                in_channels=in_channels,
                fusion_mode=self.fusion_mode,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def _dataset(self, split, annotation, transform):
        return MOTDatasetRGBT(
            data_dir=self.data_dir,
            json_file=annotation,
            name=split,
            img_size=self.input_size if split == "train" else self.test_size,
            preproc=transform,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        dataset = self._dataset(
            "train",
            self.train_ann,
            RGBTTrainTransform(
                p=0.5,
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(dataset), seed=self.seed if self.seed is not None else 0
        )
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=False,
        )
        return DataLoader(
            dataset,
            num_workers=self.data_num_workers,
            pin_memory=False,
            batch_sampler=batch_sampler,
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        del testdev
        dataset = self._dataset(
            "test",
            self.val_ann,
            RGBTValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.data_num_workers,
            pin_memory=False,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
