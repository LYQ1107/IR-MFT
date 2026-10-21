# encoding: utf-8
"""V1: independent RGB/IR YOLOPAFPN branches with average fusion."""

import json
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
            "/data3/liuyeqiang/datasets/mft26-ir",
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
        # Paired JPEG decoding and geometric preprocessing are CPU-bound.
        # InfiniteSampler now emits plain Python indices, so multiprocessing
        # is safe and keeps the GPUs fed during the two-card run.
        self.data_num_workers = 4

        protocol_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "cfp",
            "protocol_v2.json",
        )
        with open(protocol_path, "r", encoding="utf-8") as handle:
            self.protocol = json.load(handle)
        self.protocol_train_sequences = tuple(self.protocol["train"])
        self.protocol_dev_sequences = tuple(self.protocol["dev"])
        self.protocol_test_sequences = tuple(self.protocol["historical_test"])

        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.rgb_means = (0.485, 0.456, 0.406)
        self.rgb_std = (0.229, 0.224, 0.225)
        # V1 keeps the historical visible normalization for both streams.
        # Later experiments can override only the IR statistics.
        self.ir_means = self.rgb_means
        self.ir_std = self.rgb_std
        self.fusion_mode = "avg"
        # ``legacy`` reads the archived RGB-derived auxiliary folders.  Strict
        # causal generation is selected by the evaluation wrapper with
        # workers=0; random training batches must use a precomputed causal
        # cache or the clip loader before they are labelled causal.
        self.auxiliary_mode = os.environ.get("MFT26IR_AUXILIARY_MODE", "legacy")
        self.train_auxiliary_mode = os.environ.get(
            "MFT26IR_TRAIN_AUXILIARY_MODE", self.auxiliary_mode
        )
        self.causal_alpha = 0.01
        self.causal_cache_root = os.environ.get(
            "MFT26IR_CAUSAL_CACHE_ROOT",
            "/data3/liuyeqiang/IR-MFT_assets/causal_aux_cache",
        )

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

    def _dataset(self, split, annotation, transform, data_name=None):
        if data_name is None:
            data_name = "train" if split in {"train", "dev"} else "test"
        if split == "train":
            sequence_filter = self.protocol_train_sequences
        elif split == "dev":
            sequence_filter = self.protocol_dev_sequences
        elif split == "test":
            sequence_filter = self.protocol_test_sequences
        else:
            sequence_filter = None
        return MOTDatasetRGBT(
            data_dir=self.data_dir,
            json_file=annotation,
            name=data_name,
            img_size=self.input_size if split == "train" else self.test_size,
            preproc=transform,
            auxiliary_mode=(
                self.train_auxiliary_mode if split == "train" else self.auxiliary_mode
            ),
            causal_alpha=self.causal_alpha,
            causal_cache_root=self.causal_cache_root,
            sequence_filter=sequence_filter,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        dataset = self._dataset(
            "train",
            self.train_ann,
            RGBTTrainTransform(
                p=0.5,
                rgb_means=self.rgb_means,
                std=self.rgb_std,
                ir_means=self.ir_means,
                ir_std=self.ir_std,
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
            pin_memory=True,
            batch_sampler=batch_sampler,
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        del testdev
        dataset = self._dataset(
            "test",
            self.val_ann,
            RGBTValTransform(
                rgb_means=self.rgb_means,
                std=self.rgb_std,
                ir_means=self.ir_means,
                ir_std=self.ir_std,
            ),
        )
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        eval_workers = 0 if self.auxiliary_mode == "causal" else self.data_num_workers
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=eval_workers,
            pin_memory=True,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
