"""V3 fusion with explicitly versioned auxiliary data and train-only scaling."""

import json
import os

from yolox_mft26ir_rgbt_v3 import Exp as V3Exp


class Exp(V3Exp):
    def __init__(self):
        super().__init__()
        self.exp_name = "pseudo_ir_comparison_20261120"
        self.auxiliary_mode = os.environ.get("MFT26IR_COMPARISON_INPUT", "diffv2ir")
        self.train_auxiliary_mode = self.auxiliary_mode
        self.generated_ir_root = os.environ.get(
            "MFT26IR_GENERATED_IR_ROOT",
            "/data3/liuyeqiang/nas/A100/IR-MFT/diffv2ir_final_20261104/full_generation",
        )
        self.generated_ir_stats = os.environ.get(
            "MFT26IR_GENERATED_IR_STATS",
            "/data3/liuyeqiang/IR-MFT/diffv2ir_final_20261104/train_stats.json",
        )
        if self.auxiliary_mode == "diffv2ir":
            with open(self.generated_ir_stats, encoding="utf-8") as handle:
                stats = json.load(handle)
            if not stats.get("complete") or stats.get("split") != "train":
                raise ValueError("DiffV2IR normalization requires complete train-only statistics")
            self.ir_means = (float(stats["mean_0_1"]),) * 3
            self.ir_std = (float(stats["std_0_1"]),) * 3
        self.input_size = (480, 864)
        self.test_size = (800, 1440)
        self.random_size = None
