# encoding: utf-8
"""Causal Complementarity Fusion experiment family.

The detector and head stay on the validated V3 shape.  ``FUSION_MODE`` or the
normal YOLOX ``--opts fusion_mode ...`` switch selects one registered candidate
without duplicating the detector definition.
"""

import os

from yolox_mft26ir_rgbt_v1 import Exp as V1Exp


class Exp(V1Exp):
    def __init__(self):
        super().__init__()
        self.exp_name = "qra_dynamic_quality"
        self.fusion_mode = os.environ.get("MFT26IR_FUSION_MODE", "dynamic_quality")
        self.auxiliary_mode = os.environ.get("MFT26IR_AUXILIARY_MODE", "legacy")
        self.train_auxiliary_mode = os.environ.get(
            "MFT26IR_TRAIN_AUXILIARY_MODE", self.auxiliary_mode
        )
        # Stage 1 is deliberately short; extension is selected from Dev loss
        # and metrics, never from the historical Test set.
        self.max_epoch = 3
        self.warmup_epochs = 1
        self.no_aug_epochs = 0
        self.enable_mixup = False


if __name__ == "__main__":
    print(Exp())

