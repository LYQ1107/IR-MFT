# encoding: utf-8
"""V3: calibrated gated fusion with cautious end-to-end fine tuning."""

# Experiment files are imported as top-level modules by yolox.exp.build.
from yolox_mft26ir_rgbt_v2 import Exp as V2Exp


class Exp(V2Exp):
    """Continue from V2 after the safe IR-only adaptation.

    V2 establishes a calibrated complementary stream without changing the
    visible detector.  V3 releases both RGB and head parameters with a lower
    fine-tuning learning rate so the detector can learn how to consume the
    fused features while retaining the learned level-wise gates.
    """

    def __init__(self):
        super().__init__()
        self.exp_name = "v3_gated_end_to_end"
        self.freeze_rgb = False
        self.freeze_head = False
        self.basic_lr_per_img = 0.0002 / 64.0
        self.max_epoch = 6


if __name__ == "__main__":
    print(Exp())
