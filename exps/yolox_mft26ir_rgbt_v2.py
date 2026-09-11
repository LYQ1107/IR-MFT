# encoding: utf-8
"""V2: calibrated IR with RGB-dominant learnable gated residual fusion."""

import torch
import torch.nn as nn

# yolox.exp.build imports experiment files as top-level modules after adding
# their directory to sys.path, so use the same import style here.
from yolox_mft26ir_rgbt_v1 import Exp as V1Exp


class Exp(V1Exp):
    """Conservative multimodal adaptation initialized from visible ByteTrack.

    The RGB branch and detection head remain fixed in this first diagnostic
    variant.  Only the calibrated IR branch and three level-wise fusion gates
    are learned, so an unhelpful IR stream cannot erase the visible baseline.
    """

    def __init__(self):
        super().__init__()
        self.exp_name = "v2_gated_calibrated"
        self.fusion_mode = "gated_residual"
        self.freeze_rgb = True
        self.freeze_head = True

        # The statistics are in RGB order because preproc reverses OpenCV's
        # BGR input before normalization.  They were estimated from a
        # representative sample of the MFT26-IR training frames.
        self.ir_means = (0.0651, 0.0262, 0.0581)
        self.ir_std = (0.1916, 0.1015, 0.1084)

        # Keep the paired branch within the memory-tested development shape.
        # The detector starts from the already trained 800x1440 visible model;
        # this run adapts the complementary stream without re-training it.
        self.input_size = (640, 1152)
        self.test_size = (800, 1440)
        self.random_size = None
        self.max_epoch = 3
        self.no_aug_epochs = 0
        self.enable_mixup = False
        self.eval_interval = 100

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
                freeze_rgb=self.freeze_rgb,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        if self.freeze_head:
            for parameter in self.model.head.parameters():
                parameter.requires_grad = False
        return self.model

    def get_optimizer(self, batch_size):
        """Build YOLOX parameter groups while including fusion logits."""
        if "optimizer" in self.__dict__:
            return self.optimizer

        if self.warmup_epochs > 0:
            lr = self.warmup_lr
        else:
            lr = self.basic_lr_per_img * batch_size

        pg0, pg1, pg2 = [], [], []
        for name, module in self.model.named_modules():
            if (
                hasattr(module, "bias")
                and isinstance(module.bias, nn.Parameter)
                and module.bias.requires_grad
            ):
                pg2.append(module.bias)
            if isinstance(module, nn.BatchNorm2d) or "bn" in name:
                if hasattr(module, "weight") and isinstance(
                    module.weight, nn.Parameter
                ) and module.weight.requires_grad:
                    pg0.append(module.weight)
            elif hasattr(module, "weight") and isinstance(
                module.weight, nn.Parameter
            ) and module.weight.requires_grad:
                pg1.append(module.weight)

        fusion_logits = self.model.backbone.fusion_logits
        if fusion_logits.requires_grad:
            pg0.append(fusion_logits)

        optimizer = torch.optim.SGD(
            pg0, lr=lr, momentum=self.momentum, nesterov=True
        )
        if pg1:
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )
        if pg2:
            optimizer.add_param_group({"params": pg2})
        self.optimizer = optimizer
        return optimizer


if __name__ == "__main__":
    # Keep the experiment file importable by the normal YOLOX launcher.
    print(Exp())
