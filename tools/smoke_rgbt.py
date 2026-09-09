#!/usr/bin/env python3
"""Small GPU smoke test for the V1 paired pipeline."""

import argparse
import math

import torch

from yolox.exp import get_exp
from yolox.utils import postprocess


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_file", default="exps/yolox_mft26ir_rgbt_v1.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-batches", type=int, default=20)
    parser.add_argument("--inference-frames", type=int, default=200)
    parser.add_argument("--size", nargs=2, type=int, default=(320, 576))
    parser.add_argument("--device", default="cuda:0")
    return parser


def main():
    args = make_parser().parse_args()
    device = torch.device(args.device)
    exp = get_exp(args.exp_file, None)
    exp.input_size = tuple(args.size)
    exp.test_size = tuple(args.size)
    exp.random_size = None

    model = exp.get_model().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

    train_loader = exp.get_data_loader(
        batch_size=1, is_distributed=False, no_aug=True
    )
    model.train()
    rgb_grad = None
    ir_grad = None
    train_losses = []
    loader_iter = iter(train_loader)
    for step in range(args.train_batches):
        images, targets, _, _ = next(loader_iter)
        images = images.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, targets[:, :, :5])
        loss = outputs["total_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at batch {step}: {loss}")
        loss.backward()
        rgb_grad = model.backbone.rgb_branch.backbone.stem.conv.conv.weight.grad
        ir_grad = model.backbone.ir_branch.backbone.stem.conv.conv.weight.grad
        if rgb_grad is None or ir_grad is None:
            raise RuntimeError("one of the two branches has no gradient")
        if not torch.isfinite(rgb_grad).all() or not torch.isfinite(ir_grad).all():
            raise FloatingPointError(f"non-finite branch gradient at batch {step}")
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

    model.eval()
    dataset = exp._dataset(
        "test",
        exp.val_ann,
        __import__("yolox.data", fromlist=["RGBTValTransform"]).RGBTValTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    )
    detections = 0
    with torch.no_grad():
        for frame_index in range(args.inference_frames):
            image, _, _, _ = dataset[frame_index % len(dataset)]
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            raw = model(image)
            processed = postprocess(raw, 1, exp.test_conf, exp.nmsthre)
            if processed[0] is not None:
                detections += int(processed[0].shape[0])

    print("smoke: PASS")
    print("train_batches:", args.train_batches)
    print("inference_frames:", args.inference_frames)
    print("input_shape:", tuple(images.shape))
    print("loss_first:", train_losses[0])
    print("loss_last:", train_losses[-1])
    print("rgb_stem_grad_norm:", float(rgb_grad.norm().detach().cpu()))
    print("ir_stem_grad_norm:", float(ir_grad.norm().detach().cpu()))
    print("detections_after_nms:", detections)
    if not all(math.isfinite(value) for value in train_losses):
        raise FloatingPointError("non-finite loss was recorded")


if __name__ == "__main__":
    main()
