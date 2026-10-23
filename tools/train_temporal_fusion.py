#!/usr/bin/env python3
"""Train the causal temporal fusion module on real consecutive clips.

The regular YOLOX trainer samples independent frames.  This entry point is
deliberately separate: it uses ``MOTDatasetRGBTClip`` and unfolds each clip in
time while the temporal GRU keeps its autograd state.  The detector branches
and head are frozen for the short F3 screening run, so the optimization is a
focused temporal-fusion adaptation of the fixed V3 detector.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from yolox.data import MOTDatasetRGBTClip, RGBTTrainTransform, clip_collate
from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-file", default="exps/yolox_mft26ir_rgbt_qra.py")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--clip-len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument(
        "--save-optimizer",
        action="store_true",
        help="keep optimizer state in the external checkpoint (uses extra space)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    return parser


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_model(exp, checkpoint_path, device):
    exp.fusion_mode = "temporal_complementarity"
    model = exp.get_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)

    # Only the causal temporal fusion module and the V3 scalar fusion logits
    # are adapted.  Frozen detector branches/head stay in eval mode so their
    # BatchNorm statistics cannot drift during the clip run.
    for parameter in model.parameters():
        parameter.requires_grad = False
    temporal_modules = []
    for module in model.backbone.fusion_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
        temporal_modules.append(module)
    model.backbone.fusion_logits.requires_grad = True

    model.train()
    model.backbone.rgb_branch.eval()
    model.backbone.ir_branch.eval()
    # The YOLOX head must remain in ``training`` mode to return detection
    # losses.  Its parameters are frozen, and its BatchNorm statistics are
    # explicitly held in eval mode so the temporal-only adaptation cannot
    # change the fixed detector.
    model.head.train()
    for module in model.head.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    model.set_fusion_stateful(True, detach_state=False)
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("temporal configuration produced no trainable parameters")
    return model, temporal_modules, trainable


def build_dataset(exp, args):
    cache_root = os.path.join(exp.causal_cache_root, "train")
    transform = RGBTTrainTransform(
        p=0.0,
        rgb_means=exp.rgb_means,
        std=exp.rgb_std,
        ir_means=exp.ir_means,
        ir_std=exp.ir_std,
        max_labels=500,
    )
    return MOTDatasetRGBTClip(
        data_dir=exp.data_dir,
        json_file=exp.train_ann,
        name="train",
        img_size=exp.input_size,
        preproc=transform,
        auxiliary_mode=exp.train_auxiliary_mode,
        causal_alpha=exp.causal_alpha,
        causal_cache_root=cache_root,
        sequence_filter=exp.protocol_train_sequences,
        clip_len=args.clip_len,
        stride=args.stride,
    )


def save_checkpoint(path: Path, model, optimizer, epoch, history, args):
    """Atomically keep one compact external checkpoint.

    The detector weights dominate the file size.  Optimizer state and per-
    epoch snapshots are optional so a failed write cannot leave several
    hundred megabytes of unusable partial artifacts behind.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "start_epoch": int(epoch + 1),
        "model": model.state_dict(),
        "history": history,
        "protocol": {
            "clip_len": args.clip_len,
            "stride": args.stride,
            "stateful_within_clip": True,
            "detach_between_clips": True,
            "trainable_scope": "temporal_fusion_and_static_logits",
            "optimizer_saved": bool(args.save_optimizer),
        },
    }
    if args.save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    args = make_parser().parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    seed_everything(args.seed)

    exp = get_exp(args.exp_file, None)
    exp.fusion_mode = "temporal_complementarity"
    dataset = build_dataset(exp, args)
    if args.max_clips is not None:
        if args.max_clips <= 0:
            raise ValueError("max-clips must be positive")
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(args.max_clips, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=clip_collate,
        drop_last=False,
    )

    device = torch.device(args.device)
    model, temporal_modules, trainable = configure_model(exp, args.ckpt, device)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    manifest = {
        "checkpoint": os.path.abspath(args.ckpt),
        "data_root": os.path.abspath(exp.data_dir),
        "sequences": list(exp.protocol_train_sequences),
        "clip_len": args.clip_len,
        "stride": args.stride if args.stride is not None else args.clip_len,
        "clips": len(dataset),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "trainable_parameter_count": int(sum(p.numel() for p in trainable)),
        "stateful_within_clip": True,
        "detach_between_clips": True,
        "auxiliary_mode": exp.train_auxiliary_mode,
    }
    with (output_dir / "train_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    for epoch in range(args.epochs):
        model.train()
        model.backbone.rgb_branch.eval()
        model.backbone.ir_branch.eval()
        model.head.train()
        for module in model.head.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        model.set_fusion_stateful(True, detach_state=False)
        epoch_loss = 0.0
        frame_count = 0
        start_time = time.perf_counter()
        for step, (images, targets, metadata) in enumerate(loader):
            del metadata
            model.reset_fusion_state()
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            clip_loss = None
            time_steps = images.shape[1]
            for time_index in range(time_steps):
                frame_targets = targets[:, time_index, :, :5]
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    outputs = model(images[:, time_index], frame_targets)
                    frame_loss = outputs["total_loss"] / float(time_steps)
                clip_loss = frame_loss if clip_loss is None else clip_loss + frame_loss
                frame_count += int(images.shape[0])
            if clip_loss is None or not torch.isfinite(clip_loss):
                raise FloatingPointError(
                    f"non-finite temporal clip loss at epoch={epoch} step={step}"
                )
            scaler.scale(clip_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(trainable, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            model.reset_fusion_state()
            epoch_loss += float(clip_loss.detach().cpu())

        elapsed = max(time.perf_counter() - start_time, 1e-9)
        record = {
            "epoch": epoch + 1,
            "mean_clip_loss": epoch_loss / max(len(loader), 1),
            "clips": len(loader),
            "frames": frame_count,
            "seconds": elapsed,
            "clips_per_second": len(loader) / elapsed,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        save_checkpoint(
            output_dir / "latest_ckpt.pth.tar", model, optimizer, epoch, history, args
        )

    with (output_dir / "train_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)
    print("saved:", str(output_dir / "latest_ckpt.pth.tar"))


if __name__ == "__main__":
    main()
