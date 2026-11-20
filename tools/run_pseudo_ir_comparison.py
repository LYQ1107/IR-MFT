#!/usr/bin/env python3
"""Matched V3 adaptation and ByteTrack evaluation of versioned pseudo IR.

One process uses one explicitly allowed physical GPU. Run old/new workers on
GPU2/GPU3 concurrently. Images are read in place; only compact checkpoints,
post-NMS detections and text tracks are stored outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from yolox.data import RGBTTrainTransform, RGBTValTransform
from yolox.exp import get_exp
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils import ModelEMA, postprocess
from tools.eval_mft26ir import compute_motmetrics, run_trackeval, write_sequence_results
from tools.init_rgbt_from_byte_checkpoint import build_mapping
from tools.pseudo_ir_continuation import checkpoint_training_budget, check_resume, joint_lr, load_plan


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def guard_storage(output, reserve_gib):
    Path(output).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free / 2**30
    if free < reserve_gib:
        raise RuntimeError(f"Storage reserve would be violated: {free:.1f} GiB free")
    return free


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)


def setup(args):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"2", "3"}:
        raise RuntimeError("Each GPU worker requires CUDA_VISIBLE_DEVICES=2 or 3")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    guard_storage(args.output, args.reserve_gib)
    exp = get_exp(str(ROOT / "exps/yolox_mft26ir_pseudo_ir_comparison.py"), None)
    # The experiment reads normalization after this variable is set in main.
    exp.input_size = tuple(args.train_size)
    exp.test_size = (800, 1440)
    if args.ir_stats:
        stats = json.loads(Path(args.ir_stats).read_text())
        if stats.get("split") != "train" or not stats.get("complete"):
            raise ValueError("Auxiliary statistics must come from the complete Train split")
        exp.ir_means = tuple(stats["mean_0_1"])
        exp.ir_std = tuple(stats["std_0_1"])
    return exp


def make_dataset(exp, args, split, training):
    transform_cls = RGBTTrainTransform if training else RGBTValTransform
    opts = dict(rgb_means=exp.rgb_means, std=exp.rgb_std,
                ir_means=exp.ir_means, ir_std=exp.ir_std)
    if training:
        opts.update(p=0.5, max_labels=500)
    dataset = exp._dataset(split, "test.json" if split == "test" else "train.json",
                           transform_cls(**opts))
    if args.sequences:
        requested = set(args.sequences.split(","))
        unknown = requested - {a[2].split("/", 1)[0] for a in dataset.annotations}
        if unknown:
            raise ValueError(f"Sequences outside {split} protocol: {unknown}")
        dataset = Subset(dataset, [i for i, a in enumerate(dataset.annotations)
                                  if a[2].split("/", 1)[0] in requested])
    if args.max_frames:
        dataset = Subset(dataset, range(min(args.max_frames, len(dataset))))
    return dataset


def model_from_checkpoint(exp, args):
    if args.variant == "rgb":
        visible_exp = get_exp(str(ROOT / "exps/yolox_mft26ir_visible.py"), None)
        model = visible_exp.get_model()
    else:
        model = exp.get_model()
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    source = checkpoint.get("train_model" if args.resume else "model", checkpoint)
    paired_source = "backbone.fusion_logits" in source
    if args.variant == "rgb" and paired_source:
        state = {}
        for key in model.state_dict():
            source_key = ("backbone.rgb_branch." + key[len("backbone."):]
                          if key.startswith("backbone.") else key)
            state[key] = source[source_key]
        mapping = "V3_RGB_branch_and_shared_head"
    elif args.variant != "rgb" and not paired_source:
        state, _, _ = build_mapping(source, model.state_dict())
        mapping = "duplicate_RGB_into_both_branches"
    else:
        state = source
        mapping = "strict_direct"
    model.load_state_dict(state, strict=True)
    training_budget = checkpoint_training_budget(checkpoint)
    del checkpoint, source, state
    return model.cuda(), mapping, training_budget


def train_mode(model, variant, calibration):
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not calibration or name.startswith("backbone.ir_branch.")
                                 or name == "backbone.fusion_logits")
    # Keep pretrained normalization stable for both modalities and the RGB
    # control. Small batches must not silently rewrite detector BN statistics.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    if variant != "rgb":
        model.backbone.rgb_branch.eval() if calibration else None


def train(args, exp, model, mapping):
    dataset = make_dataset(exp, args, "train", True)
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        worker_init_fn=seed_worker, generator=g,
                        persistent_workers=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=128.0)
    ema = ModelEMA(model)
    history = []
    start_epoch = 0
    calibration_epochs = 0 if args.variant == "rgb" else args.calibration_epochs
    epochs = calibration_epochs + args.finetune_epochs
    continuation, plan_sha = load_plan(args.continuation_plan) if args.continuation_plan else (None, None)
    if continuation:
        if not args.resume or not args.stop_after_joint_epochs:
            raise ValueError("Continuation requires --resume and an explicit joint-epoch stage boundary")
        if not args.no_checkpoint and (args.max_steps or args.max_frames or args.sequences):
            raise ValueError("A formal continuation must include every Train frame and update")
        if not (3 < args.stop_after_joint_epochs <= args.finetune_epochs):
            raise ValueError("Continuation stage boundary is outside the planned budget")
        epochs = calibration_epochs + args.stop_after_joint_epochs
    manifest = dict(vars(args), physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                    initialization=mapping, source_sha256=sha256(args.ckpt),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    train_frames=len(dataset), train_sequences=list(exp.protocol_train_sequences),
                    test_size=list(exp.test_size), ir_means=exp.ir_means, ir_std=exp.ir_std,
                    rgb_means=exp.rgb_means, rgb_std=exp.rgb_std,
                    bn_policy="frozen_all_running_statistics", torch=torch.__version__,
                    checkpoint_selection="fixed_final_epoch_not_Test",
                    storage_free_gib=guard_storage(args.output, args.reserve_gib))
    dump(Path(args.output) / "train_manifest.json", manifest)
    if args.resume:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        prior = checkpoint["manifest"]
        check_resume(prior, manifest, checkpoint, continuation, plan_sha)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        ema.ema.load_state_dict(checkpoint["model"], strict=True)
        ema.updates = checkpoint["ema_updates"]
        history = checkpoint["history"]
        start_epoch = checkpoint["completed_epochs"]
        if start_epoch > epochs:
            raise ValueError("Resume checkpoint is later than this requested stage boundary")
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        g.set_state(checkpoint["loader_rng"])
        if continuation:
            event = dict(source=str(args.ckpt), source_sha256=manifest["source_sha256"],
                         completed_epochs=start_epoch, target_joint_epochs=args.stop_after_joint_epochs,
                         physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"], git_commit=manifest["git_commit"],
                         runner_sha256=sha256(__file__), started_unix=time.time())
            manifest = dict(prior, output=args.output, finetune_epochs=args.finetune_epochs,
                            max_steps=args.max_steps, no_checkpoint=args.no_checkpoint,
                            physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                            checkpoint_selection="best_Dev_HOTA_across_fixed_three_joint_epoch_checks",
                            continuation=dict(plan=str(Path(args.continuation_plan).resolve()), plan_sha256=plan_sha),
                            execution_events=prior.get("execution_events", []) + [event])
        else:
            manifest = prior
        dump(Path(args.output) / "train_manifest.json", manifest)
        dump(Path(args.output) / "resume_event.json", dict(completed_epochs=start_epoch,
              source=str(args.ckpt), source_sha256=sha256(args.ckpt)))
        del checkpoint
    for epoch in range(start_epoch, epochs):
        calibration = epoch < calibration_epochs
        train_mode(model, args.variant, calibration)
        phase_epoch = epoch if calibration else epoch - calibration_epochs
        phase_epochs = calibration_epochs if calibration else args.finetune_epochs
        # Original V2/V3 rates are specified at reference batch64.
        reference_lr = args.calibration_lr if calibration else args.lr
        effective_batch = args.batch_size * args.accumulate
        lr = (joint_lr(continuation, phase_epoch, effective_batch) if continuation and not calibration else
              reference_lr * effective_batch / 64.0 * (0.2 + 0.8 * (
                  1 + math.cos(math.pi * phase_epoch / max(phase_epochs, 1))) / 2))
        for group in optimizer.param_groups:
            group["lr"] = lr
        running = 0.0
        frames = 0
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        skipped_updates = 0
        for step, (images, targets, _, _) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            if args.variant == "rgb":
                images = images[:, :3]
            targets = targets.cuda(non_blocking=True)[:, :, :5]
            accumulation = min(args.accumulate, len(loader) - (step // args.accumulate) * args.accumulate)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(images, targets)["total_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss epoch {epoch} step {step}")
            scaler.scale(loss / accumulation).backward()
            if (step + 1) % args.accumulate == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= previous_scale:
                    ema.update(model)
                else:
                    skipped_updates += 1
            running += float(loss.detach())
            frames += int(images.shape[0])
            if (step + 1) % 100 == 0 or step + 1 == len(loader):
                elapsed = time.perf_counter() - start
                progress = dict(epoch=epoch + 1, phase="calibration" if calibration else "V3_finetune",
                                step=step + 1, steps=len(loader), frames=frames,
                                mean_loss=running / (step + 1), seconds=elapsed,
                                fps=frames / elapsed, lr=lr,
                                gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                                skipped_amp_updates=skipped_updates)
                dump(Path(args.output) / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if args.max_steps and step + 1 >= args.max_steps:
                break
        record = dict(epoch=epoch + 1, phase="calibration" if calibration else "V3_finetune",
                      frames=frames, loss=running / (step + 1), seconds=time.perf_counter() - start,
                      lr=lr, skipped_amp_updates=skipped_updates,
                      fusion_gates=(torch.sigmoid(model.backbone.fusion_logits).detach().cpu().tolist()
                                          if args.variant != "rgb" else None))
        history.append(record)
        dump(Path(args.output) / "train_history.json", history)
        if args.no_checkpoint:
            print("BENCHMARK_COMPLETE", json.dumps(dict(record,
                  gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)
            continue
        guard_storage(args.output, args.reserve_gib)
        # One atomic resumable checkpoint; no per-epoch image/prediction dump.
        target = Path(args.output) / "latest_ckpt.pth.tar"
        temporary = target.with_suffix(".tmp")
        torch.save(dict(model=ema.ema.state_dict(), train_model=model.state_dict(),
                        optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        completed_epochs=epoch + 1, ema_updates=ema.updates, manifest=manifest,
                        history=history, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                        python_rng=random.getstate(), loader_rng=g.get_state()), temporary)
        os.replace(temporary, target)
        print("EPOCH_COMPLETE", json.dumps(record), flush=True)
    target = Path(args.output) / "latest_ckpt.pth.tar"
    status = "BENCHMARK" if args.no_checkpoint else ("STAGE_COMPLETE" if continuation else "COMPLETE")
    dump(Path(args.output) / "completion.json", dict(status=status,
                epochs=len(history), checkpoint=None if args.no_checkpoint else str(target),
                sha256=None if args.no_checkpoint else sha256(target),
                joint_epochs=len(history) - calibration_epochs,
                continuation_plan_sha256=plan_sha))


def infer(args, exp, model, mapping, training_budget):
    dataset = make_dataset(exp, args, args.split, False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        worker_init_fn=seed_worker, persistent_workers=args.workers > 0,
                        collate_fn=lambda items: items)
    model.half().eval()
    rows, caches = {}, {}
    tracker, current = None, None
    total = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for items in loader:
            images = torch.from_numpy(np.stack([x[0] for x in items])).cuda().half()
            if args.variant == "rgb":
                images = images[:, :3]
            outputs = postprocess(model(images), 1, args.conf, args.nms)
            for item, output in zip(items, outputs):
                info = item[2]
                seq, frame = info[4].split("/", 1)[0], int(info[2])
                if seq != current:
                    tracker = BYTETracker(SimpleNamespace(track_thresh=args.track_thresh,
                        track_buffer=args.track_buffer, match_thresh=args.match_thresh, mot20=False), frame_rate=25)
                    current = seq
                    rows[seq], caches[seq] = [], []
                if output is None:
                    det = np.empty((0, 5), np.float32)
                else:
                    a = output.float().cpu().numpy()
                    scale = min(exp.test_size[0] / info[0], exp.test_size[1] / info[1])
                    det = np.column_stack([a[:, :4] / scale, a[:, 4] * a[:, 5]]).astype(np.float32)
                caches[seq].append((frame, det))
                for track in tracker.update(det, (1, 1), (1, 1)):
                    if track.tlwh[2] * track.tlwh[3] > 100:
                        rows[seq].append((frame, track.tlwh.copy(), track.track_id, track.score))
                total += 1
            if total % 256 < args.batch_size or total == len(dataset):
                progress = dict(frames=total, expected=len(dataset), seconds=time.perf_counter() - start)
                dump(Path(args.output) / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    result = Path(args.output) / "track_results"
    result.mkdir(parents=True, exist_ok=True)
    for seq, data in rows.items():
        write_sequence_results(str(result / (seq + ".txt")), data)
        pairs = caches[seq]
        dets = np.concatenate([x[1] for x in pairs], axis=0)
        dest = Path(args.output) / "detections"
        dest.mkdir(exist_ok=True)
        np.savez_compressed(dest / (seq + ".npz"), frame_ids=np.array([x[0] for x in pairs]),
                            offsets=np.r_[0, np.cumsum([len(x[1]) for x in pairs])],
                            boxes=dets[:, :4], scores=dets[:, 4])
    protocol = dict(vars(args), initialization=mapping, frames=total, sequences=list(rows),
                    physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"], ir_means=exp.ir_means,
                    ir_std=exp.ir_std, input_size=list(exp.test_size), source_sha256=sha256(args.ckpt),
                    generated_ir_root=exp.generated_ir_root if args.variant == "diffv2ir" else None)
    protocol["source_training"] = training_budget
    if training_budget:
        protocol["calibration_epochs"] = training_budget["calibration_epochs"]
        protocol["finetune_epochs"] = training_budget["joint_epochs"]
    elapsed = time.perf_counter() - start
    report = dict(protocol=protocol, runtime=dict(seconds=elapsed, fps=total / elapsed))
    if not args.skip_evaluation and not args.max_frames:
        report["motmetrics"] = compute_motmetrics(str(result), args.gt_root, list(rows),
                                                   data_split="test" if args.split == "test" else "train")
        report["trackeval"] = run_trackeval(str(result), args.gt_root, args.output, list(rows),
                                             args.trackeval_root,
                                             data_split="test" if args.split == "test" else "train")
    dump(Path(args.output) / "metrics.json", report)
    print("EVALUATION_COMPLETE", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["train", "infer"])
    parser.add_argument("--variant", choices=["rgb", "legacy", "causal_cache", "diffv2ir"], required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--resume", action="store_true", help="resume from a saved full epoch boundary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--train-size", type=int, nargs=2, default=[480, 864])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ir-stats", help="optional three-channel train-only statistics JSON")
    parser.add_argument("--calibration-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--continuation-plan", help="explicit frozen plan allowing a longer resumed joint budget")
    parser.add_argument("--stop-after-joint-epochs", type=int, help="complete this continuation stage then return for Dev evaluation")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--calibration-lr", type=float, default=0.001)
    parser.add_argument("--memory-fraction", type=float, default=0.55)
    parser.add_argument("--reserve-gib", type=float, default=40.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-checkpoint", action="store_true", help="benchmark only; never a formal run")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--sequences")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--nms", type=float, default=0.7)
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.9)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--gt-root", default="/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt")
    parser.add_argument("--trackeval-root", default="/data3/liuyeqiang/IR-MFT_assets/vendor/VUP-PFTrack")
    args = parser.parse_args()
    os.environ["MFT26IR_COMPARISON_INPUT"] = "repeat_rgb" if args.variant == "rgb" else args.variant
    exp = setup(args)
    model, mapping, training_budget = model_from_checkpoint(exp, args)
    if args.action == "train":
        train(args, exp, model, mapping)
    else:
        infer(args, exp, model, mapping, training_budget)


if __name__ == "__main__":
    main()
