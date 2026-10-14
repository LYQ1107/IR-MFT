#!/usr/bin/env python3
"""Train one fixed FDTA-style adapter and record held-out crop retrieval."""
import argparse
import bisect
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from cfp.crop_retrieval import embedding_crop_retrieval
from cfp.fdta_fish_adapter import FDTAFishAdapter
from cfp.local_identity_training import SameFrameTemporalSampler, local_identity_objective
from cfp.provenance import sha256, snapshot_sources, write_json


def load_banks(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "COMPLETE_FROZEN_DINOV2_TWO_CROP_EVIDENCE":
        raise ValueError("Complete frozen DINO two-crop source required")
    banks = {}
    for video, entry in manifest["videos"].items():
        directory = root / video
        records = json.loads((directory / "records.json").read_text())
        arrays = {name: np.load(directory / (name + ".npy"), mmap_mode="r")
                  for name in ("cls", "local4", "visibility4", "valid")}
        if len(records) != len(arrays["valid"]) or arrays["cls"].shape[1:] != (384,):
            raise ValueError("Aligned DINO crop records/features required")
        for name, info in entry["artifacts"].items():
            if name in arrays and sha256(directory / (name + ".npy")) != info["sha256"]:
                raise ValueError("Frozen DINO artifact drift: " + video + "/" + name)
        by_identity = {}
        for index, row in enumerate(records):
            if arrays["valid"][index] and row["identity"] >= 0:
                by_identity.setdefault(int(row["identity"]), []).append(index)
        for identity, indices in by_identity.items():
            indices.sort(key=lambda j: (int(records[j]["frame"]), j))
        banks[video] = {"records": records, "arrays": arrays, "by_identity": by_identity}
    return banks, manifest


def history_for(video, index, banks, length):
    records, arrays = banks[video]["records"], banks[video]["arrays"]
    row = records[index]
    identity, frame = row["identity"], int(row["frame"])
    if identity < 0:
        return np.zeros((length, 384), np.float32), np.zeros(length, bool)
    candidates = banks[video]["by_identity"].get(int(identity), [])
    frames = [int(records[j]["frame"]) for j in candidates]
    candidates = candidates[:bisect.bisect_left(frames, frame)][-length:]
    values = np.zeros((length, 384), np.float32)
    valid = np.zeros(length, bool)
    if candidates:
        values[-len(candidates):] = np.asarray(arrays["cls"][candidates], np.float32)
        valid[-len(candidates):] = True
    return values, valid


def batch(observations, banks, labels, mode, device, history_length):
    current, histories, history_valid, visibility, ids, frames, videos = [], [], [], [], [], [], []
    video_numbers = {video: i for i, video in enumerate(sorted(banks))}
    for video, index in observations:
        row, arrays = banks[video]["records"][index], banks[video]["arrays"]
        current.append(np.asarray(arrays["cls"][index], np.float32))
        h, hv = history_for(video, index, banks, history_length)
        histories.append(h); history_valid.append(hv)
        visibility.append(np.asarray(arrays["visibility4"][index], np.float32))
        ids.append(labels[(video, row["identity"])])
        frames.append(int(row["frame"])); videos.append(video_numbers[video])
    tensors = [torch.from_numpy(np.asarray(value)).to(device) for value in
               (current, histories, history_valid, visibility, ids, frames, videos)]
    tensors[2] = tensors[2].bool(); tensors[4] = tensors[4].long(); tensors[5] = tensors[5].long(); tensors[6] = tensors[6].long()
    return tensors


def evaluate(model, video, banks, device, length):
    records, arrays = banks[video]["records"], banks[video]["arrays"]
    embeddings = np.zeros((len(records), 384), np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 128):
            end = min(start + 128, len(records))
            current = torch.from_numpy(np.asarray(arrays["cls"][start:end], np.float32)).to(device)
            histories = np.asarray([history_for(video, i, banks, length)[0] for i in range(start, end)], np.float32)
            valid = np.asarray([history_for(video, i, banks, length)[1] for i in range(start, end)], bool)
            visibility = torch.from_numpy(np.asarray(arrays["visibility4"][start:end], np.float32)).to(device)
            output = model(current, torch.from_numpy(histories).to(device), torch.from_numpy(valid).to(device), visibility)
            embeddings[start:end] = output["embedding"].cpu().numpy()
    return embedding_crop_retrieval(embeddings, records, np.asarray(arrays["valid"], bool), device="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=FDTAFishAdapter.MODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (config["state"] != "FROZEN" or args.seed not in config["seed_plan"] or args.mode not in config["modes"]
            or args.output.exists() or args.output.resolve().is_relative_to(Path.cwd())):
        raise ValueError("Registered new outside-Git output, mode and seed required")
    if shutil.disk_usage(args.output.parent).free < 40 * 2**30:
        raise OSError("40GiB reserve")
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    banks, source_manifest = load_banks(args.features)
    cycle = config["videos"]
    if config["outer"] not in cycle or config["inner"] not in cycle:
        raise ValueError("Declared held-out videos missing")
    fit = list(config["fit_videos"])
    if set(fit) & {config["outer"], config["inner"]} or len(fit) != 4:
        raise ValueError("Disjoint four-video fit partition required")
    aliases = {}
    for video, bank in banks.items():
        aliases[video] = {"records": bank["records"], "arrays": {
            "imagenet_global": bank["arrays"]["cls"], "imagenet_local4": bank["arrays"]["local4"],
            "visibility4": bank["arrays"]["visibility4"], "valid": bank["arrays"]["valid"]}}
    sampler = SameFrameTemporalSampler(aliases, fit, args.seed, config["identities_per_frame"],
                                       config["observations_per_identity"], config["minimum_past_gap"])
    model = FDTAFishAdapter(len(sampler.labels), config["dimension"], config["hidden"], args.mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    sources = [__file__, "cfp/fdta_fish_adapter.py", "cfp/local_identity_training.py", "cfp/reid.py",
               "cfp/crop_retrieval.py", str(args.config)]
    guard = dict(mode=args.mode, seed=args.seed, fit=fit, inner=config["inner"], outer=config["outer"],
                 config_sha256=sha256(args.config), feature_manifest_sha256=sha256(args.features / "manifest.json"),
                 source_sha256={path: sha256(path) for path in sources},
                 torch=torch.__version__, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                 online_history=False)
    report = dict(status="RUNNING", guard=guard, source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  config=config, parameters=sum(p.numel() for p in model.parameters()), epochs=[], evaluations={},
                  source_manifest_status=source_manifest["status"], scope=config["scope"])
    args.output.mkdir(parents=True)
    write_json(args.output / "manifest.json", report)
    for epoch in range(1, config["epochs"] + 1):
        model.train(); losses = []; digest = hashlib.sha256(); start = time.monotonic()
        for step in range(config["steps_per_epoch"]):
            observations = sampler.sample()
            current, histories, history_valid, visibility, ids, frames, videos = batch(observations, banks, sampler.labels, args.mode, "cpu", config["history_length"])
            digest.update(json.dumps([(v, int(i)) for v, i in observations], separators=(",", ":")).encode())
            output = model(current, histories, history_valid, visibility)
            loss, parts = local_identity_objective(output, ids, frames, videos, config["minimum_past_gap"], config["triplet_weight"], config["same_frame_weight"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite FDTA adapter loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite FDTA adapter gradient")
            if epoch == 1 and step == 0:
                report["first_gradient_norms"] = {n: float(p.grad.double().norm()) for n, p in model.named_parameters() if p.grad is not None}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        report["epochs"].append(dict(epoch=epoch, attempted_steps=config["steps_per_epoch"], optimizer_updates=config["steps_per_epoch"],
                                      mean_loss=float(np.mean(losses)), sampled_input_sha256=digest.hexdigest(), seconds=time.monotonic()-start))
        write_json(args.output / "manifest.json", report)
    for video in (config["inner"], config["outer"]):
        report["evaluations"][video] = evaluate(model, video, banks, "cpu", config["history_length"])
        write_json(args.output / "manifest.json", report)
    torch.save({"model": model.state_dict(), "guard": guard, "epoch": config["epochs"]}, args.output / "final.pt")
    report.update(status="COMPLETE", checkpoint_sha256=sha256(args.output / "final.pt"), online_history_used=False)
    write_json(args.output / "manifest.json", report)
    print(json.dumps({"status": report["status"], "mode": args.mode, "seed": args.seed, "outer": report["evaluations"][config["outer"]]}))


if __name__ == "__main__":
    main()
