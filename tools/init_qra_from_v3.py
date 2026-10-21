#!/usr/bin/env python3
"""Initialise a QRA fusion checkpoint from a validated V3 checkpoint.

Only keys with identical names and shapes are copied.  New QRA layers retain
their zero/near-zero initialization, and a machine-readable load report is
written next to the output checkpoint.  Large checkpoints stay outside Git.
"""

import argparse
import hashlib
import json
import os
import sys

import torch


def _state_dict(checkpoint):
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a model state dict")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exp-file", default="exps/yolox_mft26ir_rgbt_qra.py")
    parser.add_argument("--fusion-mode", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from yolox.exp import get_exp

    exp = get_exp(args.exp_file, None)
    if args.fusion_mode is not None:
        exp.fusion_mode = args.fusion_mode
    model = exp.get_model().to(torch.device(args.device))
    checkpoint = torch.load(args.source, map_location="cpu")
    source_state = _state_dict(checkpoint)
    target_state = model.state_dict()
    loaded, skipped_source = [], []
    for key, value in source_state.items():
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            target_state[key] = value.to(dtype=target_state[key].dtype)
            loaded.append(key)
        else:
            skipped_source.append(key)
    missing = [key for key in model.state_dict() if key not in loaded]
    model.load_state_dict(target_state, strict=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"model": model.state_dict(), "start_epoch": 0}, args.output)
    report = {
        "source": os.path.abspath(args.source),
        "source_sha256": sha256_file(args.source),
        "output": os.path.abspath(args.output),
        "output_sha256": sha256_file(args.output),
        "exp_file": args.exp_file,
        "fusion_mode": exp.fusion_mode,
        "loaded_key_count": len(loaded),
        "loaded_keys": loaded,
        "missing_target_keys": missing,
        "unexpected_source_keys": skipped_source,
        "initialization_rule": "V3 matching keys copied; QRA correction heads remain zero initialized",
    }
    report_path = args.report or args.output + ".init_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

