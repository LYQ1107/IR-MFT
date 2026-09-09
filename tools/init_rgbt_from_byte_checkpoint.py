#!/usr/bin/env python3
"""Duplicate a visible-only YOLOX checkpoint into both RGB and IR branches."""

import argparse
import os

import torch

from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/yolox_mft26ir_rgbt_v1.py",
        help="RGBT experiment file",
    )
    parser.add_argument("--source", required=True, help="visible-only checkpoint")
    parser.add_argument(
        "--output",
        default="checkpoints/mft26ir_rgbt_v1_init.pth.tar",
        help="output checkpoint",
    )
    return parser


def build_mapping(source_state, target_state):
    mapped = {}
    loaded = {"rgb": 0, "ir": 0, "head": 0}
    missing = []
    unexpected = []

    for target_key, target_value in target_state.items():
        if target_key.startswith("backbone.rgb_branch."):
            source_key = "backbone." + target_key[len("backbone.rgb_branch."):]
            group = "rgb"
        elif target_key.startswith("backbone.ir_branch."):
            source_key = "backbone." + target_key[len("backbone.ir_branch."):]
            group = "ir"
        else:
            source_key = target_key
            group = "head"

        source_value = source_state.get(source_key)
        if source_value is None:
            missing.append((target_key, source_key))
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise ValueError(
                f"Shape mismatch for {target_key}: source {tuple(source_value.shape)} "
                f"vs target {tuple(target_value.shape)}"
            )
        mapped[target_key] = source_value
        loaded[group] += 1

    source_keys_used = set()
    for target_key in target_state:
        if target_key.startswith("backbone.rgb_branch."):
            source_keys_used.add("backbone." + target_key[len("backbone.rgb_branch."):])
        elif target_key.startswith("backbone.ir_branch."):
            source_keys_used.add("backbone." + target_key[len("backbone.ir_branch."):])
        else:
            source_keys_used.add(target_key)
    unexpected = sorted(set(source_state) - source_keys_used)
    if missing:
        raise KeyError(f"Missing target mappings ({len(missing)}): {missing[:10]}")
    return mapped, loaded, unexpected


def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, None)
    model = exp.get_model()
    target_state = model.state_dict()

    checkpoint = torch.load(args.source, map_location="cpu")
    source_state = checkpoint.get("model", checkpoint)
    if not isinstance(source_state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")

    print("source checkpoint keys:", len(source_state))
    print("source first keys:", list(source_state)[:8])
    print("target model state_dict keys:", len(target_state))
    print("target first keys:", list(target_state)[:8])

    mapped, loaded, unexpected = build_mapping(source_state, target_state)
    model.load_state_dict(mapped, strict=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"start_epoch": 0, "model": model.state_dict()}, args.output)

    print("loaded rgb params:", loaded["rgb"])
    print("loaded ir params:", loaded["ir"])
    print("loaded head params:", loaded["head"])
    print("missing keys: 0")
    print("unexpected source keys:", len(unexpected), unexpected[:10])
    print("saved:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
