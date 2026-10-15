#!/usr/bin/env python3
"""Fixed CPU MLP audit for causal native motion residual predictability.

This follows the linear residual audit with one fixed, no-search nonlinear
predictor.  It is still retrospective and read-only: prediction parity is
verified before physical labels are loaded, and no tracker output or model
checkpoint is produced.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from cfp.provenance import sha256, snapshot_sources, write_json

sys.path.insert(0, str(Path(__file__).parent))
import cfp_diagnose_motion_error_predictability as causal  # noqa: E402


EPOCHS = 40
BATCH_SIZE = 1024
SEED = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_WIDTH = 32


def _fit(rows):
    features = np.stack([row["feature"][1:] for row in rows]).astype(np.float32)
    targets = np.stack([row["target_residual"] for row in rows]).astype(np.float32)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    inputs = torch.from_numpy((features - mean) / scale)
    target = torch.from_numpy(targets)
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], HIDDEN_WIDTH),
        torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
        torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN_WIDTH, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        order = torch.randperm(len(inputs), generator=generator)
        for start in range(0, len(inputs), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            loss = ((model(inputs[indices]) - target[indices]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    state = b"".join(model.state_dict()[key].detach().numpy().tobytes() for key in sorted(model.state_dict()))
    return model, mean, scale, hashlib.sha256(state).hexdigest()


def _score(model, mean, scale, rows):
    features = np.stack([row["feature"][1:] for row in rows]).astype(np.float32)
    with torch.no_grad():
        residual = model(torch.from_numpy((features - mean) / scale)).numpy()
    values = []
    for row, correction in zip(rows, residual):
        corrected_error = float(
            np.linalg.norm(row["target_residual"] - correction)
            * row["last_diag"]
            / row["target_diag"]
        )
        values.append((row["kf_error"], corrected_error, row["cv_error"], row["gap"]))
    values = np.asarray(values, dtype=np.float64)
    long_gap = values[:, 3] > 1
    return {
        "n": int(len(values)),
        "kf_mean": float(values[:, 0].mean()),
        "corrected_mean": float(values[:, 1].mean()),
        "cv_mean": float(values[:, 2].mean()),
        "kf_median": float(np.median(values[:, 0])),
        "corrected_median": float(np.median(values[:, 1])),
        "cv_median": float(np.median(values[:, 2])),
        "corrected_minus_kf_mean": float(values[:, 1].mean() - values[:, 0].mean()),
        "gap_gt1_n": int(long_gap.sum()),
        "gap_gt1_kf_mean": float(values[long_gap, 0].mean()) if long_gap.any() else None,
        "gap_gt1_corrected_mean": float(values[long_gap, 1].mean()) if long_gap.any() else None,
        "gap_gt1_cv_mean": float(values[long_gap, 2].mean()) if long_gap.any() else None,
    }


def _verify_observation(observation):
    manifest_path = observation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE_GT_FREE_NATIVE_MOTION_OBSERVATION":
        raise ValueError("the six-video GT-free native observation is required")
    if not manifest.get("all_six_native_prediction_bytes_equal"):
        raise ValueError("all six prediction byte parities must be verified first")
    if manifest.get("annotation_labels_loaded_online"):
        raise ValueError("online annotation labels would invalidate this diagnostic")
    for video in causal.VIDEOS:
        entry = manifest["videos"][video]
        if not entry.get("baseline_byte_equal"):
            raise ValueError(f"prediction parity missing for {video}")
        if sha256(observation / f"{video}.txt") != entry["prediction_sha256"]:
            raise ValueError(f"prediction drift for {video}")
        if sha256(observation / f"{video}.motion.jsonl.gz") != entry["motion_trace_sha256"]:
            raise ValueError(f"trace drift for {video}")
    return manifest, manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("CPU-only diagnostic must not use a GPU")
    if args.output.exists():
        raise ValueError("refuse to overwrite a diagnostic report")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest, manifest_path = _verify_observation(args.observation)
    from cfp.recovery_supervision import physical_bank

    rows = {}
    report = {
        "status": "COMPLETE_CAUSAL_NONLINEAR_MOTION_ERROR_PREDICTABILITY_AUDIT",
        "observation": str(args.observation),
        "observation_manifest_sha256": sha256(manifest_path),
        "fit_videos": causal.FIT_VIDEOS,
        "inner_video": causal.INNER_VIDEO,
        "outer_video": causal.OUTER_VIDEO,
        "model": {
            "architecture": "Linear(10,32)-Tanh-Linear(32,32)-Tanh-Linear(32,2)",
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "device": "cpu",
            "feature_schema": causal.FEATURE_SCHEMA[1:],
        },
        "scope": (
            "Read-only fixed nonlinear residual audit after six GT-free native prediction parities. "
            "The MLP uses only the causal features from the linear audit. Physical IDs and current "
            "targets are loaded only after parity checks and never enter a tracker. No checkpoint, "
            "native replay, feature bank or official prediction is written."
        ),
        "sources": snapshot_sources(
            args.output.with_suffix(".sources"),
            [__file__, "tools/cfp_diagnose_motion_error_predictability.py", "cfp/recovery_supervision.py", "cfp/joint_set_supervision.py"],
        ),
        "videos": {},
    }
    for video in causal.VIDEOS:
        rows[video], counts = causal._rows_for_video(args.observation, video, physical_bank)
        if not rows[video]:
            raise ValueError(f"no supported rows for {video}")
        report["videos"][video] = {
            "role": "fit" if video in causal.FIT_VIDEOS else "inner" if video == causal.INNER_VIDEO else "outer",
            "supported_rows": len(rows[video]),
            "counts": counts,
            "gap_gt1_rows": int(sum(row["gap"] > 1 for row in rows[video])),
        }

    fit_rows = sum((rows[video] for video in causal.FIT_VIDEOS), [])
    model, mean, scale, state_hash = _fit(fit_rows)
    report["model"]["state_sha256"] = state_hash
    report["model"]["fit_rows"] = len(fit_rows)
    report["model"]["normalization_mean"] = mean.tolist()
    report["model"]["normalization_scale"] = scale.tolist()
    report["fit_score"] = _score(model, mean, scale, fit_rows)
    report["inner_score"] = _score(model, mean, scale, rows[causal.INNER_VIDEO])
    report["outer_score"] = _score(model, mean, scale, rows[causal.OUTER_VIDEO])
    report["leave_one_fit_video_out"] = {}
    for held_out in causal.FIT_VIDEOS:
        train_rows = sum((rows[video] for video in causal.FIT_VIDEOS if video != held_out), [])
        loo_model, loo_mean, loo_scale, loo_hash = _fit(train_rows)
        report["leave_one_fit_video_out"][held_out] = {
            "train_rows": len(train_rows),
            "model_state_sha256": loo_hash,
            "score": _score(loo_model, loo_mean, loo_scale, rows[held_out]),
        }
    inner = report["inner_score"]
    outer = report["outer_score"]
    report["decision_rule"] = (
        "A learned-motion native replay would require corrected_mean < kf_mean on both held-out "
        "inner and outer videos, with no worsening of the outer gap>1 subset."
    )
    report["decision"] = (
        "JUSTIFY_ONE_CONTROLLED_LEARNED_MOTION_REPLAY"
        if (
            inner["corrected_mean"] < inner["kf_mean"]
            and outer["corrected_mean"] < outer["kf_mean"]
            and outer["gap_gt1_corrected_mean"] <= outer["gap_gt1_kf_mean"]
        )
        else "DO_NOT_RUN_LEARNED_MOTION_REPLAY"
    )
    report["interpretation"] = (
        "The fixed causal MLP worsens both held-out mean errors and the outer long-gap tail; "
        "leave-one-video-out outcomes are mixed. The replay gate therefore fails and no learned "
        "nonlinear-motion native run or final-combination claim is justified."
    )
    write_json(args.output, report)
    print(json.dumps({"decision": report["decision"], "inner": inner, "outer": outer}, sort_keys=True))


if __name__ == "__main__":
    main()
