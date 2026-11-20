"""CPU-only protocol checks for extending a completed matched training prefix."""
import hashlib
import json
import math
from pathlib import Path


RESUME_KEYS = ["variant", "seed", "train_size", "batch_size", "accumulate",
               "calibration_epochs", "finetune_epochs", "lr", "calibration_lr", "ir_stats",
               "train_frames", "train_sequences", "ir_means", "ir_std", "rgb_means", "rgb_std", "bn_policy"]


def load_plan(path):
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    schedule = plan["training"]
    boundary = 0
    for segment in schedule["lr_segments"]:
        if (segment["start_joint_index"] != boundary or
                segment["end_joint_index_exclusive"] <= boundary or
                segment["reference_lr_batch64"] <= 0):
            raise ValueError("Continuation LR segments must be positive and contiguous")
        boundary = segment["end_joint_index_exclusive"]
    if boundary != schedule["maximum_joint_epochs"]:
        raise ValueError("LR schedule does not cover the maximum joint budget")
    if set(plan["allowed_physical_gpus"]) != {2, 3}:
        raise ValueError("Continuation plan must restrict physical GPUs to 2 and 3")
    return plan, hashlib.sha256(raw).hexdigest()


def joint_lr(plan, joint_index, effective_batch):
    for segment in plan["training"]["lr_segments"]:
        start, end = segment["start_joint_index"], segment["end_joint_index_exclusive"]
        if start <= joint_index < end:
            factor = 0.2 + 0.8 * (1 + math.cos(math.pi * (joint_index - start) / (end - start))) / 2
            return segment["reference_lr_batch64"] * effective_batch / 64.0 * factor
    raise ValueError(f"Joint epoch outside the continuation schedule: {joint_index}")


def checkpoint_training_budget(checkpoint):
    """Read completed adaptation epochs, rather than inference CLI defaults."""
    manifest = checkpoint.get("manifest")
    required = {"variant", "calibration_epochs", "finetune_epochs", "source_sha256"}
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        return None
    calibration = 0 if manifest["variant"] == "rgb" else manifest["calibration_epochs"]
    selection = checkpoint.get("selection", {})
    if "completed_epochs" in checkpoint:
        completed = checkpoint["completed_epochs"]
        joint = completed - calibration
        if len(checkpoint["history"]) != completed:
            raise ValueError("Checkpoint epoch counter disagrees with its training history")
    elif "joint_epochs" in selection:
        joint = selection["joint_epochs"]
        completed = calibration + joint
    else:
        return None
    if joint < 0:
        raise ValueError("Checkpoint has fewer epochs than its calibration prefix")
    return dict(calibration_epochs=calibration, joint_epochs=joint,
                completed_adaptation_epochs=completed,
                configured_maximum_joint_epochs=manifest["finetune_epochs"],
                initial_visible_sha256=manifest["source_sha256"],
                continuation_plan_sha256=manifest.get("continuation", {}).get(
                    "plan_sha256", selection.get("continuation_plan_sha256")))


def check_resume(prior, current, checkpoint, plan=None, plan_sha=None):
    """Only an explicit frozen continuation may change the joint epoch budget."""
    for key in RESUME_KEYS:
        if plan is not None and key == "finetune_epochs":
            allowed = {plan["training"]["prefix_joint_epochs"], plan["training"]["maximum_joint_epochs"]}
            if prior[key] not in allowed or current[key] != max(allowed):
                raise ValueError("Unexpected continuation epoch budget")
        elif prior[key] != current[key]:
            raise ValueError(f"Resume configuration changed: {key}")
    if plan is None:
        return
    schedule = plan["training"]
    arm = next((a for a in plan["arms"].values() if a["variant"] == current["variant"]), None)
    if arm is None:
        raise ValueError("Variant is outside the frozen continuation arms")
    expected = dict(seed=schedule["seed"], train_size=schedule["train_size"],
                    batch_size=schedule["batch_size"], accumulate=schedule["accumulate"],
                    calibration_epochs=arm["calibration_epochs"], ir_stats=arm["ir_stats"])
    for key, value in expected.items():
        if current[key] != value:
            raise ValueError(f"Continuation protocol mismatch: {key}")
    if prior["source_sha256"] != plan["initial_visible_sha256"]:
        raise ValueError("Continuation did not originate at the common visible checkpoint")
    if prior.get("continuation", {}).get("plan_sha256", plan_sha) != plan_sha:
        raise ValueError("Continuation plan changed after training began")
    cal = arm["calibration_epochs"]
    completed = checkpoint["completed_epochs"]
    history = checkpoint["history"]
    if completed != len(history) or completed < cal + schedule["prefix_joint_epochs"]:
        raise ValueError("Continuation requires the full completed training prefix")
    if prior["finetune_epochs"] == schedule["prefix_joint_epochs"]:
        if completed != cal + schedule["prefix_joint_epochs"]:
            raise ValueError("Short-budget prefix is incomplete or has extra epochs")
    elif prior.get("continuation", {}).get("plan_sha256") != plan_sha:
        raise ValueError("Extended checkpoint lacks its frozen continuation plan")
    for index, row in enumerate(history):
        if row["epoch"] != index + 1 or row["frames"] != schedule["train_frames_per_epoch"]:
            raise ValueError("Checkpoint training history has incomplete epoch/frame coverage")
        calibrating = index < cal
        if row["phase"] != ("calibration" if calibrating else "V3_finetune"):
            raise ValueError("Checkpoint training phase sequence changed")
        if calibrating:
            expected_lr = prior["calibration_lr"] * current["batch_size"] * current["accumulate"] / 64.0 * (
                0.2 + 0.8 * (1 + math.cos(math.pi * index / cal)) / 2)
        else:
            expected_lr = joint_lr(plan, index - cal, current["batch_size"] * current["accumulate"])
        if not math.isclose(row["lr"], expected_lr, rel_tol=1e-10):
            raise ValueError(f"Checkpoint LR history differs from plan at epoch {index + 1}")


def plateau_decision(plan, dev_history, budget):
    """Use Dev only; all arms share one stopping budget."""
    rule = plan["stopping"]
    training = plan["training"]
    if budget not in training["decision_joint_epochs"]:
        raise ValueError("Stopping was requested outside a planned decision budget")
    gains = {}
    for arm in plan["arms"]:
        rows = dev_history[arm]
        needed = set(range(training["prefix_joint_epochs"], budget + 1,
                           training["dev_interval_joint_epochs"]))
        if {row["joint_epochs"] for row in rows} != needed:
            raise ValueError(f"Dev checkpoint coverage is incomplete: {arm}")
        previous = max(row["HOTA"] for row in rows
                       if row["joint_epochs"] <= budget - rule["lookback_joint_epochs"])
        current = max(row["HOTA"] for row in rows)
        gains[arm] = current - previous
    plateau = all(gain <= rule["epsilon_pp"] + 1e-12 for gain in gains.values())
    capped = budget >= training["maximum_joint_epochs"]
    return dict(joint_budget=budget, gains_over_last_six_joint_epochs_pp=gains,
                plateau_rule_met=plateau, capped=capped, stop=plateau or capped,
                next_budget=None if plateau or capped else budget + training["extension_step_joint_epochs"],
                reason="Dev plateau" if plateau else ("maximum budget reached" if capped else "Dev still improving"))
