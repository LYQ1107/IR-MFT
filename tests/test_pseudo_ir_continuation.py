"""Protect the scientific budget and resume state against silent protocol changes."""
from pathlib import Path

import pytest

from tools.pseudo_ir_continuation import checkpoint_training_budget, check_resume, joint_lr, load_plan, plateau_decision


def example():
    plan, sha = load_plan(Path(__file__).resolve().parents[1] / "configs/pseudo_ir_budget_extension_20261121.json")
    previous = dict(variant="rgb", seed=0, train_size=[480, 864], batch_size=8, accumulate=1,
                    calibration_epochs=0, finetune_epochs=3, lr=0.0002,
                    calibration_lr=0.001, ir_stats=None, source_sha256=plan["initial_visible_sha256"],
                    train_frames=27251, train_sequences=["BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015"],
                    ir_means=[0.0651,0.0262,0.0581], ir_std=[0.1916,0.1015,0.1084],
                    rgb_means=[0.485,0.456,0.406], rgb_std=[0.229,0.224,0.225],
                    bn_policy="frozen_all_running_statistics")
    current = dict(previous, finetune_epochs=24)
    checkpoint = dict(completed_epochs=3, history=[dict(epoch=i+1, frames=27251, phase="V3_finetune",
        lr=joint_lr(plan, i, 8)) for i in range(3)])
    return plan, sha, previous, current, checkpoint


def test_short_lr_prefix_is_preserved_and_extension_restarts():
    plan, _, _, _, _ = example()
    assert [joint_lr(plan, i, 8) for i in range(3)] == pytest.approx([0.000025, 0.00002, 0.00001])
    assert joint_lr(plan, 3, 8) == pytest.approx(0.000025)
    assert joint_lr(plan, 12, 8) == pytest.approx(0.00001)
    assert joint_lr(plan, 18, 8) == pytest.approx(0.000005)
    with pytest.raises(ValueError, match="outside"):
        joint_lr(plan, 24, 8)


def test_epoch_change_requires_explicit_plan_and_complete_prefix():
    plan, sha, prior, current, checkpoint = example()
    with pytest.raises(ValueError, match="finetune_epochs"):
        check_resume(prior, current, checkpoint)
    check_resume(prior, current, checkpoint, plan, sha)
    checkpoint["history"][1]["frames"] -= 1
    with pytest.raises(ValueError, match="coverage"):
        check_resume(prior, current, checkpoint, plan, sha)


@pytest.mark.parametrize("change, error", [("batch_size", "batch_size"), ("seed", "seed"),
                                           ("source_sha256", "visible"), ("lr_history", "LR history")])
def test_resume_rejects_changed_scientific_controls(change, error):
    plan, sha, prior, current, checkpoint = example()
    if change == "source_sha256":
        prior[change] = "wrong source"
    elif change == "lr_history":
        checkpoint["history"][0]["lr"] *= 8
    else:
        current[change] += 1
    with pytest.raises(ValueError, match=error):
        check_resume(prior, current, checkpoint, plan, sha)


def test_all_arms_extend_when_one_dev_curve_still_improves():
    plan, _, _, _, _ = example()
    rows = {arm: [dict(joint_epochs=i, HOTA=40.0) for i in [3,6,9,12]] for arm in plan["arms"]}
    assert plateau_decision(plan, rows, 12)["stop"]
    rows["new_diffv2ir"][-1]["HOTA"] = 40.3
    decision = plateau_decision(plan, rows, 12)
    assert not decision["stop"] and decision["next_budget"] == 18
    rows["old_legacy"].pop()
    with pytest.raises(ValueError, match="coverage"):
        plateau_decision(plan, rows, 12)


def test_maximum_budget_is_reported_separately_from_plateau():
    plan, _, _, _, _ = example()
    rows = {arm: [dict(joint_epochs=i, HOTA=30+i) for i in range(3,25,3)] for arm in plan["arms"]}
    decision = plateau_decision(plan, rows, 24)
    assert decision["stop"] and decision["capped"] and not decision["plateau_rule_met"]


def test_resume_rejects_changed_normalization_even_when_stats_path_is_unchanged():
    plan, sha, prior, current, checkpoint = example()
    current["ir_means"] = [0.3,0.3,0.3]
    with pytest.raises(ValueError, match="ir_means"):
        check_resume(prior, current, checkpoint, plan, sha)


def test_evaluation_reports_completed_budget_instead_of_inference_defaults():
    _, _, _, manifest, _ = example()
    full = dict(manifest=manifest, completed_epochs=12, history=[{}]*12)
    info = checkpoint_training_budget(full)
    assert info["joint_epochs"] == 12 and info["configured_maximum_joint_epochs"] == 24
    assert info["calibration_epochs"] == 0
    manifest = dict(manifest, variant="legacy", calibration_epochs=3)
    compact = dict(manifest=manifest, selection={"joint_epochs":6, "continuation_plan_sha256":"frozen"})
    info = checkpoint_training_budget(compact)
    assert info["joint_epochs"] == 6 and info["completed_adaptation_epochs"] == 9
    assert info["continuation_plan_sha256"] == "frozen"
    assert checkpoint_training_budget({"model":{}}) is None
