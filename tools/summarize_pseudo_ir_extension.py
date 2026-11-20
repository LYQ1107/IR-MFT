#!/usr/bin/env python3
"""Audit the Dev-driven budget experiment and export learning curves and Test deltas."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from pseudo_ir_continuation import check_resume, load_plan, plateau_decision
from summarize_pseudo_ir_comparison import METRICS, audit_detection_coverage, detailed_metrics


TITLES = {"old_legacy":"V3 + old legacy IR", "new_diffv2ir":"V3 + new DiffV2IR",
          "rgb_matched":"RGB matched fine-tune + ByteTrack", "rgb_original":"Original RGB ByteTrack"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def evaluation(path, arm, split, variant, source_sha, normalization, short_plan):
    raw = read(path)
    protocol = raw["protocol"]
    expected = dict(split=split, variant=variant, frames=short_plan["frames"][split],
                    max_frames=0, skip_evaluation=False, source_sha256=source_sha,
                    input_size=short_plan["evaluation"]["size_hw"], batch_size=4)
    expected.update({key:short_plan["evaluation"][key]
                     for key in ["conf","nms","track_thresh","track_buffer","match_thresh"]})
    if normalization:
        expected.update({key:normalization[key] for key in ["ir_means", "ir_std"]})
    for key, value in expected.items():
        if protocol[key] != value:
            raise ValueError(f"Evaluation protocol changed: {path}: {key}")
    if set(protocol["sequences"]) != set(short_plan[split + "_sequences"]):
        raise ValueError(f"Incomplete sequence coverage: {path}")
    if str(protocol["physical_gpu"]) not in {"2", "3"}:
        raise ValueError(f"Unapproved GPU: {path}")
    metrics = {key:raw["trackeval"][key] for key in METRICS}
    sequences = detailed_metrics(path.parent / "trackeval_output/pedestrian_detailed.csv", protocol["sequences"], metrics)
    coverage = audit_detection_coverage(path.parent, sequences)
    if sum(row["frames"] for row in coverage.values()) != protocol["frames"]:
        raise ValueError(f"Per-frame coverage disagrees with the aggregate: {path}")
    return dict(arm=arm, method=TITLES[arm], split=split, metrics=metrics, per_sequence=sequences,
                detection_coverage=coverage, trackeval_all_metrics=raw["trackeval"], protocol=protocol,
                metrics_path=str(path), metrics_sha256=digest(path), runtime=raw["runtime"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--curve-png", type=Path)
    parser.add_argument("--curve-csv", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    plan, plan_sha = load_plan(args.plan)
    root, short = Path(plan["result_root"]), Path(plan["short_result_root"])
    repo = Path(__file__).resolve().parents[1]
    short_plan = read(repo / plan["short_plan"])
    state = read(root / "campaign_state.json")
    if state["plan_sha256"] != plan_sha:
        raise ValueError("Campaign has a different frozen plan")
    output = dict(status="COMPLETE" if state["status"] == "COMPLETE" else "INCOMPLETE", plan=plan,
                  plan_sha256=plan_sha, campaign_state=state, training={}, dev_checkpoints=[],
                  selected_test=[], original_rgb_reference=[], deltas={}, per_sequence_deltas={}, missing=[])
    curves = []
    for arm, config in plan["arms"].items():
        if arm not in state["arms"]:
            output["missing"].append("campaign initialization: " + arm)
            continue
        record = state["arms"][arm]
        folder = root / ("train_" + arm)
        paths = [folder / "train_manifest.json", folder / "train_history.json"]
        if not all(path.exists() for path in paths):
            folder = short / config["prefix_directory"]
            paths = [folder / "train_manifest.json", folder / "train_history.json"]
        manifest, history = map(read, paths)
        current = dict(manifest, finetune_epochs=plan["training"]["maximum_joint_epochs"])
        check_resume(manifest, current, dict(history=history, completed_epochs=len(history)), plan, plan_sha)
        prefix = read(short / config["prefix_directory"] / "train_history.json")
        if history[:len(prefix)] != prefix:
            raise ValueError(f"Resumed training prefix changed: {arm}")
        if str(manifest["physical_gpu"]) not in {"2","3"} or any(
                str(event["physical_gpu"]) not in {"2","3"} for event in manifest.get("execution_events", [])):
            raise ValueError(f"Unapproved training GPU: {arm}")
        output["training"][arm] = dict(manifest=manifest, history=history,
            completed_joint_epochs=len(history)-config["calibration_epochs"],
            selected_joint_epochs=record["best"]["joint_epochs"], best_checkpoint=record["best"])
        for row in history:
            curves.append(dict(arm=arm, curve="training_loss", joint_epochs=row["epoch"]-config["calibration_epochs"],
                               value=row["loss"], phase=row["phase"], lr=row["lr"]))
        best_entry = max(record["dev_history"], key=lambda entry:(entry["HOTA"], -entry["joint_epochs"]))
        if (best_entry["joint_epochs"] != record["best"]["joint_epochs"]
                or best_entry["source_sha256"] != record["best"]["selected_full_checkpoint_sha256"]
                or best_entry["metrics_sha256"] != record["best"]["dev_metrics_sha256"]):
            raise ValueError(f"Best checkpoint was not selected by the fixed Dev rule: {arm}")
        for entry in record["dev_history"]:
            path = Path(entry["metrics_path"])
            if digest(path) != entry["metrics_sha256"]:
                raise ValueError(f"Dev evidence changed: {path}")
            result = evaluation(path, arm, "dev", config["variant"], entry["source_sha256"], manifest, short_plan)
            source_budget = result["protocol"].get("source_training")
            if entry["joint_epochs"] > 3 or source_budget is not None:
                if (not source_budget or source_budget["joint_epochs"] != entry["joint_epochs"]
                        or source_budget["calibration_epochs"] != config["calibration_epochs"]
                        or source_budget["initial_visible_sha256"] != plan["initial_visible_sha256"]):
                    raise ValueError(f"Dev source reports a different completed training budget: {path}")
            if result["metrics"]["HOTA"] != entry["HOTA"]:
                raise ValueError("Dev selection score disagrees with its source")
            result["joint_epochs"] = entry["joint_epochs"]
            result["selected"] = entry["joint_epochs"] == record["best"]["joint_epochs"]
            output["dev_checkpoints"].append(result)
            for metric in METRICS:
                curves.append(dict(arm=arm, curve="Dev_"+metric, joint_epochs=entry["joint_epochs"],
                                   value=result["metrics"][metric], phase="evaluation", lr=""))
        if "selected_test" not in record:
            output["missing"].append("final selected Test: " + arm)
            continue
        test = record["selected_test"]
        if digest(test["path"]) != test["sha256"] or digest(record["best"]["checkpoint"]) != record["best"]["sha256"]:
            raise ValueError(f"Selected model or Test evidence changed: {arm}")
        result = evaluation(Path(test["path"]), arm, "test", config["variant"], record["best"]["sha256"], manifest, short_plan)
        source_budget = result["protocol"].get("source_training")
        if (not source_budget or source_budget["joint_epochs"] != record["best"]["joint_epochs"]
                or source_budget["calibration_epochs"] != config["calibration_epochs"]
                or source_budget["initial_visible_sha256"] != plan["initial_visible_sha256"]):
            raise ValueError(f"Selected Test source reports a different completed training budget: {arm}")
        result["selected_joint_epochs"] = record["best"]["joint_epochs"]
        output["selected_test"].append(result)
    for split in ["dev","test"]:
        path = short / f"eval_rgb_original_{split}/metrics.json"
        output["original_rgb_reference"].append(evaluation(path, "rgb_original", split, "rgb",
            plan["initial_visible_sha256"], None, short_plan))
    for decision in state["decisions"]:
        budget = decision["joint_budget"]
        dev = {arm:[entry for entry in record["dev_history"] if entry["joint_epochs"] <= budget]
               for arm, record in state["arms"].items()}
        if plateau_decision(plan, dev, budget) != decision:
            raise ValueError("Stored stopping decision does not follow the frozen Dev rule")
    if output["status"] == "COMPLETE":
        if not state["decisions"] or not state["decisions"][-1]["stop"]:
            raise ValueError("Completed campaign lacks its stopping decision")
        budget = state["common_final_joint_budget"]
        if any(record["trained_joint"] != budget for record in state["arms"].values()):
            raise ValueError("Final training budgets differ between arms")
        for arm, training in output["training"].items():
            if training["completed_joint_epochs"] != budget:
                raise ValueError(f"Actual checkpoint epoch budget differs from campaign state: {arm}")
            latest = state["arms"][arm]["latest_completion"]
            if digest(latest["checkpoint"]) != latest["sha256"]:
                raise ValueError(f"Final resumable checkpoint changed: {arm}")
        cleanup = read(root / "train_byte_cache_cleanup.json")
        if not cleanup["removed"] or Path(cleanup["cache_root"]).exists():
            raise ValueError("Completed experiment left its temporary Train cache")
        output["train_byte_cache_cleanup"] = cleanup
    lookup = {row["arm"]:row for row in output["selected_test"]}
    lookup["rgb_original"] = next(row for row in output["original_rgb_reference"] if row["split"] == "test")
    if "new_diffv2ir" in lookup:
        new = lookup["new_diffv2ir"]
        for arm in ["rgb_original","old_legacy","rgb_matched"]:
            if arm not in lookup:
                continue
            control = lookup[arm]
            output["deltas"]["test_new_minus_"+arm] = {key:new["metrics"][key]-control["metrics"][key] for key in METRICS}
            output["per_sequence_deltas"]["test_new_minus_"+arm] = {
                seq:{key:new["per_sequence"][seq][key]-control["per_sequence"][seq][key] for key in METRICS}
                for seq in new["per_sequence"]}
    table = [f"Status: **{output['status']}**", "", "Dev-selected checkpoints; all adapted arms use a common maximum training budget.", "",
             "| Method | Split | Selected joint epoch | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = output["original_rgb_reference"] + [row for row in output["dev_checkpoints"] if row["selected"]] + output["selected_test"]
    for row in rows:
        table.append("| " + " | ".join([row["method"],row["split"],str(row.get("selected_joint_epochs",row.get("joint_epochs",0)))] +
            [f"{row['metrics'][key]:.3f}" for key in METRICS]) + " |")
    supplementary = ["DetRe", "DetPr", "LocA", "CLR_Re", "CLR_Pr", "CLR_FN", "CLR_FP", "Frag"]
    table += ["", "Supplementary official TrackEval components on the tracker outputs. HOTA detection ratios",
              "average IoU thresholds; CLEAR recall/precision use IoU0.5. These are not raw detector AP.", "",
              "| Method | Split | " + " | ".join(supplementary) + " |",
              "| --- | --- | " + " | ".join(["---:"]*len(supplementary)) + " |"]
    for row in rows:
        values = row["trackeval_all_metrics"]
        table.append("| " + " | ".join([row["method"], row["split"]] +
            [f"{values[key]:.3f}" if index < 5 else str(int(values[key]))
             for index, key in enumerate(supplementary)]) + " |")
    table += ["", "All Dev checkpoints (no Test scores used for selection):", "",
              "| Method | Joint epoch | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Selected |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in output["dev_checkpoints"]:
        table.append("| " + " | ".join([row["method"], str(row["joint_epochs"])] +
            [f"{row['metrics'][key]:.3f}" for key in METRICS] + [str(row["selected"])]) + " |")
    table += ["", "Individual Test sequences:", "",
              "| Method | Sequence | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in [lookup["rgb_original"]] + output["selected_test"]:
        for seq, metrics in row["per_sequence"].items():
            table.append("| " + " | ".join([row["method"],seq] + [f"{metrics[key]:.3f}" for key in METRICS]) + " |")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False)+"\n")
    args.output_markdown.write_text("\n".join(table)+"\n")
    if args.curve_csv:
        with args.curve_csv.open("w",newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["arm","curve","joint_epochs","value","phase","lr"])
            writer.writeheader()
            writer.writerows(curves)
    if args.curve_png and curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,3,figsize=(13,3.7), constrained_layout=True)
        for arm in plan["arms"]:
            for ax, curve, title in zip(axes,["training_loss","Dev_HOTA","Dev_IDF1"],
                                       ["Joint training loss","Dev HOTA (%)","Dev IDF1 (%)"]):
                points = [row for row in curves if row["arm"] == arm and row["curve"] == curve and row["joint_epochs"] > 0]
                ax.plot([row["joint_epochs"] for row in points],[row["value"] for row in points], marker="o",ms=3,label=TITLES[arm])
                ax.set(xlabel="Completed joint epochs", title=title)
                ax.grid(alpha=0.25)
        axes[1].legend(fontsize=7)
        fig.savefig(args.curve_png,dpi=170)
        plt.close(fig)
    print(json.dumps(dict(status=output["status"], dev_checkpoints=len(output["dev_checkpoints"]),
                         final_test=len(output["selected_test"]), deltas=output["deltas"], missing=output["missing"]),indent=2))
    if args.require_complete and (output["status"] != "COMPLETE" or output["missing"]):
        raise RuntimeError("Extended comparison is incomplete")


if __name__ == "__main__":
    main()
