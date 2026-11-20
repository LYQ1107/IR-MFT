#!/usr/bin/env python3
"""Run three equally extended arms on GPUs 2/3, selecting solely on Dev HOTA."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from pseudo_ir_continuation import load_plan, plateau_decision

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tools/run_pseudo_ir_comparison.py"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def alive(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def storage_guard(root, reserve, additional=0):
    free = shutil.disk_usage(root).free / 2**30
    if free < reserve + additional:
        raise RuntimeError(f"Local storage reserve threatened: {free:.2f}GiB free")
    return free


def select_best(root, arm, entry, checkpoint, plan_sha, reserve):
    """Inference already casts to FP16; retain only this equivalent EMA state."""
    import torch
    storage_guard(root, reserve, 0.6)
    checkpoint = Path(checkpoint)
    if digest(checkpoint) != entry["source_sha256"]:
        raise RuntimeError("Dev selection source changed before retaining the best model")
    full = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {key: tensor.half() if tensor.is_floating_point() else tensor
             for key, tensor in full["model"].items()}
    folder = root / ("train_" + arm)
    folder.mkdir(exist_ok=True)
    target = folder / "best_dev_ema_fp16.pth.tar"
    temporary = target.with_suffix(".tmp")
    metadata = dict(joint_epochs=entry["joint_epochs"], HOTA=entry["HOTA"],
                    selected_full_checkpoint_sha256=entry["source_sha256"],
                    dev_metrics=entry["metrics_path"], dev_metrics_sha256=entry["metrics_sha256"],
                    continuation_plan_sha256=plan_sha,
                    conversion="EMA cast to FP16, identical to the existing inference model.half()")
    torch.save(dict(model=state, manifest=full["manifest"], selection=metadata), temporary)
    os.replace(temporary, target)
    metadata.update(checkpoint=str(target), sha256=digest(target), bytes=target.stat().st_size)
    write(folder / "best_dev_selection.json", metadata)
    return metadata


class Campaign:
    def __init__(self, args):
        self.args = args
        self.plan, self.plan_sha = load_plan(args.plan)
        self.root = Path(self.plan["result_root"])
        self.short = Path(self.plan["short_result_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "campaign_state.json"
        self.reserve = self.plan["storage"]["local_reserve_gib"]
        self.state = read(self.state_path) if self.state_path.exists() else dict(
            status="WAITING_FOR_SHORT_CAMPAIGN", plan_sha256=self.plan_sha,
            target_joint_budget=self.plan["training"]["minimum_joint_epochs"],
            decisions=[], arms={}, active_jobs={}, started_unix=time.time())
        if self.state["plan_sha256"] != self.plan_sha:
            raise RuntimeError("Frozen continuation plan changed")
        self.processes = {}
        self.logs = {}
        self.last_notice = 0

    def save(self):
        self.state["updated_unix"] = time.time()
        write(self.state_path, self.state)

    def prerequisites(self):
        expected = [self.short / f"eval_{arm}_{split}/metrics.json"
                    for arm in ["rgb_original", "old_legacy", "new_diffv2ir", "rgb_matched"]
                    for split in ["dev", "test"]]
        expected += [self.short / f"reference_frozen_v3_{split}/metrics.json" for split in ["dev", "test"]]
        if not all(p.exists() for p in expected):
            return False
        for gpu in [2, 3]:
            status = read(self.short / f"queue_gpu{gpu}.state.json")
            if status["status"] == "FAILED":
                raise RuntimeError(f"Short prerequisite failed: {status}")
            if status["status"] != "COMPLETE":
                return False
        if not self.args.cache_manifest.is_file():
            return False
        cache = read(self.args.cache_manifest)
        if not (cache["complete"] and cache["frames"] == 27251 and cache["byte_identical"]
                and cache["split"] == "train" and Path(cache["cache_root"]).is_dir()
                and cache["bytes"] <= 10 * 2**30):
            raise RuntimeError("Invalid bounded Train byte cache")
        self.cache_root = cache["cache_root"]
        compact = {key: value for key, value in cache.items() if key != "files"}
        compact["manifest_sha256"] = digest(self.args.cache_manifest)
        write(self.root / "train_byte_cache_summary.json", compact)
        return True

    def initialize(self):
        storage_guard(self.root, self.reserve, 8)
        for arm, config in self.plan["arms"].items():
            if arm in self.state["arms"]:
                continue
            folder = self.short / config["prefix_directory"]
            completion = read(folder / "completion.json")
            manifest = read(folder / "train_manifest.json")
            expected_epochs = config["calibration_epochs"] + self.plan["training"]["prefix_joint_epochs"]
            if (completion["status"] != "COMPLETE" or completion["epochs"] != expected_epochs
                    or manifest["source_sha256"] != self.plan["initial_visible_sha256"]):
                raise RuntimeError(f"Invalid completed short prefix: {arm}")
            checkpoint = Path(completion["checkpoint"])
            if digest(checkpoint) != completion["sha256"]:
                raise RuntimeError(f"Short checkpoint changed: {arm}")
            entry = self.dev_entry(arm, 3, self.short / f"eval_{arm}_dev/metrics.json", completion["sha256"])
            best = select_best(self.root, arm, entry, checkpoint, self.plan_sha, self.reserve)
            self.state["arms"][arm] = dict(trained_joint=3, dev_history=[entry], best=best,
                                           resume_checkpoint=str(checkpoint), prefix_completion=completion,
                                           prefix_manifest=manifest)
            self.save()

    def dev_entry(self, arm, joint, path, expected_sha):
        metrics = read(path)
        protocol = metrics["protocol"]
        if (protocol["split"] != "dev" or protocol["frames"] != 4834
                or protocol["max_frames"] or protocol["skip_evaluation"]
                or protocol["source_sha256"] != expected_sha
                or protocol["variant"] != self.plan["arms"][arm]["variant"]):
            raise RuntimeError(f"Invalid Dev checkpoint evaluation: {path}")
        return dict(joint_epochs=joint, HOTA=metrics["trackeval"]["HOTA"],
                    metrics={key: metrics["trackeval"][key] for key in ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]},
                    source_sha256=expected_sha, metrics_path=str(path), metrics_sha256=digest(path))

    def command(self, gpu, arm, kind, joint):
        config = self.plan["arms"][arm]
        record = self.state["arms"][arm]
        common = ["--variant", config["variant"], "--reserve-gib", str(self.reserve)]
        if config["ir_stats"]:
            common += ["--ir-stats", config["ir_stats"]]
        if kind == "train":
            folder = self.root / ("train_" + arm)
            command = ["train", "--ckpt", record["resume_checkpoint"], "--resume",
                       "--batch-size", "8", "--accumulate", "1", "--workers", "4",
                       "--calibration-epochs", str(config["calibration_epochs"]),
                       "--finetune-epochs", str(self.plan["training"]["maximum_joint_epochs"]),
                       "--continuation-plan", str(self.args.plan.resolve()),
                       "--stop-after-joint-epochs", str(joint)]
        else:
            folder = self.root / (f"dev_{arm}_joint{joint:02d}" if kind == "dev" else f"test_{arm}_selected")
            checkpoint = record["resume_checkpoint"] if kind == "dev" else record["best"]["checkpoint"]
            command = ["infer", "--ckpt", checkpoint, "--split", "dev" if kind == "dev" else "test",
                       "--batch-size", "4", "--workers", "4"]
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-u", str(RUNNER)] + command + common + ["--output", str(folder)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        env.pop("MFT26IR_GENERATED_IR_ROOT", None)
        if kind == "train" and config["variant"] == "diffv2ir":
            env["MFT26IR_GENERATED_IR_ROOT"] = self.cache_root
        return folder, command, env

    def launch(self, gpu, arm, kind, joint):
        storage_guard(self.root, self.reserve, 3 if kind == "train" else 0.3)
        folder, command, env = self.command(gpu, arm, kind, joint)
        log_path = folder / (f"joint{joint:02d}.process.log" if kind == "train" else "process.log")
        log = log_path.open("a")
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        key = str(gpu)
        job = dict(arm=arm, kind=kind, joint_epochs=joint, pid=process.pid, physical_gpu=gpu,
                   folder=str(folder), log=str(log_path), command=command, started_unix=time.time())
        self.processes[key] = process
        self.logs[key] = log
        self.state["active_jobs"][key] = job
        self.save()
        print("JOB_STARTED", json.dumps(job), flush=True)

    def integrate(self, job):
        arm, kind, joint = job["arm"], job["kind"], job["joint_epochs"]
        record, folder = self.state["arms"][arm], Path(job["folder"])
        if kind == "train":
            completed = read(folder / "completion.json")
            if (completed["status"] != "STAGE_COMPLETE" or completed["joint_epochs"] != joint
                    or completed["continuation_plan_sha256"] != self.plan_sha):
                raise RuntimeError(f"Incomplete continuation stage: {job}")
            if digest(completed["checkpoint"]) != completed["sha256"]:
                raise RuntimeError("Continuation checkpoint hash changed")
            record.update(trained_joint=joint, resume_checkpoint=completed["checkpoint"], latest_completion=completed)
            write(folder / f"stage_joint{joint:02d}.json", dict(completed, job=job))
        elif kind == "dev":
            entry = self.dev_entry(arm, joint, folder / "metrics.json", record["latest_completion"]["sha256"])
            if any(row["joint_epochs"] == joint for row in record["dev_history"]):
                raise RuntimeError("Duplicate Dev checkpoint selection")
            if entry["HOTA"] > record["best"]["HOTA"]:
                record["best"] = select_best(self.root, arm, entry, record["resume_checkpoint"], self.plan_sha, self.reserve)
            record["dev_history"].append(entry)
            print("DEV_RESULT", json.dumps(dict(arm=arm, entry=entry, best_joint=record["best"]["joint_epochs"])), flush=True)
        else:
            path = folder / "metrics.json"
            metrics = read(path)
            protocol = metrics["protocol"]
            if (protocol["frames"] != 15985 or protocol["split"] != "test"
                    or protocol["source_sha256"] != record["best"]["sha256"]
                    or protocol["max_frames"] or protocol["skip_evaluation"]):
                raise RuntimeError(f"Invalid final selected Test: {path}")
            record["selected_test"] = dict(path=str(path), sha256=digest(path),
                                            selected_joint_epochs=record["best"]["joint_epochs"])

    def collect(self):
        for gpu, job in list(self.state["active_jobs"].items()):
            process = self.processes.get(gpu)
            if process is None:
                if alive(job["pid"]):
                    raise RuntimeError(f"An earlier worker is still alive; do not duplicate it: {job}")
                # A prior launcher may have exited after the artifact was saved.
                code = 0
            else:
                code = process.poll()
                if code is None:
                    continue
                self.logs.pop(gpu).close()
                self.processes.pop(gpu)
            if code:
                raise RuntimeError(f"Worker failed ({code}); inspect {job['log']}")
            self.integrate(job)
            del self.state["active_jobs"][gpu]
            self.save()
            print("JOB_COMPLETE", json.dumps(job), flush=True)

    def next_job(self):
        busy = {job["arm"] for job in self.state["active_jobs"].values()}
        candidates = []
        for arm, record in self.state["arms"].items():
            if arm in busy:
                continue
            joint = record["trained_joint"]
            if not any(row["joint_epochs"] == joint for row in record["dev_history"]):
                candidates.append((0, arm, "dev", joint))
            elif self.state.get("training_stopped"):
                if "selected_test" not in record:
                    candidates.append((1, arm, "test", record["best"]["joint_epochs"]))
            elif joint < self.state["target_joint_budget"]:
                # Give the longer paired chains priority; one arm cannot advance past
                # a shared decision budget until every arm reaches it.
                remaining = (self.state["target_joint_budget"] - joint) * (0.65 if arm == "rgb_matched" else 1)
                candidates.append((2 - remaining, arm, "train", joint + 3))
        if not candidates:
            return None
        return sorted(candidates)[0][1:]

    def decide(self):
        if self.state.get("training_stopped") or self.state["active_jobs"]:
            return
        budget = self.state["target_joint_budget"]
        if not all(record["trained_joint"] == budget and record["dev_history"][-1]["joint_epochs"] == budget
                   for record in self.state["arms"].values()):
            return
        decision = plateau_decision(self.plan, {arm: record["dev_history"] for arm, record in self.state["arms"].items()}, budget)
        self.state["decisions"].append(decision)
        self.state["training_stopped"] = decision["stop"]
        if decision["stop"]:
            self.state["common_final_joint_budget"] = budget
        else:
            self.state["target_joint_budget"] = decision["next_budget"]
        self.save()
        print("DEV_BUDGET_DECISION", json.dumps(decision), flush=True)

    def notice(self):
        if time.time() - self.last_notice < 45:
            return
        self.last_notice = time.time()
        jobs = []
        for job in self.state["active_jobs"].values():
            progress = Path(job["folder"]) / "progress.json"
            fields = read(progress) if progress.exists() else {}
            jobs.append(dict(arm=job["arm"], kind=job["kind"], gpu=job["physical_gpu"],
                             target_joint=job["joint_epochs"],
                             progress={k:fields[k] for k in ["epoch","frames","expected","mean_loss","skipped_amp_updates"] if k in fields}))
        print(json.dumps(dict(time=time.strftime("%F %T"), status=self.state["status"],
            target_joint_budget=self.state["target_joint_budget"], jobs=jobs,
            arms={arm:dict(trained_joint=r["trained_joint"], best_joint=r["best"]["joint_epochs"],
                           best_Dev_HOTA=r["best"]["HOTA"]) for arm,r in self.state["arms"].items()},
            local_free_gib=round(shutil.disk_usage(self.root).free/2**30,2))), flush=True)

    def cleanup(self):
        cache = read(self.args.cache_manifest)
        target = Path(cache["cache_root"]).resolve()
        if target.parent != Path("/dev/shm") or target.name != "ir_mft_diffv2ir_train_long_20261121":
            raise RuntimeError("Refusing to remove an unexpected cache path")
        if target.exists():
            shutil.rmtree(target)
        write(self.root / "train_byte_cache_cleanup.json", dict(removed=True, cache_root=str(target),
            original_source_root=cache["source_root"], original_images_untouched=True,
            unix=time.time(), reason="all generated-input extension training and evaluation complete"))

    def run(self):
        if self.state["status"] == "COMPLETE":
            print("CAMPAIGN_ALREADY_COMPLETE", flush=True)
            return
        while not self.prerequisites():
            self.notice()
            self.save()
            time.sleep(10)
        self.initialize()
        self.state["status"] = "ACTIVE"
        self.save()
        while True:
            self.collect()
            self.decide()
            for gpu in [2, 3]:
                if str(gpu) not in self.state["active_jobs"]:
                    pending = self.next_job()
                    if pending:
                        self.launch(gpu, *pending)
            self.notice()
            if self.state.get("training_stopped") and not self.state["active_jobs"] and all(
                    "selected_test" in record for record in self.state["arms"].values()):
                self.cleanup()
                self.state.update(status="COMPLETE", completed_unix=time.time())
                self.save()
                print("CAMPAIGN_COMPLETE", json.dumps(self.state["decisions"][-1]), flush=True)
                return
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    args = parser.parse_args()
    campaign = Campaign(args)
    lock = (campaign.root / "launcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        campaign.run()
    except Exception as exc:
        campaign.state.update(status="FAILED", failure=str(exc), failed_unix=time.time())
        campaign.save()
        # Finish any running sibling job; do not kill unrelated work or discard
        # this campaign's recoverable epoch-boundary checkpoint.
        for process in campaign.processes.values():
            process.wait()
        raise


if __name__ == "__main__":
    main()
