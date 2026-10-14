#!/usr/bin/env python3
"""Run the fixed three-mode, three-seed FDTA-style crop campaign."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.state.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", None):
        raise ValueError("New CPU-only driver state required")
    if args.root.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Outside-Git output root required")
    args.root.mkdir(parents=True, exist_ok=True)
    modes = config["modes"]; seeds = config["seed_plan"]
    sources = [__file__, "tools/cfp_train_fdta_adapter.py", "cfp/fdta_fish_adapter.py",
               "cfp/local_identity_training.py", "cfp/reid.py", "cfp/crop_retrieval.py", str(args.config)]
    guards = {p: sha256(p) for p in sources}
    state = dict(status="RUNNING", modes=modes, seeds=seeds, completed=[], active=None,
                 source_guards=guards, sources=snapshot_sources(args.state.with_suffix(".sources"), sources),
                 scope=config["scope"])
    write_json(args.state, state)
    try:
        for mode in modes:
            for seed in seeds:
                if any(sha256(p) != value for p, value in guards.items()):
                    raise ValueError("FDTA source guard changed")
                output = args.root / f"fdta_adapter_{mode}_s{seed}_v1"
                command = [sys.executable, "tools/cfp_train_fdta_adapter.py", "--features", str(args.features),
                           "--config", str(args.config), "--mode", mode, "--seed", str(seed), "--output", str(output)]
                log = Path(str(output) + ".log")
                state["active"] = dict(mode=mode, seed=seed, output=str(output), command=command, log=str(log))
                write_json(args.state, state)
                with log.open("x") as stream:
                    code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
                if code:
                    raise RuntimeError(f"FDTA child failed with {code}: {mode} seed {seed}")
                manifest = json.loads((output / "manifest.json").read_text())
                if manifest["status"] != "COMPLETE":
                    raise RuntimeError("FDTA child did not complete: " + str(output))
                state["completed"].append(dict(mode=mode, seed=seed, output=str(output),
                                               manifest_sha256=sha256(output / "manifest.json"),
                                               checkpoint_sha256=manifest["checkpoint_sha256"]))
                state["active"] = None; write_json(args.state, state)
        state["status"] = "COMPLETE_FDTA_STYLE_ADAPTER_9_FITS"
        write_json(args.state, state)
        print(json.dumps({"status": state["status"], "completed": len(state["completed"])}))
    except Exception as error:
        state.update(status="HALTED_ACTUAL_FAILURE", error=str(error)); write_json(args.state, state); raise


if __name__ == "__main__":
    main()
