#!/usr/bin/env python3
"""Archive immutable, completed short-run checkpoints without changing their paths.

Copy and verify a full NAS file before atomically replacing its local file
with a symlink. Active continuation checkpoints and images are not touched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


ARMS = ["train_old_legacy_s0", "train_new_diffv2ir_s0", "train_rgb_matched_s0"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    os.replace(temporary,path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--short-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    expected_short = Path("/data3/liuyeqiang/IR-MFT_assets/pseudo_ir_comparison_20261120")
    allowed_nas = Path("/data3/liuyeqiang/nas/A100/IR-MFT")
    if args.short_root != expected_short or not args.archive_root.is_relative_to(allowed_nas):
        raise ValueError("Only this campaign's completed prefix checkpoints may be archived")
    records = []
    for arm in ARMS:
        complete = json.loads((args.short_root/arm/"completion.json").read_text())
        source = args.short_root/arm/"latest_ckpt.pth.tar"
        if complete["status"] != "COMPLETE" or complete["checkpoint"] != str(source):
            raise ValueError(f"Incomplete or unexpected source: {arm}")
        records.append(dict(arm=arm,source=str(source),destination=str(args.archive_root/arm/source.name),
                            bytes=source.stat().st_size,sha256=complete["sha256"],archived=False))
    report = dict(status="PLANNED",started_unix=time.time(),records=records,
                  policy="Keep complete bytes and SHA256 on NAS; original local path becomes a symlink; no model/image or GitHub payload changes")
    if args.dry_run:
        print(json.dumps(dict(report,reclaimable_gib=sum(r["bytes"] for r in records)/2**30),indent=2))
        return
    args.archive_root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.archive_root).free < sum(r["bytes"] for r in records) + 10*2**30:
        raise RuntimeError("NAS storage reserve is too small for this archive")
    dump(args.manifest,report)
    for record in records:
        source,destination = Path(record["source"]),Path(record["destination"])
        if source.is_symlink():
            if source.resolve() != destination.resolve() or digest(destination) != record["sha256"]:
                raise RuntimeError(f"Unexpected existing source link: {source}")
        else:
            original_stat = source.stat()
            if digest(source) != record["sha256"]:
                raise RuntimeError(f"Completed checkpoint changed: {source}")
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():
                temporary = destination.with_suffix(".partial")
                h = hashlib.sha256()
                with source.open("rb") as src, temporary.open("xb") as dst:
                    for block in iter(lambda:src.read(8*1024*1024),b""):
                        dst.write(block)
                        h.update(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                if h.hexdigest() != record["sha256"]:
                    raise RuntimeError(f"Source changed while copying: {source}")
                os.replace(temporary,destination)
            if destination.stat().st_size != record["bytes"] or digest(destination) != record["sha256"]:
                raise RuntimeError(f"NAS byte verification failed: {destination}")
            current_stat = source.stat()
            if source.is_symlink() or any(getattr(original_stat,key) != getattr(current_stat,key)
                    for key in ["st_dev","st_ino","st_size","st_mtime_ns"]):
                raise RuntimeError(f"Local source changed during archival: {source}")
            # The original inode is removed only after the complete remote file
            # has passed an independent read-back hash check.
            link = source.with_suffix(".archive_link")
            link.symlink_to(destination)
            os.replace(link,source)
        record.update(archived=True,local_path_is_symlink=source.is_symlink(),
                      completed_unix=time.time())
        dump(args.manifest,report)
        print("CHECKPOINT_ARCHIVED",json.dumps(record),flush=True)
    report.update(status="COMPLETE",reclaimed_bytes=sum(r["bytes"] for r in records),
                  completed_unix=time.time(),local_free_gib=shutil.disk_usage(args.short_root).free/2**30)
    dump(args.manifest,report)
    print("ARCHIVE_COMPLETE",json.dumps({k:v for k,v in report.items() if k!="records"}),flush=True)


if __name__ == "__main__":
    main()
