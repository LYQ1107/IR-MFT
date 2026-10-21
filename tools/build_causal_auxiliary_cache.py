#!/usr/bin/env python3
"""Build a disk-backed causal auxiliary cache without copying RGB frames."""

import argparse
import hashlib
import json
import os
import tempfile

import cv2

from yolox.data.causal_auxiliary import CausalAuxiliaryGenerator


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("MFT26IR_ROOT", "/data3/liuyeqiang/datasets/mft26-ir"))
    parser.add_argument("--annotation", default="train.json")
    parser.add_argument("--output", default=os.environ.get("MFT26IR_CAUSAL_CACHE_ROOT", "/data3/liuyeqiang/IR-MFT_assets/causal_aux_cache"))
    parser.add_argument("--protocol", default="configs/cfp/protocol_v2.json")
    parser.add_argument("--split", default="train", choices=("train", "dev", "historical_test"))
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--quality", type=int, default=100, help="WebP quality; 100 keeps the cache near-lossless")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sequence", default=None, help="build only one protocol sequence")
    args = parser.parse_args()

    with open(args.protocol, "r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    sequences = protocol[args.split]
    if args.sequence is not None:
        if args.sequence not in sequences:
            raise ValueError(f"{args.sequence} is not in protocol split {args.split}")
        sequences = [args.sequence]
    annotation_path = os.path.join(args.data_root, "annotations", args.annotation)
    with open(annotation_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    videos = {int(video["id"]): video["file_name"] for video in payload["videos"]}
    images = [
        image
        for image in payload["images"]
        if videos[int(image["video_id"])] in sequences
    ]
    images.sort(key=lambda image: (videos[int(image["video_id"])], int(image["frame_id"])))
    os.makedirs(args.output, exist_ok=True)

    generators = {sequence: CausalAuxiliaryGenerator(alpha=args.alpha) for sequence in sequences}
    counts = {sequence: 0 for sequence in sequences}
    total_bytes = 0
    for index, image in enumerate(images, start=1):
        sequence = videos[int(image["video_id"])]
        visible_path = os.path.join(args.data_root, "train", image["vi_file_name"])
        visible = cv2.imread(visible_path, cv2.IMREAD_COLOR)
        if visible is None:
            raise FileNotFoundError(visible_path)
        auxiliary = generators[sequence](visible, sequence_id=sequence, frame_id=int(image["frame_id"]))
        relative = image["vi_file_name"]
        target_dir = os.path.join(args.output, os.path.dirname(relative))
        target = os.path.join(target_dir, os.path.basename(relative).rsplit(".", 1)[0] + ".webp")
        if os.path.exists(target) and not args.overwrite:
            total_bytes += os.path.getsize(target)
        else:
            os.makedirs(target_dir, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".causal-", suffix=".webp", dir=target_dir)
            os.close(fd)
            try:
                if not cv2.imwrite(temporary, auxiliary, [cv2.IMWRITE_WEBP_QUALITY, args.quality]):
                    raise RuntimeError(f"OpenCV failed to write {temporary}")
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            total_bytes += os.path.getsize(target)
        counts[sequence] += 1
        if index == 1 or index % 500 == 0 or index == len(images):
            print(f"{index}/{len(images)} frames, cache={total_bytes / (1024 ** 3):.2f} GiB", flush=True)

    metadata = {
        "schema_version": 1,
        "source_annotation": os.path.abspath(annotation_path),
        "source_annotation_sha256": sha256(annotation_path),
        "data_root": os.path.abspath(args.data_root),
        "protocol_id": protocol["protocol_id"],
        "split": args.split,
        "sequences": sequences,
        "frames_by_sequence": counts,
        "alpha": args.alpha,
        "response_scale": 4.0,
        "encoding": "webp",
        "quality": args.quality,
        "bytes": total_bytes,
        "cache_root": os.path.abspath(args.output),
    }
    metadata_name = "CACHE_METADATA.json" if args.sequence is None else f"CACHE_METADATA_{args.sequence}.json"
    with open(os.path.join(args.output, metadata_name), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
