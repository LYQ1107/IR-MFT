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
    parser.add_argument("--encoding", choices=("jpg", "webp"), default="jpg")
    parser.add_argument(
        "--representation",
        choices=("response", "color"),
        default="response",
        help="store the one-channel causal response (recommended) or its color map",
    )
    parser.add_argument("--quality", type=int, default=100, help="lossy-cache quality (higher preserves the generated response)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sequence", default=None, help="build only one protocol sequence")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=None)
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
    if len(sequences) == 1:
        sequence_images = images
        max_frame = len(sequence_images)
        start_frame = max(1, args.start_frame)
        end_frame = max_frame if args.end_frame is None else min(args.end_frame, max_frame)
        if start_frame > end_frame:
            raise ValueError(f"empty frame range {start_frame}:{end_frame}")
    else:
        start_frame, end_frame = 1, None
    os.makedirs(args.output, exist_ok=True)

    frame_root = "test" if args.split == "historical_test" else "train"

    generators = {sequence: CausalAuxiliaryGenerator(alpha=args.alpha) for sequence in sequences}
    counts = {sequence: 0 for sequence in sequences}
    total_bytes = 0
    output_total = max(0, (end_frame - start_frame + 1) if end_frame is not None else len(images))
    output_index = 0
    for index, image in enumerate(images, start=1):
        sequence = videos[int(image["video_id"])]
        visible_path = os.path.join(args.data_root, frame_root, image["vi_file_name"])
        visible = cv2.imread(visible_path, cv2.IMREAD_COLOR)
        if visible is None:
            raise FileNotFoundError(visible_path)
        auxiliary = generators[sequence](visible, sequence_id=sequence, frame_id=int(image["frame_id"]))
        if end_frame is not None and not (start_frame <= index <= end_frame):
            continue
        relative = image["vi_file_name"]
        target_dir = os.path.join(args.output, os.path.dirname(relative))
        extension = ".jpg" if args.encoding == "jpg" else ".webp"
        target = os.path.join(target_dir, os.path.basename(relative).rsplit(".", 1)[0] + extension)
        if os.path.exists(target) and not args.overwrite:
            total_bytes += os.path.getsize(target)
        else:
            os.makedirs(target_dir, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".causal-", suffix=extension, dir=target_dir)
            os.close(fd)
            try:
                quality_flag = (
                    cv2.IMWRITE_JPEG_QUALITY
                    if args.encoding == "jpg"
                    else cv2.IMWRITE_WEBP_QUALITY
                )
                cache_image = (
                    generators[sequence].last_response
                    if args.representation == "response"
                    else auxiliary
                )
                if not cv2.imwrite(temporary, cache_image, [quality_flag, args.quality]):
                    raise RuntimeError(f"OpenCV failed to write {temporary}")
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            total_bytes += os.path.getsize(target)
        counts[sequence] += 1
        output_index += 1
        if output_index == 1 or output_index % 500 == 0 or output_index == output_total:
            print(f"{output_index}/{output_total} frames, cache={total_bytes / (1024 ** 3):.2f} GiB", flush=True)

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
        "encoding": args.encoding,
        "representation": args.representation,
        "quality": args.quality,
        "bytes": total_bytes,
        "cache_root": os.path.abspath(args.output),
    }
    if args.sequence is None:
        metadata_name = "CACHE_METADATA.json"
    elif args.end_frame is None and args.start_frame == 1:
        metadata_name = f"CACHE_METADATA_{args.sequence}.json"
    else:
        metadata_name = f"CACHE_METADATA_{args.sequence}_{start_frame:05d}_{end_frame:05d}.json"
    with open(os.path.join(args.output, metadata_name), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
