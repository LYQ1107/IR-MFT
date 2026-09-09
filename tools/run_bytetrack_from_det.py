import os
import cv2
import glob
import argparse
import numpy as np
from types import SimpleNamespace

from yolox.tracker.byte_tracker import BYTETracker


def read_det_file(det_path):
    dets_by_frame = {}
    with open(det_path, "r") as f:
        for line in f:
            line = line.strip().replace(",", " ")
            if not line:
                continue
            parts = line.split()
            frame = int(float(parts[0]))
            x, y, w, h, score = map(float, parts[2:7])
            x1, y1, x2, y2 = x, y, x + w, y + h
            dets_by_frame.setdefault(frame, []).append([x1, y1, x2, y2, score])
    return dets_by_frame


def get_image_size(seq_img_dir):
    imgs = []
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
        imgs.extend(glob.glob(os.path.join(seq_img_dir, ext)))
    imgs = sorted(imgs)
    if len(imgs) == 0:
        raise FileNotFoundError(f"No images found in {seq_img_dir}")
    img = cv2.imread(imgs[0])
    h, w = img.shape[:2]
    return h, w, imgs


def run_one_sequence(seq_name, det_path, img_dir, out_path, args):
    dets_by_frame = read_det_file(det_path)
    h, w, imgs = get_image_size(img_dir)

    tracker_args = SimpleNamespace(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        mot20=False
    )

    tracker = BYTETracker(tracker_args, frame_rate=args.frame_rate)
    results = []

    num_frames = len(imgs)
    for frame_id in range(1, num_frames + 1):
        dets = np.array(dets_by_frame.get(frame_id, []), dtype=np.float32)

        if dets.shape[0] == 0:
            dets = np.empty((0, 5), dtype=np.float32)

        online_targets = tracker.update(
            dets,
            img_info=(h, w),
            img_size=(h, w)
        )

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            x, y, bw, bh = tlwh
            score = t.score

            if bw <= 0 or bh <= 0:
                continue

            results.append(
                f"{frame_id},{tid},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},{score:.4f},-1,-1,-1\n"
            )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(results)

    print(f"[Done] {seq_name}: saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_root", required=True, help="检测结果目录，每个序列一个 txt")
    parser.add_argument("--img_root", required=True, help="图像根目录，每个序列一个文件夹")
    parser.add_argument("--out_root", required=True, help="ByteTrack 输出目录")
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--frame_rate", type=int, default=30)
    args = parser.parse_args()

    det_files = sorted(glob.glob(os.path.join(args.det_root, "*.txt")))

    for det_path in det_files:
        seq_name = os.path.splitext(os.path.basename(det_path))[0]
        img_dir = os.path.join(args.img_root, seq_name)
        out_path = os.path.join(args.out_root, f"{seq_name}.txt")

        if not os.path.exists(img_dir):
            print(f"[Skip] image dir not found: {img_dir}")
            continue

        run_one_sequence(seq_name, det_path, img_dir, out_path, args)


if __name__ == "__main__":
    main()
