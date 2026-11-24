#!/usr/bin/env python3
"""Compare every retained candidate crop with pinned original Train pixels."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from tools.native_pyramid_content_guard import make_image_reader
from tools.neck72_experiment import LENGTHS, registered
from tools.neck72_runtime import read, require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.native_cue import native_crop


def compare_crops(sequence, cue, frames, boxes, crops, rectangles, reader, progress=None):
    if cue not in {'rgb', 'ir'} or len(frames) != len(boxes) or len(frames) != len(crops):
        raise ValueError('Aligned current Train crop arrays required')
    order = np.argsort(frames, kind='stable')
    unique, starts, counts = np.unique(frames[order], return_index=True, return_counts=True)
    checked = 0
    for frame, start, count in zip(unique, starts, counts):
        image = reader.read(sequence, int(frame), cue)
        for index in order[start:start+count]:
            current, rectangle = native_crop(image, boxes[index])
            if not np.array_equal(current, crops[index]) or not np.array_equal(rectangle, rectangles[index]):
                raise RuntimeError(f'Cached Train crop differs from pinned current pixels: '
                                   f'{sequence}/{cue}/frame{frame}/observation{index}')
            checked += 1
        if progress is not None:
            progress(checked, len(unique), int(frame))
    return dict(observations=checked, actual_source_frames=len(unique), every_crop_and_rectangle_exact=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--cue', choices=['rgb', 'ir'], required=True)
    args = parser.parse_args()
    plan, protocol, work, study, _ = registered()
    assert args.sequence in protocol['train']
    cv2.setNumThreads(1); require_host_memory(8*2**30); require_storage(work)
    path = work / 'neck72_crop_verification' / args.sequence / (args.cue + '.json')
    assert not path.exists(), 'Preserve earlier crop verification evidence'
    reader, reference = make_image_reader(work, plan, protocol, LENGTHS, args.cue)
    reader.max_bytes = 256*2**20
    source = work / 'native_candidate_crops' / args.sequence / 'crops.npz'
    metadata_path = source.parent / 'complete.json'; metadata = read(metadata_path)
    assert metadata['status'] == 'COMPLETE_EXACT_TRAIN_CANDIDATE_CROPS'
    assert not metadata['Dev_rows'] and not metadata['Test_rows']
    assert sha256(source) == metadata['crop_sha256']
    index = work / 'native_candidate_index' / args.sequence / 'index.npz'
    assert sha256(index) == metadata['index_sha256']
    with np.load(index) as array:
        frames, boxes = array['frame'].copy(), array['boxes'].copy()
    with np.load(source) as array:
        np.testing.assert_array_equal(frames, array['frame'])
        np.testing.assert_array_equal(boxes, array['boxes'])
        crops, rectangles = array[args.cue], array['rectangles']
    started = time.monotonic()
    result = compare_crops(args.sequence, args.cue, frames, boxes, crops, rectangles, reader)
    assert result['observations'] == metadata['observations'] and result['actual_source_frames'] == metadata['frames']
    require_storage(work)
    dump(path, dict(status='COMPLETE_NECK72_ALL_CACHED_CROPS_EXACT_TO_PINNED_CURRENT_TRAIN',
        sequence=args.sequence, cue=args.cue, **result, reference=reference,
        reader_statistics=reader.statistics(), study_sha256=sha256(ROOT / 'configs/perceptual_neck72_study_20261124.json'),
        crop_bank_sha256=sha256(source), crop_metadata_sha256=sha256(metadata_path),
        index_sha256=sha256(index), source_sha256=sha256(__file__),
        GT_read=False, new_pixels_saved=False, GPU_work=False, seconds=time.monotonic()-started))
    print(args.sequence, args.cue, result, flush=True)


if __name__ == '__main__':
    main()
