#!/usr/bin/env python3
"""Check bounded-RAM decoding on the same twelve previously checked Train frames."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from tools.run_pseudo_ir_comparison import dump, sha256
from tools.structural_train_image_reader import TrainImageReader


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cv2.setNumThreads(1)
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    previous_path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_TRAIN_STREAM_INPUT_PARITY_GPU5.json'
    previous = json.loads(previous_path.read_text())
    assert previous['status'] == 'COMPLETE_TRAIN_INPUT_PARITY_PROBE'
    assert len(previous['rows']) == 12 and set(r['sequence'] for r in previous['rows']) == set(protocol['train'])
    reader = TrainImageReader(plan['data_root'], plan['generated_ir_root'], protocol['train'])
    rows = []
    started = time.perf_counter()
    for row in previous['rows']:
        seq, frame = row['sequence'], row['frame']
        result = dict(sequence=seq, frame=frame, modalities={})
        for modality in ['rgb', 'ir']:
            root, folder, suffix, flag = ((plan['data_root'], 'visible', 'jpg', 1) if modality == 'rgb' else
                                         (plan['generated_ir_root'], 'infrared_diffv2ir', 'png', -1))
            path = Path(root) / 'train' / seq / folder / f'{frame:06d}.{suffix}'
            expected = cv2.imread(str(path), flag)
            assert expected is not None
            for repeat in range(2):
                actual = reader.read(seq, frame, modality)
                assert np.array_equal(actual, expected) and actual.dtype == expected.dtype
            result['modalities'][modality] = dict(shape=list(expected.shape), dtype=str(expected.dtype),
                original_file_sha256=sha256(path), both_decodes_exact_to_original_imread=True)
        rows.append(result)
    report = dict(status='COMPLETE_ORIGINAL_TRAIN_BYTE_READER_PARITY', source_sha256=sha256(__file__),
        reader_sha256=sha256(ROOT / 'tools/structural_train_image_reader.py'),
        protocol_sha256=sha256(ROOT / plan['protocol']), prior_GPU_input_proof_sha256=sha256(previous_path),
        Train_unique_frames=12, original_image_files=24, decoder_comparisons=48, rows=rows,
        statistics=reader.statistics(), seconds=time.perf_counter() - started,
        GPU_work=False, training=False, Dev_read=False, Test_read=False,
        scope='Same12Trainframes as earlier GPUinput probe; RGBand actualNovemberIR encodedbytes only in boundedRAM. '
              'This verifies original decodedpixels, not GPUupdates, expandedTraincoverage or accuracy. '
              'Newcache inserts require40GiB available hostRAM after insertion; other hostprocesses remain uncontrolled.',
        goal_met=False, supported_innovations=0)
    destination = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_TRAIN_STREAM_RAM_READER_PARITY.json'
    assert not destination.exists(), 'Preserve the earlier proof; do not overwrite'
    dump(destination, report)
    print(json.dumps(dict(status=report['status'], statistics=report['statistics'],
                         seconds=report['seconds']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
