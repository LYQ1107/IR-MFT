#!/usr/bin/env python3
"""Reproduce original RGB Test tracks exactly, without labels or accuracy."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from tools.eval_mft26ir import write_sequence_results
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.byte_tracker import BYTETracker


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());work=Path(plan['work_root'])
    source=Path(plan['frozen_detection_sources']['rgb_test'])
    folder=work/'canonical_RGB_track_parity';folder.mkdir(exist_ok=False)
    records=[]
    for sequence in protocol['historical_test']:
        packet=source/'detections'/(sequence+'.npz')
        with np.load(packet)as z:
            data={key:z[key]for key in ['frame_ids','offsets','boxes','scores']}
        np.testing.assert_array_equal(data['frame_ids'],np.arange(1,len(data['frame_ids'])+1))
        tracker=BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),frame_rate=25)
        rows=[]
        for frame in data['frame_ids']:
            lo,hi=data['offsets'][frame-1:frame+1]
            detections=np.c_[data['boxes'][lo:hi],data['scores'][lo:hi]].astype(np.float32)
            rows.extend((int(frame),track.tlwh.copy(),track.track_id,track.score)
                for track in tracker.update(detections,(1,1),(1,1))if track.tlwh[2]*track.tlwh[3]>100)
        output=folder/(sequence+'.txt');write_sequence_results(str(output),rows)
        original=source/'track_results'/(sequence+'.txt');current_hash=sha256(output)
        record=dict(sequence=sequence,frames=len(data['frame_ids']),output_rows=len(rows),
            packet_sha256=sha256(packet),original_track_sha256=sha256(original),replay_track_sha256=current_hash,
            tracks_byte_exact=current_hash==sha256(original))
        records.append(record);dump(folder/'progress.json',dict(rows=records,GT_read=False,accuracy_read=False))
        print(json.dumps(record),flush=True)
        assert record['tracks_byte_exact'],'Keep failed replay for diagnosis; no baseline accuracy claim'
        output.unlink()  # Only this newly generated, verified duplicate; retain originals.
    assert sum(r['frames']for r in records)==15985
    dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_CANONICAL_RGB_TEST_TRACK_PARITY.json',
        dict(status='COMPLETE_ALL7_ORIGINAL_RGB_TEST_TRACKS_BYTE_EXACT',rows=records,
            frames=15985,GT_read=False,accuracy_read=False,source_sha256=sha256(Path(__file__)),
            tracker_sha256=sha256(ROOT/'yolox/tracker/byte_tracker.py'),
            scope='Original cached RGB proposals replayed through current unchanged ByteTrack. '
                  'No image inference, label access, Test metric calculation or model selection. '
                  'Only verified newly generated duplicate text tracks were removed.'))


if __name__=='__main__':
    main()
