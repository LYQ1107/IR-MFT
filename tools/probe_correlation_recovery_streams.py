#!/usr/bin/env python3
"""Verify new interleaved writing against all frozen CGR original Dev tracks.

Existing actual-image detection packets only; no encoder, GT or GPU work.
All original IDs must match at emission, including continuation across videos.
"""
import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from tools.correlation_recovery_inference import LegacyCaseTrackWriter
from tools.run_pseudo_ir_comparison import dump, sha256


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    spec_path = ROOT / 'configs/perceptual_correlation_recovery_backfill_20261124.json'
    spec = read(spec_path)
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    protocol = read(ROOT / plan['protocol'])
    work = Path(plan['work_root'])
    folder = work / 'correlation_recovery_new_stream_Dev'
    assert not folder.exists(), 'Preserve previous actual replay evidence'
    folder.mkdir()
    state_path = folder / 'state.json'
    cases = [row for row in spec['cases'] if row.get('source') and row['intervention'] == 'normal']
    assert len(cases) == 8
    names = [row['id'] for row in cases]
    counters = {name: [0] for name in names}
    sources, proof = {}, {}
    for row in cases:
        path = Path(row['source']['source_Dev_metrics'])
        assert sha256(path) == row['source']['source_Dev_metrics_sha256']
        sources[row['id']] = read(path)['manifest']
    state = dict(status='VERIFYING_ALL8_CGR_LEGACY_IDS_ON_FULL_DEV', controller_pid=os.getpid(),
        specification_sha256=sha256(spec_path), GPU_work=False, cases=names, sequences=protocol['dev'])
    dump(state_path, state)
    try:
        for sequence in protocol['dev']:
            packets = {}
            for name, source in sources.items():
                path = Path(source['sources'][sequence]['path'])
                assert sha256(path) == source['sources'][sequence]['sha256']
                with np.load(path) as saved:
                    packets[name] = {k: saved[k] for k in ['frame_ids', 'offsets', 'boxes', 'scores']}
                frame_ids = packets[name]['frame_ids']
                np.testing.assert_array_equal(frame_ids, np.arange(1, len(frame_ids)+1))
            lengths = {len(packet['frame_ids']) for packet in packets.values()}
            assert len(lengths) == 1
            frames = lengths.pop()
            writer = LegacyCaseTrackWriter(folder, sequence, names, counters)
            try:
                for frame in range(1, frames+1):
                    current = {}
                    for name, packet in packets.items():
                        start, stop = packet['offsets'][frame-1:frame+1]
                        current[name] = np.c_[packet['boxes'][start:stop], packet['scores'][start:stop]].astype(np.float32)
                    writer.append(frame, current)
                    if frame % 500 == 0 or frame == frames:
                        writer.flush()
                        state.update(sequence=sequence, frame=frame, frames=frames)
                        dump(state_path, state)
                        print(json.dumps(dict(sequence=sequence, frame=frame, cases=len(cases))), flush=True)
                complete = writer.close(frames)
            except Exception:
                writer.abort()
                raise
            proof[sequence] = {}
            for name, result in complete.items():
                original = sources[name]['coverage'][sequence]['output_sha256']
                assert result['track_sha256'] == original, (sequence, name)
                proof[sequence][name] = dict(result, original_track_sha256=original,
                    literal_legacy_track_IDs_exact=True, terminal_private_counter=counters[name][0])
        assert sum(next(iter(rows.values()))['frames'] for rows in proof.values()) == 4834
        report = dict(status='COMPLETE_ALL8_CGR_NEW_STREAMS_LEGACY_IDS_EXACT_ON_ALL4834DEV',
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), proof=proof,
            specification_sha256=sha256(spec_path), fixed_seed=0, additional_seed_work=False,
            GPU_work=False, GT_read=False, official_metrics_reevaluated=False,
            source_hashes={name: sha256(ROOT / name) for name in [
                'tools/probe_correlation_recovery_streams.py', 'tools/correlation_recovery_inference.py',
                'tools/native_pyramid_stream_tracks.py', 'yolox/tracker/byte_tracker.py']},
            scope='All8normal/paired frozen source cases,2complete Dev videos/4834frames each. '
                'Real original detector packets replayed through interleaved per-case counters. '
                'All16literal track files including IDs and scores are byte-exact to old separate-process '
                'CGR replay. Only numbering state crosses videos; trackers/evidence reset. '
                'No neural inference or Test, no new HOTA; GPU pixel/source engineering remains pending.',
            goal_met=False, supported_innovations=0)
        path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_CGR_NEW_STREAM_FULL_DEV_PROOF.json'
        assert not path.exists()
        dump(path, report)
        state.update(status=report['status'], terminal_private_counters={k: v[0] for k, v in counters.items()})
        dump(state_path, state)
        print(report['status'], flush=True)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_CGR_NEW_STREAM_PROBE', error=repr(error))
        dump(state_path, state)
        raise


if __name__ == '__main__':
    main()
