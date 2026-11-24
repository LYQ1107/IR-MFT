"""One-case causal prefix timing; no model loader, GPU permission or accuracy.

The registered driver supplies complete-production references and frozen models.
Artificial CPU tests exercise timing/state boundaries, not real GPU costs.
"""
import hashlib
import math
import statistics
import time

import numpy as np
import torch

from tools.fusion_memory_parent_io import require
from tools.native_pyramid_stream_tracks import mot_line


def retained_state_bytes(engine):
    """Logical array/tensor bytes, excluding NN weights and Python/container heap.

    Sampled after each frame, not a peak-allocation or total host-memory estimate.
    Removed-track KF/box arrays are included; consumed ROI observations must be gone.
    """
    groups = dict(track_appearance=0, track_KF_and_boxes=0, visual_temporal=0,
                  previous_gray=0)
    arrays, storages, tracks = set(), set(), set()

    def count(value, group):
        if isinstance(value, np.ndarray):
            if id(value) not in arrays:
                arrays.add(id(value))
                groups[group] += value.nbytes
        elif torch.is_tensor(value):
            storage = value.untyped_storage()
            key = str(value.device), storage.data_ptr(), storage.nbytes()
            if key not in storages:
                storages.add(key)
                groups[group] += storage.nbytes()
        elif isinstance(value, (tuple, list, dict)):
            for item in (value.values() if isinstance(value, dict) else value):
                count(item, group)

    for tracker in engine.trackers.values():
        if tracker.core is None:
            continue
        for collection in [tracker.core.tracked_stracks, tracker.core.lost_stracks,
                           tracker.core.removed_stracks]:
            for track in collection:
                if id(track) in tracks:
                    continue
                tracks.add(id(track))
                require(track.observation is None, 'Consumed ROI observation was retained')
                for name in ['rgb_memory', 'aux_memory', 'fused_memory']:
                    count(getattr(track, name), 'track_appearance')
                for name in ['_tlwh', 'mean', 'covariance']:
                    count(getattr(track, name, None), 'track_KF_and_boxes')
    for state in engine.detector.states.values():
        if state is not None:
            for name in ['latent', 'covariance', 'valid']:
                count(getattr(state, name), 'visual_temporal')
    count(engine.detector.previous_gray, 'previous_gray')
    require(all(c.current is None and c.rgb is None and c.fused is None and c.auxiliary is None
                for c in engine.captures.values()), 'A consumed current feature capture was retained')
    return dict(groups, total=sum(groups.values()))


def _reference(reference, case, sequence, frames):
    require(reference['sequence'] == sequence and reference['prefix_frames'] == frames and
            len(reference['prefix_source_rows']) == frames, 'Complete matching prefix reference required')
    for frame, row in enumerate(reference['prefix_source_rows'], 1):
        require(len(row) == 4 and row[:2] == [sequence, frame] and all(
            isinstance(v, str) and len(v) == 64 for v in row[2:]), 'Exact RGB/IR byte references required')
    for key in ['prefix_packet_sha256', 'prefix_literal_MOT_sha256']:
        value = reference[key][case['id']]
        require(isinstance(value, str) and len(value) == 64, 'Original packet and MOT prefix hashes required')


def _pass(engine_factory, images_factory, case, sequence, reference, frames, warmup,
          timed, synchronize, clock, resource_check):
    engine, images = engine_factory(), None
    try:
        require(engine.cases == [case] and len(engine.routes) == 1 and
                len(engine.trackers) == 1, 'Measure exactly one frozen visual plus memory configuration')
        require(engine.sequence is None and engine.last_frame == 0,
                'Independent timing must begin with fresh causal state')
        images = images_factory()
        require(images.sequence == sequence and images.last_frame == 0 and images.frames >= frames,
                'Fresh sequential image reader with the complete declared source range required')
        packets, tracks = hashlib.sha256(), hashlib.sha256()
        durations, largest = [], {}
        for frame in range(1, frames+1):
            measured = timed and frame > warmup
            if measured:
                synchronize()
                started = clock()
            rgb, ir, row = images.read_frame(frame, engine.needs_actual_IR_image)
            rows, detections, _ = engine.step(rgb, ir, sequence, frame)
            if timed:
                synchronize()
            if measured:
                elapsed = clock()-started
                require(math.isfinite(elapsed) and elapsed > 0, 'Each actual latency must be positive and finite')
                durations.append(elapsed)
            # Verification serialization/state accounting are outside the timer.
            require(row == reference['prefix_source_rows'][frame-1], 'Timed/verification source image bytes changed')
            require(set(rows) == set(detections) == {case['id']}, 'Bundled/unregistered inference is not an individual cost')
            values = detections[case['id']]
            packets.update(np.asarray([frame, len(values)], dtype='<i8').tobytes())
            packets.update(np.asarray(values, dtype='<f4').tobytes())
            tracks.update(''.join(mot_line(item) for item in rows[case['id']]).encode())
            sampled = retained_state_bytes(engine)
            largest = {k: max(largest.get(k, 0), v) for k, v in sampled.items()}
            if frame % 8 == 0 or frame == frames:
                resource_check()
        require(packets.hexdigest() == reference['prefix_packet_sha256'][case['id']],
                'Independent detection prefix differs from complete production')
        require(tracks.hexdigest() == reference['prefix_literal_MOT_sha256'][case['id']],
                'Independent literal MOT prefix differs from complete production')
        return dict(sequence=sequence, prefix_frames=frames, warmup_frames=warmup if timed else 0,
            measured_frames=len(durations), durations_seconds=durations,
            packet_sha256=packets.hexdigest(), literal_MOT_sha256=tracks.hexdigest(),
            memory_statistics={name: dict(tracker.statistics) for name, tracker in engine.trackers.items()},
            sampled_retained_state_logical_bytes_max=largest,
            IR_byte_frames_hashed=images.hashed_IR_frames, IR_frames_decoded=images.decoded_IR_frames,
            source_image_bytes_exact=True, complete_production_packets_and_MOT_exact=True,
            warmup_to_timing_state_preserved=bool(timed), simultaneous_conditions=1)
    finally:
        engine.close()


def measure_sequence(engine_factory, images_factory, case, sequence, reference, *,
                     prefix_frames=32, warmup_frames=8, synchronize=lambda: None,
                     clock=time.perf_counter, resource_check=lambda: None):
    """Separate verification and timed passes, each with independent state.

    A reduced range is for engineering tests only; the formal driver fixes 32/8.
    Reading a prefix does not produce a complete Dev/Test observation receipt.
    """
    require(isinstance(prefix_frames, int) and isinstance(warmup_frames, int) and
            0 < warmup_frames < prefix_frames <= 32, 'Valid warmup and prefix coverage required')
    _reference(reference, case, sequence, prefix_frames)
    resource_check()
    verified = _pass(engine_factory, images_factory, case, sequence, reference, prefix_frames,
        warmup_frames, False, synchronize, clock, resource_check)
    timed = _pass(engine_factory, images_factory, case, sequence, reference, prefix_frames,
        warmup_frames, True, synchronize, clock, resource_check)
    require(verified['packet_sha256'] == timed['packet_sha256'] and
            verified['literal_MOT_sha256'] == timed['literal_MOT_sha256'], 'Two independent passes disagree')
    return dict(verification=verified, timing=timed,
                exact_source_and_output_checked_before_and_during_timing=True)


def aggregate_cost(sequences):
    require(sequences and len({r['timing']['sequence'] for r in sequences}) == len(sequences),
            'Unique measured sequences required')
    durations = [v for r in sequences for v in r['timing']['durations_seconds']]
    require(durations and all(math.isfinite(v) and v > 0 for v in durations), 'Actual measured latencies required')
    return dict(measured_frames=len(durations), seconds=sum(durations), FPS=len(durations)/sum(durations),
        mean_latency_ms=1000*statistics.mean(durations), median_latency_ms=1000*statistics.median(durations),
        sampled_retained_state_logical_bytes_max={key: max(r['timing']['sampled_retained_state_logical_bytes_max'][key]
            for r in sequences) for key in sequences[0]['timing']['sampled_retained_state_logical_bytes_max']},
        retained_state_scope='Observed logical arrays/tensor storage after each prefix frame; excludes NN and Python heap, not full-sequence peak')
