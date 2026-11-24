"""Independent source/ByteTrack/memory comparisons for the future F4 GPU probe.

This component can run on artificial CPU fixtures. It grants no parent, CUDA,
complete-sequence, performance or accuracy qualification.
"""
import hashlib
import json

import numpy as np

from tools.fusion_memory_design import make_case
from tools.fusion_memory_evaluation import read, sha
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_online import discard_consumed_observations
from tools.fusion_memory_online import visual_routes
from tools.fusion_memory_parent_io import require
from tools.fusion_memory_runtime_sources import literal_MOT_prefix
from tools.native_pyramid_stream_tracks import IndependentByteTrack, mot_line
from tools.probe_fusion_memory import state_digest
from yolox.tracker.fusion_memory_tracker import FusionMemoryConfig, FusionMemoryTracker


class EngineeringMismatch(ValueError):
    def __init__(self, kind, sequence, frame, case, **details):
        self.details = dict(kind=kind, sequence=sequence, frame=frame, case=case, **details)
        super().__init__(json.dumps(self.details, sort_keys=True))


def memory_digest(tracker):
    digest = hashlib.sha256()
    for group in [tracker.core.tracked_stracks, tracker.core.lost_stracks, tracker.core.removed_stracks]:
        for track in group:
            for name in ['rgb_memory', 'aux_memory', 'fused_memory']:
                value = getattr(track, name)
                digest.update(b'NONE' if value is None else value.tobytes())
            digest.update(repr(track.memory_last_frame).encode())
            digest.update(repr(track.memory_last_reliability).encode())
    return digest.hexdigest()


def engineering_cases(plan):
    """Fixed normal cases plus missing/quarter-image roll, never Dev/Test tuning."""
    cases = list(plan['normal_cases'])
    primary = plan['role_visual_keys']['primary_IR']
    ir = [c for c in cases if c['visual_key'] == primary]
    require(len(ir) == 9 and all(c['intervention'] == 'normal' and c['minimum_reliability'] == .2 for c in ir),
            'All nine original primary memory cases required for engineering')
    for intervention in ['IR_unavailable', 'IR_image_spatial_roll']:
        cases.extend(make_case(c['visual'], c['profile'], c['appearance_weight'], intervention=intervention) for c in ir)
    require(len({c['id'] for c in cases}) == len(cases), 'Unique engineering cases required')
    return cases


def original_Dev_prefixes(workflow, cases, frames=32):
    plan = workflow.parents.plan
    selection = workflow.parents.evidence['selection']
    chosen = next(r for r in selection['records'] if r['family'] == selection['chosen_family'])
    originals, pins, source_rows = {}, {}, {}
    for sequence in ['BT-005', 'SN-013']:
        originals[sequence], pins[sequence] = {}, {}
        for case in cases:
            if case['profile'] != 'byte':
                continue
            if case['intervention'] == 'IR_unavailable':
                visual = plan['visuals'][plan['role_visual_keys']['fresh_RGB']]
            elif case['intervention'] == 'IR_image_spatial_roll':
                visual = chosen['interventions']['IR_image_spatial_roll']
            else:
                visual = case['visual']
            receipt = visual['source_receipts'][sequence]
            require(sha(receipt['source_manifest']) == receipt['source_manifest_sha256'], 'Original Dev producer changed')
            producer = read(receipt['source_manifest'])
            require(producer['split'] == 'dev' and producer['sequence'] == sequence and producer['GT_read'] is False and
                    producer['frames'] == LENGTHS['dev'][sequence] and producer['future_frame_input'] is False and
                    producer['previous_outputs_rewritten'] is False and producer['status'] in [
                        'COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT',
                        'COMPLETE_SPATIAL_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT'],
                    'Use the already-qualified complete original Dev source')
            stream = producer['streams'][visual['model']['id']]
            require(sha(stream['track_path']) == stream['track_sha256'] == receipt['track_sha256'],
                    'Original complete Dev MOT bytes changed')
            originals[sequence][case['id']] = literal_MOT_prefix(stream['track_path'], frames)
            pins[sequence][case['id']] = dict(source_manifest=receipt['source_manifest'],
                source_manifest_sha256=receipt['source_manifest_sha256'], track_path=stream['track_path'],
                track_sha256=stream['track_sha256'], prefix_literal_MOT_sha256=originals[sequence][case['id']])
            if producer['status'] == 'COMPLETE_SPATIAL_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT':
                require(frames == 32 and len(producer['prefix32_source_rows']) == 32, 'Original stored prefix length differs')
                previous = source_rows.setdefault(sequence, producer['prefix32_source_rows'])
                require(previous == producer['prefix32_source_rows'], 'Original stored source-image prefixes differ')
    return originals, pins, source_rows


def unchanged_Dev_prefixes(pins):
    for values in pins.values():
        for pin in values.values():
            require(sha(pin['source_manifest']) == pin['source_manifest_sha256'] and
                    sha(pin['track_path']) == pin['track_sha256'], 'Original Dev prefix source changed during engineering')


def compare_prefix(composed, reference, images, cases, original_zero_MOT, *, frames=32,
                   expected_source_rows=None, original_zero_packets=None, resource_check=lambda: None):
    """Compare independent NN sources, original ByteTrack and current-ROI replay.

    Nonzero replay uses captured actual current ROI packets; it independently
    checks association/retained state, not a second ROI implementation. Known-plane
    coordinate checks and original-source NN checks establish separate contracts.
    """
    try:
        return _compare_prefix(composed, reference, images, cases, original_zero_MOT, frames=frames,
            expected_source_rows=expected_source_rows, original_zero_packets=original_zero_packets,
            resource_check=resource_check)
    finally:
        composed.close()


def _compare_prefix(composed, reference, images, cases, original_zero_MOT, *, frames,
                    expected_source_rows, original_zero_packets, resource_check):
    sequence = images.sequence
    require(cases == composed.cases and composed.sequence is None and images.last_frame == 0 and
            1 <= frames <= images.frames, 'Fresh, matching engineering sequence state required')
    require(reference.model is not composed.detector.model and reference.sequence is None,
            'Independent RGB model and fresh reference state required')
    require(reference.rows == [r['model'] for r in composed.routes.values()] and
            set(reference.modules) == set(composed.routes), 'Independent reference routes differ')
    left = {id(m) for m in composed.detector.modules.values() if m is not None}
    require(not left.intersection(id(m) for m in reference.modules.values() if m is not None),
            'Reference/composed fusion modules must be independently loaded')
    zeros = {c['id'] for c in cases if c['profile'] == 'byte'}
    require(set(original_zero_MOT) == zeros, 'Every zero case needs its original Dev literal MOT prefix')
    byte = {route: IndependentByteTrack() for route in composed.routes}
    replay = {c['id']: FusionMemoryTracker(FusionMemoryConfig(appearance_weight=c['appearance_weight'],
        feature_source=c['feature_source'], quality_mode='provided' if c['quality_policy'] == 'attention_proxy' else 'uniform',
        minimum_reliability=c['minimum_reliability'])) for c in cases if c['appearance_weight']}
    mot = {c['id']: hashlib.sha256() for c in cases}
    packets = {c['id']: hashlib.sha256() for c in cases}
    per_frame, current_packets, restored, feature_rows, source_rows = [], {}, [], {}, []
    active_frame = 0
    try:
        for key, capture in composed.captures.items():
            take = capture.take_packet
            def observed(sequence_name, frame, detections, *, key=key, take=take):
                packet = take(sequence_name, frame, detections)
                metadata = packet['metadata']
                require(metadata['sequence'] == sequence and metadata['frame'] == active_frame and
                        metadata['quality_is_calibrated_reliability'] is False, 'Current ROI metadata/proxy source differs')
                current_packets[key] = packet
                stats = feature_rows.setdefault(str(key), dict(rgb_ROIs=0, available_aux_ROIs=0, valid_fused_ROIs=0,
                    fused_differs_from_RGB_ROIs=0, feature_source=key[1], quality_policy=key[2], route=key[0]))
                stats['rgb_ROIs'] += len(packet['rgb'])
                available = packet['auxiliary_available']
                stats['available_aux_ROIs'] += int(available.sum())
                if 'fused' in packet:
                    valid = np.asarray(metadata['descriptor_validity']['fused'], dtype=bool) & available
                    stats['valid_fused_ROIs'] += int(valid.sum())
                    stats['fused_differs_from_RGB_ROIs'] += int(np.count_nonzero(
                        np.any(packet['fused'] != packet['rgb'], axis=1) & valid))
                return packet
            restored.append((capture, take))
            capture.take_packet = observed
        for frame in range(1, frames+1):
            active_frame = frame
            current_packets.clear()
            rgb, ir, source = images.read_frame(frame, composed.needs_actual_IR_image)
            if expected_source_rows is not None:
                require(source == expected_source_rows[frame-1], 'Actual current image bytes differ from original stored prefix')
            source_rows.append(source)
            reference_packets, _ = reference.step(rgb, ir, sequence, frame)
            original_rows = {route: tracker.update(reference_packets[route], frame) for route, tracker in byte.items()}
            rows, actual, _ = composed.step(rgb, ir, sequence, frame)
            fingerprints = {}
            for case in cases:
                name, route = case['id'], composed.case_routes[case['id']]
                expected, found = reference_packets[route], actual[name]
                expected_hash = hashlib.sha256(np.asarray(expected, dtype='<f4').tobytes()).hexdigest()
                actual_hash = hashlib.sha256(np.asarray(found, dtype='<f4').tobytes()).hexdigest()
                if expected.shape != found.shape or expected_hash != actual_hash:
                    raise EngineeringMismatch('DETECTION_PACKET', sequence, frame, name,
                        expected_sha256=expected_hash, actual_sha256=actual_hash,
                        expected_shape=list(expected.shape), actual_shape=list(found.shape),
                        maximum_absolute_difference=float(np.max(np.abs(expected-found)))
                            if expected.shape == found.shape and expected.size else None)
                packets[name].update(np.asarray([frame, len(found)], dtype='<i8').tobytes())
                packets[name].update(np.asarray(found, dtype='<f4').tobytes())
                actual_MOT = ''.join(mot_line(row) for row in rows[name])
                mot[name].update(actual_MOT.encode())
                if case['profile'] == 'byte':
                    expected_rows, expected_core = original_rows[route], byte[route].tracker
                else:
                    tracker = replay[name]
                    feature_packet = current_packets[composed.case_captures[name]]
                    tracks = tracker.update(sequence, frame, expected, feature_packet)
                    expected_rows = [(frame, t.tlwh.copy(), t.track_id, t.score) for t in tracks if t.tlwh[2]*t.tlwh[3] > 100]
                    expected_core = tracker.core
                    if memory_digest(tracker) != memory_digest(composed.trackers[name]) or tracker.statistics != composed.trackers[name].statistics:
                        raise EngineeringMismatch('RETAINED_MEMORY_OR_COST', sequence, frame, name)
                    discard_consumed_observations(tracker)
                if state_digest(expected_core) != state_digest(composed.trackers[name].core):
                    raise EngineeringMismatch('KF_ID_TRACK_STATE', sequence, frame, name)
                if ''.join(mot_line(row) for row in expected_rows) != actual_MOT:
                    raise EngineeringMismatch('CURRENT_MOT', sequence, frame, name)
                fingerprints[name] = actual_hash
            per_frame.append(dict(frame=frame, detection_packet_sha256=fingerprints,
                all_independent_source_packets_KF_ID_and_memories_exact=True))
            current_packets.clear()
            if frame % 8 == 0 or frame == frames:
                resource_check()
        for name in zeros:
            if mot[name].hexdigest() != original_zero_MOT[name]:
                raise EngineeringMismatch('ORIGINAL_ZERO_MOT_PREFIX', sequence, frames, name,
                    expected_sha256=original_zero_MOT[name], actual_sha256=mot[name].hexdigest())
            if original_zero_packets is not None and packets[name].hexdigest() != original_zero_packets[name]:
                raise EngineeringMismatch('ORIGINAL_ZERO_PACKET_PREFIX', sequence, frames, name,
                    expected_sha256=original_zero_packets[name], actual_sha256=packets[name].hexdigest(),
                    actual_per_frame_packet_sha256={r['frame']: r['detection_packet_sha256'][name] for r in per_frame},
                    actual_MOT_sha256=mot[name].hexdigest(), original_MOT_sha256=original_zero_MOT[name])
        return dict(status='F4_ENGINEERING_PREFIX_COMPARISONS_PASSED_NOT_GPU_AUTHORIZATION', sequence=sequence,
            prefix_frames=frames, declared_source_frames=images.frames, per_frame=per_frame,
            source_rows=source_rows, packet_sha256={n: p.hexdigest() for n, p in packets.items()},
            MOT_sha256={n: p.hexdigest() for n, p in mot.items()}, feature_rows=feature_rows,
            memory_statistics={name: dict(t.statistics) for name, t in composed.trackers.items()},
            retained_fused_vectors={name: sum(t.fused_memory is not None for t in
                tracker.core.tracked_stracks+tracker.core.lost_stracks) for name, tracker in composed.trackers.items()},
            independent_NN_source=True, actual_current_ROI_replay=True,
            separate_ROI_implementation=False, full_Dev_Test_coverage=False,
            GPU_authorization_established_here=False, new_accuracy=False)
    finally:
        for capture, take in restored:
            capture.take_packet = take


def require_active_primary_memory(plan, results):
    primary = plan['role_visual_keys']['primary_IR']
    active = [c for c in plan['normal_cases'] if c['visual_key'] == primary and
              c['profile'] == 'fused_uniform' and c['appearance_weight'] == .05]
    require(len(active) == 1, 'Original primary fixed fused-uniform engineering case required')
    name = active[0]['id']
    routes, _ = visual_routes(active)
    route = next(iter(routes))
    require(any(r['retained_fused_vectors'][name] > 0 and r['memory_statistics'][name]['changed_edges'] > 0 and any(
        f['route'] == route and f['feature_source'] == 'fused' and f['quality_policy'] == 'uniform' and
        f['fused_differs_from_RGB_ROIs'] > 0 for f in r['feature_rows'].values()) for r in results),
            'Actual primary fused memory/positive cost path was not exercised; no GPU qualification')
