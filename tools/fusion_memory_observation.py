"""Complete fixed-protocol frame delivery and causal F4 observation receipts.

Caller owns model qualification, registration, release, CUDA and evaluation.
Train prefixes are engineering only; formal Dev/Test prefixes are rejected.
"""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from tools.fusion_memory_parent_io import require
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.native_pyramid_stream_tracks import mot_line

LENGTHS = dict(train={'BT-001': 3000, 'BT-003': 243, 'MSK-002': 1754, 'PF-001': 15000, 'SN-001': 684, 'SN-015': 6570},
    dev={'BT-005': 1834, 'SN-013': 3000},
    test={'BT-002': 493, 'BT-004': 763, 'MSK-003': 629, 'PF-002': 3000, 'SN-009': 4500, 'SN-011': 4500, 'SN-014': 2100})


class FixedSequenceImages:
    def __init__(self, plan, split, sequence, frames=None):
        require(split in LENGTHS and sequence in LENGTHS[split], 'Original ProtocolV2 split/sequence required')
        complete = LENGTHS[split][sequence]
        frames = complete if frames is None else frames
        require(isinstance(frames, int) and 1 <= frames <= complete, 'Valid registered frame coverage required')
        require(split == 'train' or frames == complete, 'Formal Dev/Test requires the complete sequence')
        self.plan, self.split, self.sequence, self.frames = plan, split, sequence, frames
        self.last_frame, self.complete_source_frames = 0, complete
        self.decoded_IR_frames, self.hashed_IR_frames = 0, 0

    def read_frame(self, frame, needs_ir):
        require(frame == self.last_frame+1 and frame <= self.frames, 'Every current frame must be read once in order')
        rgb_path = Path(self.plan['data_root'])/('test' if self.split == 'test' else 'train')/self.sequence/'visible'/f'{frame:06d}.jpg'
        ir_path = Path(self.plan['generated_ir_root'])/self.split/self.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
        rgb_bytes, ir_bytes = rgb_path.read_bytes(), ir_path.read_bytes()
        rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
        require(rgb is not None and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3,
                'Current fixed RGB image cannot be decoded')
        ir = None
        if needs_ir:
            ir = cv2.imdecode(np.frombuffer(ir_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            require(ir is not None and ir.dtype == np.uint8 and ir.ndim == 2 and ir.shape == rgb.shape[:2],
                    'Current fixed scalar IR image cannot be decoded/aligned')
            self.decoded_IR_frames += 1
        self.hashed_IR_frames += 1
        self.last_frame = frame
        return rgb, ir, [self.sequence, frame, hashlib.sha256(rgb_bytes).hexdigest(), hashlib.sha256(ir_bytes).hexdigest()]


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def observe_sequence(engine, images, cases, directory, expected_image_delivery_sha256, provenance, *, resource_check=lambda: None):
    """One complete declared range; no GT, past-output changes or array saves."""
    directory = Path(directory)
    sequence, split, count = images.sequence, images.split, images.frames
    names = [c['id'] for c in cases]
    require(names == [c['id'] for c in engine.cases], 'Observation engine differs from registered cases/order')
    require(images.last_frame == 0 and engine.sequence is None, 'Start with fresh image and online sequence state')
    complete_path = directory/'sequence_manifests'/(sequence+'.json')
    require(not complete_path.exists(), 'Preserve existing complete observations')
    writer = FusionMemoryStreamWriter(directory, sequence, names)
    source, prefixes, prefix_tracks = hashlib.sha256(), {n: hashlib.sha256() for n in names}, {n: hashlib.sha256() for n in names}
    source_rows, frame, started = [], 0, time.monotonic()
    try:
        resource_check()
        for frame in range(1, count+1):
            rgb, ir, source_row = images.read_frame(frame, engine.needs_actual_IR_image)
            source.update(json.dumps(source_row, separators=(',', ':')).encode()+b'\n')
            rows, packets, _ = engine.step(rgb, ir, sequence, frame)
            writer.append(frame, rows, packets)
            if frame <= 32:
                source_rows.append(source_row)
                for name, packet in packets.items():
                    prefixes[name].update(np.asarray([frame, len(packet)], dtype='<i8').tobytes())
                    prefixes[name].update(np.asarray(packet, dtype='<f4').tobytes())
                    prefix_tracks[name].update(''.join(mot_line(row) for row in rows[name]).encode())
            if frame % 100 == 0 or frame == count:
                resource_check()
                write_json(directory/'sequence_manifests'/(sequence+'.progress.json'),
                    dict(frame=frame, frames=count, split=split, sequence=sequence, cases=len(cases),
                         seconds=time.monotonic()-started, GT_read=False, complete_receipt=False))
        require(source.hexdigest() == expected_image_delivery_sha256, 'Full source image-byte delivery differs from fixed anchor')
        streams = writer.close(count)
        result = dict(status='COMPLETE_F4_TRAIN_PREFIX_ENGINEERING_OBSERVATIONS' if split == 'train' else
            'COMPLETE_F4_CAUSAL_SEQUENCE_OBSERVATIONS', sequence=sequence, split=split, frames=count,
            complete_source_sequence_frames=images.complete_source_frames,
            formal_Dev_Test_sequence_coverage=split in ['dev', 'test'], provenance=provenance,
            streams=streams, cases=cases, original_image_bytes_delivery_sha256=source.hexdigest(),
            prefix_source_rows=source_rows, prefix_frames=min(32, count),
            prefix_packet_sha256={n: v.hexdigest() for n, v in prefixes.items()},
            prefix_literal_MOT_sha256={n: v.hexdigest() for n, v in prefix_tracks.items()},
            memory_statistics={n: dict(t.statistics) for n, t in engine.trackers.items()},
            actual_IR_byte_frames_hashed=images.hashed_IR_frames, actual_IR_frames_decoded=images.decoded_IR_frames,
            unused_IR_bytes_hashed_for_source_provenance=True, current_image_cache=False,
            GT_read=False, future_frame_input=False, previous_outputs_rewritten=False,
            feature_or_detection_disk_cache=False, independent_input_batch_size=1,
            formal_authorization_established_here=False, seconds=time.monotonic()-started)
        resource_check()
        write_json(complete_path, result, exclusive=True)
        return result
    except Exception as error:
        write_json(directory/'sequence_manifests'/(sequence+f'.failure_{time.time_ns()}.json'),
            dict(error=repr(error), attempted_frame=frame, emitted_frames=writer.frames, declared_frames=count,
                 split=split, sequence=sequence, complete_receipt=False, GT_read=False, prior_outputs_retained=True), exclusive=True)
        raise
    finally:
        writer.abort()
        engine.close()
