#!/usr/bin/env python3
"""CGR GPU source checks and full frozen inference, without GT or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from tools.correlation_gate_observer import CorrelationGateObserver, SCOPE
from tools.correlation_recovery_inference import CorrelationCaseDetector, LegacyCaseTrackWriter, input_paths, load_adapters
from tools.correlation_recovery_workflow import SPEC, WORKFLOW, audit_source_frame, context, frame_packet, gpu_settings, packet, predecessor_ready, read, registered, require_probes
from tools.native_pyramid_training_loop import guard_storage, state_sha256
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.exp import get_exp
from yolox.models.local_correlation_repair import CorrelationRepair


def load_teacher(parent):
    assert sha256(parent['source_RGB_checkpoint']) == parent['source_RGB_sha256']
    model = get_exp(str(ROOT / 'exps/yolox_mft26ir_visible.py'), None).get_model()
    saved = torch.load(parent['source_RGB_checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    return model.cuda().half().eval().requires_grad_(False)


def read_batch(plan, split, seq, frames, needs_IR, digest=None):
    paths, aux_paths = input_paths(plan, split, seq, frames, needs_IR)
    rgb, ir = [], [] if needs_IR else None
    for index, (frame, path) in enumerate(zip(frames, paths)):
        data = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.dtype == np.uint8
        rgb.append(image)
        aux_sha = None
        if needs_IR:
            aux_data = aux_paths[index].read_bytes()
            aux = cv2.imdecode(np.frombuffer(aux_data, np.uint8), cv2.IMREAD_UNCHANGED)
            assert aux is not None and aux.dtype == np.uint8 and aux.ndim == 2 and aux.shape == image.shape[:2]
            ir.append(aux)
            if digest is not None:
                aux_sha = hashlib.sha256(aux_data).hexdigest()
        if digest is not None:
            digest.update(json.dumps([seq, frame, hashlib.sha256(data).hexdigest(), aux_sha], separators=(',', ':')).encode()+b'\n')
    return rgb, ir


def original_packets(cases, sequence):
    originals = {}
    for row in cases:
        if row.get('source') and row['intervention'] == 'normal':
            source = row['source']
            assert sha256(source['source_Dev_metrics']) == source['source_Dev_metrics_sha256']
            meta = read(source['source_Dev_metrics'])['manifest']['sources'][sequence]
            assert sha256(meta['path']) == meta['sha256']
            originals[row['id']] = packet(meta['path'])
    return originals


def require_test_freeze(work):
    freeze = read(work / 'correlation_recovery_Test_selection_frozen.json')
    assert freeze['workflow_sha256'] == sha256(WORKFLOW)
    assert freeze['source_specification_sha256'] == sha256(SPEC)
    assert freeze['complete_all13roles_12cases_Dev_before_Test']
    assert freeze['full_Dev_report_sha256'] == sha256(work / 'correlation_recovery_full_Dev_proof.json')
    return freeze


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['probe', 'dev', 'test'])
    parser.add_argument('--sequence')
    args = parser.parse_args()
    plan, spec, protocol, parent, calibration, work = context()
    workflow = registered()
    assert predecessor_ready(work), 'Existing fullTrain/legacy/native work retains GPU priority'
    if args.phase != 'probe':
        require_probes(work)
    physical, uuid = gpu_settings()
    cv2.setNumThreads(1)
    guard_storage(work)
    if args.phase == 'test':
        require_test_freeze(work)
        assert args.sequence in protocol['historical_test']
    else:
        assert args.sequence is None
    split = 'test' if args.phase == 'test' else 'dev'
    sequences = [args.sequence] if args.phase == 'test' else protocol['dev']
    cases = list(spec['cases'])
    modules, information = load_adapters(cases, spec['repair_plan_sha256'], parent['source_RGB_sha256'], 'cuda')
    if args.phase == 'probe':
        for cue in ['rgb', 'gray', 'ir']:
            for radius in [0, 2]:
                name = f'CRB_zero_{cue}_r{radius}'
                cases.append(dict(id=name, intervention='normal', source=dict(cue=cue, output_policy='raw'), zero_adapter=True))
                modules[name] = CorrelationRepair(radius=radius).cuda().float().eval().requires_grad_(False)
    model = load_teacher(parent)
    before = state_sha256(model.state_dict()) if args.phase == 'probe' else None
    engine = CorrelationCaseDetector(model, cases, modules, parent['inference_input_hw'], calibration)
    counters = {row['id']: [0] for row in cases}
    base = work / 'correlation_recovery_formal'
    folder = (base / 'engineering' / ('gpu'+physical) if args.phase == 'probe' else work / ('correlation_recovery_'+split))
    manifest_dir = folder / 'sequence_manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fresh_key = spec['roles']['fresh_B4']
    off_key = spec['roles']['unavailable']
    sequence_proofs = {}
    for sequence in sequences:
        canonical_path = Path(plan['frozen_detection_sources']['rgb_'+split]) / 'detections' / (sequence+'.npz')
        canonical = packet(canonical_path)
        frames_total = len(canonical['frame_ids'])
        np.testing.assert_array_equal(canonical['frame_ids'], np.arange(1, frames_total+1))
        limit = 8 if args.phase == 'probe' else frames_total
        originals = original_packets(spec['cases'], sequence) if split == 'dev' else {}
        path = manifest_dir / (sequence+'.json')
        assert not path.exists()
        observer = CorrelationGateObserver()
        for name, module in modules.items():
            if module is not None:
                observer.attach(name, module)
        writer = None if args.phase == 'probe' else LegacyCaseTrackWriter(folder, sequence, modules, counters)
        digest, drift, checked, started = hashlib.sha256(), [], 0, time.perf_counter()
        try:
            for first in range(1, limit+1, 4):
                frames = list(range(first, min(first+4, limit+1)))
                rgb, ir = read_batch(plan, split, sequence, frames, engine.needs_IR, digest)
                old = [frame_packet(canonical, frame) for frame in frames]
                outputs, fresh = engine.batch(rgb, ir, old)
                for index, frame in enumerate(frames):
                    if args.phase == 'probe':
                        np.testing.assert_array_equal(fresh[index], old[index])
                    else:
                        change = audit_source_frame(fresh[index], old[index], frame, frames_total)
                        if change is not None:
                            drift.append(change)
                    for name, original in originals.items():
                        np.testing.assert_array_equal(outputs[name][index], frame_packet(original, frame))
                    np.testing.assert_array_equal(outputs[off_key][index], fresh[index])
                    if args.phase == 'probe':
                        for row in cases:
                            if row.get('zero_adapter'):
                                np.testing.assert_array_equal(outputs[row['id']][index], fresh[index])
                    else:
                        writer.append(frame, {key: value[index] for key, value in outputs.items()})
                    checked += 1
                if frames[-1] % 100 < 4 or frames[-1] == limit:
                    if writer is not None:
                        writer.flush()
                    guard_storage(work)
                    progress = dict(sequence=sequence, frame=frames[-1], frames=limit,
                        phase=args.phase, physical_gpu=physical, cases=len(cases), seconds=time.perf_counter()-started)
                    dump(manifest_dir / (sequence+'.progress.json'), progress)
                    print(json.dumps(progress), flush=True)
            streams = writer.close(limit) if writer is not None else {}
            if streams:
                for field in ['track_sha256', 'detection_packet_sha256']:
                    assert streams[off_key][field] == streams[fresh_key][field]
                if split == 'dev':
                    for row in spec['cases']:
                        if row['id'] not in originals:
                            continue
                        expected = read(row['source']['source_Dev_metrics'])['manifest']['coverage'][sequence]
                        assert streams[row['id']]['track_sha256'] == expected['output_sha256']
            gates = {name: observer.summary(name) for name in observer.rows}
            assert all(all(r['frames'] == limit for r in rows) for rows in gates.values())
            result = dict(status='COMPLETE_CGR_GPU_PROBE' if args.phase == 'probe' else 'COMPLETE_CGR_CURRENT_IMAGE_STREAMS_WITHOUT_GT',
                phase=args.phase, split=split, sequence=sequence, frames=checked, streams=streams,
                workflow_sha256=sha256(WORKFLOW), source_specification_sha256=sha256(SPEC),
                models=information, source_RGB_sha256=parent['source_RGB_sha256'],
                canonical_proposal_path=str(canonical_path), canonical_proposal_sha256=sha256(canonical_path),
                original_image_bytes_delivery_sha256=digest.hexdigest(), source_tail_differences=drift,
                original_normal_paired_Dev_prediction_streams_exact=len(originals),
                unavailable_full_detections_and_tracks_exact_to_fresh=True,
                zero_full_head_and_NMS_cases=6 if args.phase == 'probe' else 0,
                gate_statistics=gates, gate_statistics_scope=SCOPE, physical_gpu=physical, GPU_uuid=uuid,
                GT_read=False, future_frame_information_used=False, previous_outputs_rewritten=False,
                independent_sample_batch_size=4, fixed_seed=0, additional_seed_work=False,
                ID_scope='Original per-case counters continue across Dev videos; formal Test starts at1 per video. No cross-video tracking/evidence state.',
                seconds=time.perf_counter()-started, peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,
                runtime_scope='Shared12case production including diagnostics, checksums and source checks; not per-method FPS.')
            dump(path, result)
            sequence_proofs[sequence] = dict(path=str(path), sha256=sha256(path))
        except Exception:
            if writer is not None:
                writer.abort()
            raise
        finally:
            observer.close()
    registered()
    if args.phase == 'probe':
        assert state_sha256(model.state_dict()) == before
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        proof_path = folder / 'complete.json'
        assert not proof_path.exists()
        dump(proof_path, dict(status='COMPLETE_CGR_ACTUAL_GPU_SOURCE_ZERO_MISSING_AND_ORIGINAL_PREFIX',
            workflow_sha256=sha256(WORKFLOW), physical_gpu=physical, GPU_uuid=uuid,
            sequences=sequence_proofs, frames_each=8, original_Dev_cases_checked=8,
            frozen_RGB_weights_and_BN_unchanged=True, zero_branches_each=6,
            source_missing_and_zero_full_head_NMS_exact=True, GT_read=False,
            fixed_seed=0, additional_seed_work=False, goal_met=False, supported_innovations=0))


if __name__ == '__main__':
    main()
