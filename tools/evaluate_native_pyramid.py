#!/usr/bin/env python3
"""Native sequential observation producer and separate CPU official evaluator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from tools.native_pyramid_experiment import (
    FIT_CONFIGS, STUDY, context, gpu_settings, load_teacher, predecessor_ready, read, registry,
)
from tools.native_pyramid_stream_tracks import StreamTrackWriter
from tools.run_pseudo_ir_comparison import dump, sha256


def packet(path):
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k] for k in ['frame_ids', 'offsets', 'boxes', 'scores']}
    np.testing.assert_array_equal(result['frame_ids'], np.arange(1, len(result['frame_ids'])+1))
    assert result['offsets'].shape == (len(result['frame_ids'])+1,)
    assert result['offsets'][0] == 0 and result['offsets'][-1] == len(result['boxes'])
    assert (np.diff(result['offsets']) >= 0).all()
    return result


def lengths(plan, protocol, split):
    result = {}
    for sequence in protocol['dev' if split == 'dev' else 'historical_test']:
        path = Path(plan['frozen_detection_sources']['rgb_'+split]) / 'detections' / (sequence+'.npz')
        with np.load(path, allow_pickle=False) as z:
            frames = z['frame_ids']
        np.testing.assert_array_equal(frames, np.arange(1, len(frames)+1))
        result[sequence] = len(frames)
    assert sum(result.values()) == {'dev': 4834, 'test': 15985}[split]
    return result


def read_image(path, scalar=False):
    import cv2
    encoded = Path(path).read_bytes()
    value = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED if scalar else cv2.IMREAD_COLOR)
    if value is None or (scalar and (value.ndim != 2 or value.dtype != np.uint8)):
        raise ValueError('Actual current source image is missing or invalid')
    return value, hashlib.sha256(encoded).hexdigest()


def require_test_freeze(work, specification_path=None):
    freeze_path = work / 'native_pyramid_Test_selection_frozen.json'
    freeze = read(freeze_path)
    assert freeze['status'] == 'FROZEN_FROM_COMPLETE_NATIVE_DEV_BEFORE_TEST'
    assert freeze['study_sha256'] == sha256(STUDY) and freeze['complete_Dev_before_Test']
    if specification_path is not None:
        assert freeze['specification_sha256'] == sha256(specification_path)
    return freeze


def load_modules(rows, study):
    import torch
    from yolox.models.native_pyramid_fusion import NativePyramidFusion
    modules, information = {}, {}
    for row in rows:
        name = row['id']
        if row.get('baseline'):
            assert row['baseline'] == 'fresh_RGB_B1'
            modules[name], information[name] = None, dict(row)
            continue
        path = Path(row['checkpoint'])
        assert sha256(path) == row['checkpoint_sha256']
        arm_path = FIT_CONFIGS / (row['fit_id']+'.json')
        arm = read(arm_path)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        meta = saved['metadata']
        assert (meta['fit_id'], meta['cue'], meta['seed']) == (row['fit_id'], row['cue'], 0)
        assert meta['study_sha256'] == sha256(STUDY) and meta['source_RGB_sha256'] == study['source_RGB_sha256']
        assert meta['fit_config_sha256'] == sha256(arm_path) == row['fit_config_sha256']
        assert saved['completed_epochs'] == row['epoch'] and len(saved['history']) == row['epoch']
        assert all(h['delivery']['complete_all_Train_frames'] and h['frozen_RGB_exact_unchanged'] for h in saved['history'])
        module = NativePyramidFusion(**arm['module_kwargs'])
        module.load_state_dict(saved['module'], strict=True)
        modules[name] = module.cuda().float().eval().requires_grad_(False)
        information[name] = dict(row, fit_metadata=meta, trainable_parameter_count=sum(p.numel() for p in module.parameters()))
    return modules, information


def produce(args):
    import torch
    from tools.native_pyramid_online_inference import OnlineNativeDetector
    from tools.native_pyramid_training_loop import guard_storage
    plan, study, protocol, calibration, work, checkpoint = context()
    registry(work)
    assert predecessor_ready(work)
    specification_path = Path(args.specification)
    specification = read(specification_path)
    assert specification['study_sha256'] == sha256(STUDY) and specification['split'] == args.split
    if args.split == 'test': require_test_freeze(work, specification_path)
    all_lengths = lengths(plan, protocol, args.split)
    assert args.sequence in all_lengths
    physical, uuid = gpu_settings()
    source_path = Path(plan['frozen_detection_sources']['rgb_'+args.split]) / 'detections' / (args.sequence+'.npz')
    canonical = packet(source_path)
    model = load_teacher(checkpoint, training=False)
    modules, information = load_modules(specification['models'], study)
    assert 'NP_source_fresh_RGB' in modules
    engine = OnlineNativeDetector(model, specification['models'], modules, study['inference_input_hw'], calibration)
    folder = Path(specification['stream_directory'])
    complete_path = folder / 'sequence_manifests' / (args.sequence+'.json')
    assert not complete_path.exists()
    guard_storage(work)
    writer = StreamTrackWriter(folder, args.sequence, modules)
    digest, parity = hashlib.sha256(), dict(frames=0, exact_frames=0, equal_count_frames=0,
        maximum_absolute_count_difference=0, different_strict06_count_frames=0, different_strict07_count_frames=0)
    started = time.perf_counter()
    try:
        for frame in range(1, all_lengths[args.sequence]+1):
            rgb_path = Path(plan['data_root']) / ('train' if args.split == 'dev' else 'test') / args.sequence / 'visible' / f'{frame:06d}.jpg'
            rgb, rgb_sha = read_image(rgb_path)
            ir, ir_sha = None, None
            if engine.needs_actual_IR_image:
                ir_path = Path(plan['generated_ir_root']) / args.split / args.sequence / 'infrared_diffv2ir' / f'{frame:06d}.png'
                ir, ir_sha = read_image(ir_path, scalar=True)
            digest.update(json.dumps([args.sequence, frame, rgb_sha, ir_sha], separators=(',', ':')).encode()+b'\n')
            observations, info = engine.step(rgb, ir, args.sequence, frame)
            fresh = observations['NP_source_fresh_RGB']
            lo, hi = canonical['offsets'][frame-1:frame+1]
            old = np.c_[canonical['boxes'][lo:hi], canonical['scores'][lo:hi]].astype(np.float32)
            parity['frames'] += 1
            parity['exact_frames'] += int(np.array_equal(fresh, old))
            parity['equal_count_frames'] += int(len(fresh) == len(old))
            parity['maximum_absolute_count_difference'] = max(parity['maximum_absolute_count_difference'], abs(len(fresh)-len(old)))
            for threshold, field in [(.6, 'different_strict06_count_frames'), (.7, 'different_strict07_count_frames')]:
                parity[field] += int(np.count_nonzero(fresh[:, 4] > threshold) != np.count_nonzero(old[:, 4] > threshold))
            for row in specification['models']:
                if row.get('intervention') == 'unavailable':
                    np.testing.assert_array_equal(observations[row['id']], fresh)
            writer.append(frame, observations)
            if frame % 100 == 0 or frame == all_lengths[args.sequence]:
                writer.flush()
                guard_storage(work)
                dump(folder / 'sequence_manifests' / (args.sequence+'.progress.json'),
                     dict(frame=frame, frames=all_lengths[args.sequence], physical_gpu=physical,
                          models=len(modules), seconds=time.perf_counter()-started))
                print(json.dumps(dict(sequence=args.sequence, frame=frame, models=len(modules), physical_gpu=physical)), flush=True)
        streams = writer.close(all_lengths[args.sequence])
        fresh = streams['NP_source_fresh_RGB']
        for row in specification['models']:
            if row.get('intervention') == 'unavailable':
                assert streams[row['id']]['track_sha256'] == fresh['track_sha256']
                assert streams[row['id']]['detection_packet_sha256'] == fresh['detection_packet_sha256']
        registry(work)
        dump(complete_path, dict(status='COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT',
            sequence=args.sequence, split=args.split, frames=all_lengths[args.sequence], streams=streams,
            specification_sha256=sha256(specification_path), study_sha256=sha256(STUDY),
            source_RGB_sha256=study['source_RGB_sha256'], models=information,
            original_image_bytes_delivery_sha256=digest.hexdigest(), canonical_detection_sha256=sha256(source_path),
            fresh_B1_vs_historical_B4=parity, physical_gpu=physical, GPU_uuid=uuid,
            GT_read=False, future_frame_input=False, previous_outputs_rewritten=False,
            independent_input_batch_size=1, seconds=time.perf_counter()-started,
            peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,
            runtime_scope='Shared bundle production, not per-method end-to-end FPS; separate individual benchmark required.'))
    except Exception:
        writer.abort()
        raise


def produce_canonical(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint = context()
    registry(work)
    specification = read(args.specification)
    assert specification['study_sha256'] == sha256(STUDY)
    if args.split == 'test': require_test_freeze(work, args.specification)
    folder = Path(specification['stream_directory'])
    for sequence, count in lengths(plan, protocol, args.split).items():
        manifest_path = folder / 'sequence_manifests' / (sequence+'.canonical.json')
        if manifest_path.exists():
            previous = read(manifest_path)
            assert previous['specification_sha256'] == sha256(args.specification)
            continue
        path = Path(plan['frozen_detection_sources']['rgb_'+args.split]) / 'detections' / (sequence+'.npz')
        values = packet(path)
        writer = StreamTrackWriter(folder, sequence, ['NP_source_canonical_RGB'])
        try:
            for frame in range(1, count+1):
                lo, hi = values['offsets'][frame-1:frame+1]
                current = np.c_[values['boxes'][lo:hi], values['scores'][lo:hi]].astype(np.float32)
                writer.append(frame, dict(NP_source_canonical_RGB=current))
            streams = writer.close(count)
        except Exception:
            writer.abort()
            raise
        dump(manifest_path, dict(status='COMPLETE_NATIVE_CANONICAL_SOURCE_WITHOUT_GT',
            sequence=sequence, split=args.split, frames=count, streams=streams,
            specification_sha256=sha256(args.specification), canonical_detection_sha256=sha256(path),
            study_sha256=sha256(STUDY), GT_read=False, future_frame_input=False,
            previous_outputs_rewritten=False, GPU_work=False))


def evaluate(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Official evaluation is CPU-only'
    from tools.eval_mft26ir import run_trackeval
    plan, study, protocol, calibration, work, checkpoint = context()
    registration = registry(work)
    case_path = Path(args.config)
    case = read(case_path)
    assert case['split'] == args.split and case['study_sha256'] == sha256(STUDY)
    if args.split == 'test': require_test_freeze(work, case['specification'])
    specification = read(case['specification'])
    assert sha256(case['specification']) == case['specification_sha256']
    folder = Path(case['stream_directory'])
    sequences = protocol['dev' if args.split == 'dev' else 'historical_test']
    expected_lengths = lengths(plan, protocol, args.split)
    coverage, sources = {}, {}
    for sequence in sequences:
        suffix = '.canonical.json' if case.get('canonical') else '.json'
        manifest_path = folder / 'sequence_manifests' / (sequence+suffix)
        producer = read(manifest_path)
        assert producer['frames'] == expected_lengths[sequence] and not producer['GT_read']
        assert producer['split'] == args.split and producer['sequence'] == sequence
        assert producer['specification_sha256'] == case['specification_sha256']
        observation = producer['streams'][case['id']]
        assert observation['frames'] == producer['frames']
        assert sha256(observation['track_path']) == observation['track_sha256']
        coverage[sequence] = dict(frames=producer['frames'], output_rows=observation['track_rows'],
            output_sha256=observation['track_sha256'], detections=observation['detections'],
            detection_packet_sha256=observation['detection_packet_sha256'])
        sources[sequence] = dict(manifest_path=str(manifest_path), manifest_sha256=sha256(manifest_path))
    output = work / 'native_pyramid_evaluation' / args.split / case['id']
    assert not (output / 'metrics.json').exists(), 'Preserve prior official evaluation'
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION', split=args.split,
        config=case, config_sha256=sha256(case_path), study_sha256=sha256(STUDY),
        plan_sha256=study['main_plan_sha256'], sequences=sequences, frames=sum(expected_lengths.values()),
        coverage=coverage, sources=sources, source_versions=registration['source_hashes'],
        GT_inference_input=False, future_frame_input=False, previous_outputs_rewritten=False,
        identity_scope='Sequence-local IDs from1. Original ByteTrack numerical/association behavior retained.',
        runtime_scope='Official evaluation; not an end-to-end speed benchmark.')
    dump(output / 'inference_manifest.json', manifest)
    metrics = run_trackeval(str(folder / case['id'] / 'track_results'), plan['evaluation']['GT_root'],
        str(output), sequences, plan['evaluation']['TrackEval_root'], data_split='train' if args.split == 'dev' else 'test')
    assert len(metrics) == 39
    if case.get('canonical'):
        # Reuse is conditional on the unchanged canonical source reproducing all
        # official fields/CSV. Local ID isolation is not an excuse to change them.
        source_dir = Path(plan['frozen_detection_sources']['rgb_'+args.split])
        previous = read(source_dir / 'metrics.json')
        assert metrics == previous['trackeval']
        assert sha256(output / 'trackeval_output/pedestrian_detailed.csv') == sha256(source_dir / 'trackeval_output/pedestrian_detailed.csv')
    registry(work)
    dump(output / 'metrics.json', dict(status='COMPLETE', trackeval=metrics, manifest=manifest))
    print(json.dumps(dict(status='COMPLETE_NATIVE_OFFICIAL_EVALUATION', case=case['id'], split=args.split)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['produce', 'canonical', 'evaluate'])
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--sequence')
    parser.add_argument('--specification')
    parser.add_argument('--config')
    args = parser.parse_args()
    if args.mode == 'produce':
        assert args.sequence and args.specification
        produce(args)
    elif args.mode == 'canonical':
        assert args.specification
        produce_canonical(args)
    else:
        assert args.config
        evaluate(args)


if __name__ == '__main__':
    main()
