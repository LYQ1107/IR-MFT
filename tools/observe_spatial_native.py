#!/usr/bin/env python3
"""Spatial B1 observations and separate CPU official evaluation.

Actual image and fresh-source hashes must match the completed native campaign.
No Test model choice is made here; a frozen Dev-only specification is required.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from tools.evaluate_native_pyramid import lengths, packet, read_image
from tools.native_pyramid_experiment import gpu_settings, load_teacher, read
from tools.native_pyramid_stream_tracks import StreamTrackWriter
from tools.neck72_runtime import require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.spatial_native_study import STUDY, REGISTRY, arm_path, context, predecessor_ready, registered, validate_history

FRESH = dict(id='NS_source_fresh_RGB', cue='rgb', seed=0, baseline='fresh_RGB_B1')
CANONICAL = dict(id='NS_source_canonical_RGB', cue='rgb', seed=0, baseline='historical_canonical_RGB_B4')


def require_test_freeze(base, specification_path):
    freeze = read(Path(base)/'Test_selection_frozen.json')
    assert freeze['status'] == 'FROZEN_SPATIAL_ALL_COMPLETE_DEV_BEFORE_TEST'
    assert freeze['study_sha256'] == sha256(STUDY) and freeze['complete_Dev_before_Test']
    assert not freeze['Test_accuracy_used_for_selection']
    assert freeze['specification_sha256'] == sha256(specification_path)
    assert freeze['complete_Dev_sha256'] == sha256(Path(base)/'complete_Dev.json')
    return freeze


def source_anchor(work, split, sequence):
    phase = 'dev_normal_06' if split == 'dev' else 'test_frozen'
    path = Path(work)/'native_pyramid_specs'/(phase+'.json')
    specification = read(path)
    assert specification['study_sha256'] == sha256(ROOT/'configs/perceptual_native_pyramid_study_20261124.json')
    assert specification['split'] == split
    if split == 'test':
        freeze = read(Path(work)/'native_pyramid_Test_selection_frozen.json')
        assert freeze['specification_sha256'] == sha256(path)
    manifest_path = Path(specification['stream_directory'])/'sequence_manifests'/(sequence+'.json')
    value = read(manifest_path)
    assert value['specification_sha256'] == sha256(path) and not value['GT_read']
    assert value['status'] == 'COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT'
    assert value['sequence'] == sequence and value['split'] == split
    return value, dict(path=str(manifest_path), sha256=sha256(manifest_path))


def load_modules(rows, study, base, registration, device, observer=None):
    import torch
    from tools.spatial_native_fusion import SpatialNativeFusion
    modules, information = {}, {}
    for row in rows:
        name = row['id']
        if row.get('baseline'):
            assert row == FRESH
            modules[name], information[name] = None, dict(row)
            continue
        path = Path(row['checkpoint'])
        assert path == Path(base)/'fits'/row['fit_id']/f"epoch_{row['epoch']:02d}.pt"
        assert sha256(path) == row['checkpoint_sha256']
        config_path = arm_path(base, row['fit_id'])
        arm = read(config_path)
        assert sha256(config_path) == row['fit_config_sha256']
        assert row['cue'] == arm['cue'] and row['variant'] == arm['variant'] and row['seed'] == 0
        assert row.get('geometry_fixed', False) == arm['geometry_fixed'] and row['mode'] == arm['mode']
        saved = torch.load(path, map_location='cpu', weights_only=False)
        metadata = validate_history(saved, row['epoch'], arm)
        assert metadata['fit_config_sha256'] == sha256(config_path)
        assert metadata['registry_sha256'] == sha256(Path(base)/REGISTRY)
        assert metadata['source_hashes'] == registration['source_hashes']
        assert metadata['source_RGB_sha256'] == study['source_RGB_sha256']
        module = SpatialNativeFusion(**arm['module_kwargs'])
        module.load_state_dict(saved['module'], strict=True)
        modules[name] = module.to(device).float().eval().requires_grad_(False)
        if observer is not None:
            observer.attach(name, modules[name])
        information[name] = dict(row, fit_metadata=metadata,
            total_auxiliary_parameters=sum(p.numel() for p in module.parameters()))
    assert len(modules) == len(rows)
    return modules, information


def full_precision_selection(csv_path, split, protocol):
    with Path(csv_path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert len(reader.fieldnames) == 276
        rows = list(reader)
    sequences = protocol['dev' if split == 'dev' else 'historical_test']
    assert len(rows) == len(sequences)+1 and {r['seq'] for r in rows} == set(sequences)|{'COMBINED'}
    row = next(r for r in rows if r['seq'] == 'COMBINED')
    assert float(row['CLR_Frames']) == {'dev': 4834, 'test': 15985}[split]
    result = {k: 100*float(row[k+'___AUC' if k != 'IDF1' else k]) for k in ['HOTA', 'AssA', 'IDF1']}
    assert all(math.isfinite(v) and 0 <= v <= 100 for v in result.values())
    return result


def produce(args):
    import torch
    from tools.native_pyramid_online_inference import OnlineNativeDetector
    from tools.spatial_native_observer import SpatialObserver
    plan, study, protocol, calibration, work, checkpoint, base = context()
    registration = registered(work)
    assert predecessor_ready(work)
    specification_path = Path(args.specification)
    specification = read(specification_path)
    assert specification['study_sha256'] == sha256(STUDY) and specification['split'] == args.split
    if args.split == 'test':
        require_test_freeze(base, specification_path)
    expected = lengths(plan, protocol, args.split)
    assert args.sequence in expected
    reference, anchor_reference = source_anchor(work, args.split, args.sequence)
    assert reference['frames'] == expected[args.sequence]
    require_host_memory(8*2**30, maximum_workers=1)
    require_storage(work, growth_bytes=int(.5*2**30))
    physical, uuid = gpu_settings()
    teacher = load_teacher(checkpoint, training=False)
    observer = SpatialObserver()
    modules, information = load_modules(specification['models'], study, base, registration, 'cuda', observer)
    assert FRESH['id'] in modules
    engine = OnlineNativeDetector(teacher, specification['models'], modules, study['inference_input_hw'], calibration)
    assert engine.needs_actual_IR_image, 'The full source image anchor includes actual IR, not an absent placeholder'
    folder = Path(specification['stream_directory'])
    manifest_path = folder/'sequence_manifests'/(args.sequence+'.json')
    assert not manifest_path.exists()
    writer = StreamTrackWriter(folder, args.sequence, modules)
    source_digest = hashlib.sha256()
    prefixes = {key: hashlib.sha256() for key in modules}
    prefix_rows, prefix_tracks = [], {key: [] for key in modules}
    started = time.perf_counter()
    try:
        for frame in range(1, expected[args.sequence]+1):
            rgb_path = Path(plan['data_root'])/('train' if args.split == 'dev' else 'test')/args.sequence/'visible'/f'{frame:06d}.jpg'
            ir_path = Path(plan['generated_ir_root'])/args.split/args.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
            rgb, rgb_sha = read_image(rgb_path)
            ir, ir_sha = read_image(ir_path, scalar=True)
            source = [args.sequence, frame, rgb_sha, ir_sha]
            source_digest.update(json.dumps(source, separators=(',', ':')).encode()+b'\n')
            values, _ = engine.step(rgb, ir, args.sequence, frame)
            writer.append(frame, values)
            if frame <= 32:
                prefix_rows.append(source)
                for name, array in values.items():
                    prefixes[name].update(np.asarray([frame, len(array)], dtype='<i8').tobytes())
                    prefixes[name].update(np.asarray(array, dtype='<f4').tobytes())
            if frame % 100 == 0 or frame == expected[args.sequence]:
                writer.flush()
                require_storage(work)
                dump(folder/'sequence_manifests'/(args.sequence+'.progress.json'),
                    dict(frame=frame, frames=expected[args.sequence], physical_gpu=physical,
                        models=len(modules), seconds=time.perf_counter()-started))
        streams = writer.close(expected[args.sequence])
        assert source_digest.hexdigest() == reference['original_image_bytes_delivery_sha256']
        for field in ['track_sha256', 'detection_packet_sha256']:
            assert streams[FRESH['id']][field] == reference['streams']['NP_source_fresh_RGB'][field]
        for row in specification['models']:
            if row.get('intervention') == 'unavailable':
                for field in ['track_sha256', 'detection_packet_sha256']:
                    assert streams[row['id']][field] == streams[FRESH['id']][field]
        for name, stream in streams.items():
            with Path(stream['track_path']).open() as handle:
                lines = []
                for line in handle:
                    if int(line.split(',', 1)[0]) > 32:
                        break
                    lines.append(line)
            prefix_tracks[name] = hashlib.sha256(''.join(lines).encode()).hexdigest()
        registered(work)
        dump(manifest_path, dict(status='COMPLETE_SPATIAL_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT',
            study_sha256=sha256(STUDY), specification_sha256=sha256(specification_path),
            sequence=args.sequence, split=args.split, frames=expected[args.sequence], streams=streams,
            models=information, original_image_bytes_delivery_sha256=source_digest.hexdigest(),
            parent_native_source_anchor=anchor_reference, parent_fresh_packets_and_tracks_exact=True,
            prefix32_source_rows=prefix_rows, prefix32_packet_sha256={k: v.hexdigest() for k, v in prefixes.items()},
            prefix32_literal_MOT_sha256=prefix_tracks, gate_statistics={name: observer.summary(name) for name in observer.rows},
            physical_gpu=physical, GPU_uuid=uuid, independent_input_batch=1,
            seconds=time.perf_counter()-started, peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
            GT_read=False, future_frame_input=False, previous_outputs_rewritten=False,
            runtime_scope='Shared, instrumented production is not per-method system FPS.'))
    except Exception:
        writer.abort()
        raise
    finally:
        observer.close()


def canonical(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint, base = context()
    registered(work)
    specification = read(args.specification)
    assert specification['study_sha256'] == sha256(STUDY) and specification['split'] == args.split
    if args.split == 'test':
        require_test_freeze(base, args.specification)
    folder = Path(specification['stream_directory'])
    for sequence, count in lengths(plan, protocol, args.split).items():
        path = Path(plan['frozen_detection_sources']['rgb_'+args.split])/'detections'/(sequence+'.npz')
        values = packet(path)
        writer = StreamTrackWriter(folder, sequence, [CANONICAL['id']])
        try:
            for frame in range(1, count+1):
                lo, hi = values['offsets'][frame-1:frame+1]
                writer.append(frame, {CANONICAL['id']: np.c_[values['boxes'][lo:hi], values['scores'][lo:hi]].astype(np.float32)})
            streams = writer.close(count)
        except Exception:
            writer.abort()
            raise
        dump(folder/'sequence_manifests'/(sequence+'.canonical.json'),
            dict(status='COMPLETE_SPATIAL_CANONICAL_REPLAY_WITHOUT_GT', split=args.split,
                sequence=sequence, frames=count, streams=streams, GT_read=False,
                specification_sha256=sha256(args.specification), source_packet_sha256=sha256(path)))


def evaluate(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    from tools.eval_mft26ir import run_trackeval
    plan, study, protocol, calibration, work, checkpoint, base = context()
    registration = registered(work)
    case = read(args.config)
    assert case['study_sha256'] == sha256(STUDY) and case['split'] == args.split
    assert sha256(case['specification']) == case['specification_sha256']
    specification = read(case['specification'])
    assert specification['split'] == args.split and specification['stream_directory'] == case['stream_directory']
    assert case['model']['id'] == case['id']
    assert case['model'] in specification['models'] or (case.get('canonical') and case['model'] == CANONICAL)
    if args.split == 'test':
        require_test_freeze(base, case['specification'])
    directory = Path(case['stream_directory'])
    expected = lengths(plan, protocol, args.split)
    coverage, sources = {}, {}
    for sequence, count in expected.items():
        suffix = '.canonical.json' if case.get('canonical') else '.json'
        path = directory/'sequence_manifests'/(sequence+suffix)
        value = read(path)
        assert value['specification_sha256'] == case['specification_sha256'] and not value['GT_read']
        assert value['status'] == ('COMPLETE_SPATIAL_CANONICAL_REPLAY_WITHOUT_GT' if case.get('canonical')
            else 'COMPLETE_SPATIAL_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT')
        assert value['frames'] == count and value['sequence'] == sequence and value['split'] == args.split
        if not case.get('canonical'):
            assert value['parent_fresh_packets_and_tracks_exact']
            assert {k: value['models'][case['id']][k] for k in case['model']} == case['model']
        stream = value['streams'][case['id']]
        assert sha256(stream['track_path']) == stream['track_sha256'] and stream['frames'] == count
        coverage[sequence] = stream
        sources[sequence] = dict(manifest_path=str(path), manifest_sha256=sha256(path))
    output = base/'evaluation'/args.split/case['id']
    assert not (output/'metrics.json').exists()
    output.mkdir(parents=True, exist_ok=True)
    sequences = protocol['dev' if args.split == 'dev' else 'historical_test']
    manifest = dict(status='SPATIAL_INFERENCE_COMPLETE_BEFORE_GT_EVALUATION', config=case,
        config_sha256=sha256(args.config), split=args.split, study_sha256=sha256(STUDY),
        sequences=sequences, frames=sum(expected.values()), coverage=coverage, sources=sources,
        source_versions=registration['source_hashes'], GT_inference_input=False,
        future_frame_input=False, previous_outputs_rewritten=False)
    dump(output/'inference_manifest.json', manifest)
    metrics = run_trackeval(str(directory/case['id']/'track_results'), plan['evaluation']['GT_root'],
        str(output), sequences, plan['evaluation']['TrackEval_root'], data_split='train' if args.split == 'dev' else 'test')
    assert len(metrics) == 39
    csv_path = output/'trackeval_output/pedestrian_detailed.csv'
    selected = full_precision_selection(csv_path, args.split, protocol)
    assert all(abs(selected[k]-metrics[k]) <= .000501 for k in selected)
    if case.get('canonical'):
        previous = Path(plan['frozen_detection_sources']['rgb_'+args.split])
        assert metrics == read(previous/'metrics.json')['trackeval']
        assert sha256(csv_path) == sha256(previous/'trackeval_output/pedestrian_detailed.csv')
    registered(work)
    dump(output/'metrics.json', dict(status='COMPLETE', trackeval=metrics, manifest=manifest,
        selection_metrics=selected, selection_precision='original276CSV_COMBINED_full_precision',
        official_CSV_sha256=sha256(csv_path)))


if __name__ == '__main__':
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
        canonical(args)
    else:
        assert args.config
        evaluate(args)
