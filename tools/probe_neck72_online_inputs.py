#!/usr/bin/env python3
"""CPU evidence for uncached point production and causal prefix assembly."""
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from tools.eval_mft26ir import write_sequence_results
from tools.native_neck_io import load_native_encoder
from tools.native_pyramid_training_loop import state_sha256
from tools.native_query_replay import load_native_query_evidence
from tools.neck72_experiment import context
from tools.neck72_online_inputs import CurrentPointStream, PrefixPackets, FEATURE_FIELDS
from tools.neck72_runtime import read, require_storage, require_host_memory
from tools.neck72_runtime_study import sources
from tools.run_neck72_replay import tracker_for
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.native_cue import native_crop, native_input


def point_slice(point, frame, begin, end):
    values = dict(quality=point['quality'][begin:end])
    for lag in [1, 2, 4, 8]:
        for kind, names in [('vote', [f'vote_{lag}_{s}_{t}' for s in ['bbox', 'rgb'] for t in ['bbox', 'rgb']]),
            ('count', [f'{k}_{lag}_{s}' for k in ['good', 'total'] for s in ['bbox', 'rgb']])]:
            a, b = map(int, point[f'{kind}_offsets_{lag}'][frame-1:frame+1])
            for key in names:
                values[key] = point[key][a:b]
    return values


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    cv2.setNumThreads(1); torch.set_num_threads(2); torch.manual_seed(0)
    plan, protocol, work = context(); before = {p: sha256(ROOT / p) for p in sources()}
    output = work / 'neck72_online_inputs_CPU_proof.json'; assert not output.exists()
    require_storage(work); require_host_memory(4*2**30, maximum_workers=1)
    started = time.monotonic(); log = work / 'neck72_online_inputs_pytest.log'
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_neck72_online_inputs.py',
               'tests/test_neck72_frontend_serialization.py', 'tests/test_neck72_frontend_loader.py']
    with log.open('x') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    count = re.search(r'\b(\d+) passed\b', log.read_text()); assert count and int(count.group(1)) >= 20
    serialization_path = work / 'neck72_frontend_serialization_proof.json'
    assert read(serialization_path)['all_frontends_exact']
    records, artifacts = [], {}
    for sequence in protocol['dev']:
        references = {}; producers = {arm: CurrentPointStream() for arm in ['new', 'rgb']}
        for arm in producers:
            p = work / 'features/dev' / sequence / (arm+'.npz')
            with np.load(p) as arrays:
                reference = {k: arrays[k] for k in ['offsets', 'boxes']}
            point_path = work / 'transport/dev' / sequence / (arm+'.npz')
            with np.load(point_path) as arrays:
                reference['point'] = {k: arrays[k] for k in arrays.files}
            artifacts[str(p)] = sha256(p); artifacts[str(point_path)] = sha256(point_path)
            references[arm] = reference
        checks = {arm: 0 for arm in producers}
        for frame in range(1, 33):
            path = Path(plan['data_root']) / 'train' / sequence / 'visible' / f'{frame:06d}.jpg'
            rgb = cv2.imread(str(path)); assert rgb is not None
            for arm, producer in producers.items():
                reference = references[arm]; a, b = map(int, reference['offsets'][frame-1:frame+1])
                values = producer.step(frame, rgb, reference['boxes'][a:b])
                expected = point_slice(reference['point'], frame, a, b)
                np.testing.assert_array_equal(values['quality'][:, 0], expected['quality'][:, 0])
                for key in expected:
                    if key != 'quality':
                        np.testing.assert_array_equal(values[key], expected[key]); checks[arm] += 1
            artifacts[str(path)] = sha256(path)
        records.extend(dict(sequence=sequence, frontend=arm, frames=32,
            compared_point_arrays=checks[arm]+32, consumed_arrays_bit_exact=True,
            actual_IR_images_opened=0, new_feature_or_image_files_saved=0) for arm in producers)
        del references, producers
    # Exercise the same frozen query classes before the new72-query heads exist.
    # Old feature banks are engineering references here, never a fresh-image
    # inference or performance claim and never a source of new accuracy.
    streamed = []
    for algorithm, old_name in [('ordered', 'robust'), ('consensus', 'consensus')]:
        for arm in ['new', 'rgb']:
            for cue in ['rgb', 'ir']:
                path = ROOT / 'configs/perceptual_native_common_test_cases_20261124' / f'NC36_{old_name}_{arm}_{cue}_normal.json'
                config = read(path)
                with np.load(work / 'features/dev/BT-005' / (arm+'.npz')) as arrays:
                    original = {k: arrays[k] for k in arrays.files}
                with np.load(work / 'transport/dev/BT-005' / (arm+'.npz')) as arrays:
                    original['foreground_transport'] = {k: arrays[k] for k in arrays.files}
                native_metadata = load_native_query_evidence(original, work, 'dev', 'BT-005', config)
                outputs, posteriors = [], []
                for online in [False, True]:
                    tracker = tracker_for(config)
                    frozen = (state_sha256(tracker.model.state_dict()), state_sha256(tracker.native_model.state_dict()))
                    packets = PrefixPackets(32); rows = []
                    for frame in range(1, 33):
                        if online:
                            a, b = map(int, original['offsets'][frame-1:frame+1])
                            values = {k: original[k][a:b] for k in FEATURE_FIELDS if k in original}
                            points = point_slice(original['foreground_transport'], frame, a, b)
                            data = packets.append(frame, np.c_[original['boxes'][a:b], original['scores'][a:b]], values, points)
                        else:
                            data = original
                        rows.extend(tracker.update(frame, data))
                    posterior = tracker.query_posterior(32, dict(frame=1, index=0))
                    assert posterior['probability'].size > 0
                    posteriors.append({k: posterior[k].copy() for k in ['probability', 'null', 'delta']})
                    p = work / 'neck72_online_CPU_prefix' / f'{algorithm}_{arm}_{cue}_{online}.txt'
                    write_sequence_results(str(p), rows); outputs.append(p)
                    assert frozen == (state_sha256(tracker.model.state_dict()), state_sha256(tracker.native_model.state_dict()))
                assert outputs[0].read_bytes() == outputs[1].read_bytes()
                for key in posteriors[0]:
                    np.testing.assert_array_equal(posteriors[0][key], posteriors[1][key])
                streamed.append(dict(algorithm=algorithm, frontend=arm, cue=cue, frames=32,
                    literal_tracks_and_nonempty_posteriors_exact=True, original_model_state_unchanged=True,
                    config_sha256=sha256(path), native_metadata=native_metadata,
                    original36encoder_models_only=True, new72_query_model=False,
                    track_hashes=[sha256(p) for p in outputs]))
    # Actual current Train pixels also exercise the exact72 post-neck endpoint.
    index = work / 'native_candidate_index/BT-003/index.npz'
    with np.load(index) as array:
        positions = [0, int(np.flatnonzero(array['frame'] != array['frame'][0])[0])]
        observations = [(int(array['frame'][i]), array['boxes'][i].copy()) for i in positions]
    pixels_checked = []
    for cue in ['rgb', 'ir']:
        model, endpoint = load_native_encoder(cue, 0, work, protocol); frozen = state_sha256(model.state_dict())
        pixels, rectangles, images = [], [], []
        for frame, box in observations:
            p = ((Path(plan['data_root']) / 'train/BT-003/visible' / f'{frame:06d}.jpg') if cue == 'rgb'
                else (Path(plan['generated_ir_root']) / 'train/BT-003/infrared_diffv2ir' / f'{frame:06d}.png'))
            image = cv2.imread(str(p), cv2.IMREAD_COLOR if cue == 'rgb' else cv2.IMREAD_UNCHANGED)
            assert image is not None and (cue != 'ir' or image.ndim == 2 and image.dtype == np.uint8)
            crop, rectangle = native_crop(image, box); pixels.append(crop); rectangles.append(rectangle)
            images.append(dict(frame=frame, path=str(p), sha256=sha256(p)))
            artifacts[str(p)] = sha256(p)
        value = native_input(torch.from_numpy(np.stack(pixels)), torch.from_numpy(np.stack(rectangles)), 'cpu')
        with torch.inference_mode():
            result = F.normalize(model.neck(model.projection(model.backbone(value))), dim=-1)
        assert result.shape == (2, 512) and torch.isfinite(result).all()
        np.testing.assert_allclose(result.norm(dim=-1).numpy(), 1, rtol=0, atol=1e-6)
        assert frozen == state_sha256(model.state_dict())
        artifacts[endpoint['path']] = endpoint['sha256']
        pixels_checked.append(dict(cue=cue, seed=0, endpoint=endpoint, actual_Train_images=images,
                                   frozen_weights_BN_unchanged=True, finite_unit_postneck_vectors=True))
        del model, value, result
    assert before == {p: sha256(ROOT / p) for p in sources()} and not torch.cuda.is_initialized()
    report = dict(status='COMPLETE_CPU_NECK72_ONLINE_INPUT_AND_PREFIX_PACKET_PROOF',
        source_hashes=before, artifact_hashes=artifacts, tests_passed=int(count.group(1)),
        runtime_serialization_revision=2, frontend_serialization_proof_sha256=sha256(serialization_path),
        pytest_command=command, test_log_sha256=sha256(log), real_Dev_point_prefixes=records,
        all_consumed_real_Dev_point_arrays_exact=True, unique_Dev_RGB_frames=64,
        method_point_frames=128, actual_IR_images_opened_for_point_production=0,
        existing_model_prefixes=streamed, existing_model_streamed_prefix_tracks_exact=True,
        actual_Train_pixel_checks=pixels_checked, GPU_work=False, new_accuracy=False,
        detector_or_FP16_appearance_GPU_source_proof=False, actual_complete_runtime_measurement=False,
        fixed_seed=0, additional_seed_work=False, seconds=time.monotonic()-started,
        scope='CPU source and packet engineering only. Real point arrays compare to all original '
            'four-mode/two-frontend pooled caches; streamed queries use existing36-encoder seed0 heads. '
            'Four actual Train images exercise72-encoder FP32 endpoints. '
            'Fresh detector/FP16 appearance/B1 versus B4 numerics and all frozen72-head equivalence '
            'still require actual4/5GPU engineering; no FPS or new method metric is established. '
            'Serialization revision2 reuses existing compact artifacts with independently proven '
            'complete Module.half state identity, preserving the original scientific specification.',
        goal_met=False, supported_innovations=0)
    require_storage(work); dump(output, report)
    print(report['status'], report['tests_passed'], 'tests;', len(records), 'real point prefixes;',
          len(streamed), 'old head stream comparisons;', report['seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
