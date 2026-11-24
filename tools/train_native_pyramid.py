#!/usr/bin/env python3
"""Source-pinned real-GPU engineering or full-Train native-pyramid worker."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from tools.native_pyramid_experiment import (
    FIT_CONFIGS, PIPELINE_REVISION, STUDY, context, gpu_settings, load_teacher, predecessor_ready, read, registry,
)
from tools.native_pyramid_fit import optimizer_step
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_train_data import OrderedTrainPixels, epoch_clips
from tools.native_pyramid_training_loop import guard_storage, run_training_to_epoch, state_sha256
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.native_pyramid_content_guard import make_image_reader
from yolox.models.native_pyramid_fusion import NativePyramidFusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['probe', 'fit'])
    parser.add_argument('--fit-id', required=True)
    parser.add_argument('--through-epoch', choices=[6, 9, 12], type=int)
    args = parser.parse_args()
    plan, study, protocol, calibration, work, checkpoint = context()
    registration = registry(work)
    assert predecessor_ready(work), 'Do not initialize CUDA before existing structural jobs release both cards'
    arm_path = FIT_CONFIGS / (args.fit_id + '.json')
    arm = read(arm_path)
    assert arm['fit_id'] == args.fit_id and arm['parent_study_sha256'] == sha256(STUDY)
    assert registration['fit_configs'].get(args.fit_id) == sha256(arm_path) or (
        arm.get('removal') and read(work / 'native_pyramid_removals_registered.json')['fit_configs'].get(args.fit_id) == sha256(arm_path))
    physical, uuid = gpu_settings()
    guard_storage(work)
    teacher = load_teacher(checkpoint, training=True)
    frozen = dict(encoder=state_sha256(teacher.backbone.state_dict()), head=state_sha256(teacher.head.state_dict()))
    # Teacher construction must not consume the matched native initialization.
    torch.manual_seed(0)
    module = NativePyramidFusion(**arm['module_kwargs']).cuda().float()
    initial = state_sha256(module.state_dict())
    assert sum(p.numel() for p in module.parameters()) == study['architecture'][arm['mode']+'_parameters']
    optimizer = torch.optim.AdamW(module.parameters(), lr=arm['fit']['learning_rate'], weight_decay=arm['fit']['weight_decay'])
    reader, content_reference = make_image_reader(work, plan, protocol, arm['Train_delivery']['lengths'], arm['cue'])
    provider = OrderedTrainPixels(plan, protocol, arm['Train_delivery']['lengths'], calibration,
                                  reader, arm['training_input_hw'], 'cuda')
    metadata = dict(fit_id=args.fit_id, mode=arm['mode'], cue=arm['cue'], removal=arm['removal'], seed=0, pipeline_revision=PIPELINE_REVISION,
        fit_config_sha256=sha256(arm_path), study_sha256=sha256(STUDY),
        registry_sha256=sha256(work / 'native_pyramid_pipeline_registered.json'),
        source_RGB_sha256=study['source_RGB_sha256'], source_hashes=registration['source_hashes'],
        protocol_sha256=study['protocol_sha256'], calibration_sha256=study['calibration_sha256'],
        Train_GT_sha256=provider.gt_hashes, initial_native_sha256=initial,
        physical_gpu=physical, GPU_uuid=uuid, past_memory_stop_gradient=True,
        original_RGB_encoder_and_head_frozen=True, Dev_fit_rows=0, Test_fit_rows=0,
        input_content_reference=content_reference, input_content_guard_revision=1)
    probe_path = work / 'native_pyramid_engineering' / args.fit_id / 'proof.json'
    if args.mode == 'probe':
        assert not probe_path.exists(), 'Preserve prior engineering evidence'
        started = time.monotonic()
        provider.begin_epoch(1)
        clips = list(epoch_clips(provider.lengths, 1))[:2]
        state, gray, rows = None, None, []
        for position, clip in enumerate(clips):
            batch = provider.prepare(clip, arm['cue'])
            row, state, gray = optimizer_step(module, teacher.backbone, teacher.head,
                                               batch, optimizer, arm, position, state, gray)
            rows.append(row)
        assert rows[0]['initial_full_RGB_exact'] and rows[0]['native_stem_gradient_L1'] == 0
        assert rows[1]['native_stem_gradient_L1'] > 0 and all(r['optimizer_updated'] for r in rows)
        if not arm['geometry_fixed']: assert rows[1]['decoded_box_max_change'] > 0
        else: assert all(r['decoded_box_max_change'] == 0 for r in rows)
        assert frozen == dict(encoder=state_sha256(teacher.backbone.state_dict()), head=state_sha256(teacher.head.state_dict()))
        assert all(p.grad is None for p in teacher.parameters())
        if arm['mode'] == 'temporal' and arm['module_kwargs']['use_memory']:
            assert state.frame == 8 and state.sequence == clips[0].sequence
        # Production inference has an original FP16 head and sequential batch1.
        teacher.head.half()
        module.eval().requires_grad_(False)
        torch.manual_seed(0)
        zero = NativePyramidFusion(**arm['module_kwargs']).cuda().float().eval().requires_grad_(False)
        observations = [dict(id='source_RGB', cue='rgb', baseline=True),
            dict(id='trained_native', cue=arm['cue'], geometry_fixed=arm['geometry_fixed']),
            dict(id='zero_native', cue=arm['cue'], geometry_fixed=arm['geometry_fixed'], expect_RGB_exact=True),
            dict(id='trained_unavailable', cue='ir', intervention='unavailable')]
        engine = OnlineNativeDetector(teacher, observations,
            dict(source_RGB=None, trained_native=module, zero_native=zero, trained_unavailable=module),
            study['inference_input_hw'], calibration)
        from tools.native_pyramid_gate_observer import NativeGateObserver
        observer = NativeGateObserver()
        observer.attach('trained_aux', module)
        observer.attach('zero_aux', zero)
        from tools.extract_structural_relative import detections
        from tools.native_pyramid_train_data import preprocess_observation
        from yolox.utils import postprocess
        import numpy as np
        identities = []
        with torch.no_grad():
            for clip in clips:
                for frame in clip.frames:
                    rgb = reader.read(clip.sequence, frame, 'rgb')
                    ir = reader.read(clip.sequence, frame, 'ir') if arm['cue'] == 'ir' else None
                    values, info = engine.step(rgb, ir, clip.sequence, frame)
                    np.testing.assert_array_equal(values['zero_native'], values['source_RGB'])
                    np.testing.assert_array_equal(values['trained_unavailable'], values['source_RGB'])
                    pixels, _, _, _, ratio, _ = preprocess_observation(rgb, None, 'rgb', study['inference_input_hw'], calibration)
                    direct = teacher(torch.from_numpy(pixels[None]).cuda().half())
                    direct = detections(postprocess(direct.clone(), 1, .1, .7)[0], ratio)
                    np.testing.assert_array_equal(direct, values['source_RGB'])
                    identities.append(dict(sequence=clip.sequence, frame=frame,
                        direct_B1_source_exact=True, zero_full_head_exact=info['full_head_exact_to_fresh_RGB']['zero_native'],
                        trained_unavailable_full_head_exact=info['full_head_exact_to_fresh_RGB']['trained_unavailable']))
        gates = {name: observer.summary(name) for name in observer.rows}
        assert gates['trained_aux']['frames'] == 16 and gates['trained_aux']['missing_frames'] == 8
        assert gates['zero_aux']['frames'] == 8
        assert all(level['nonzero_feature_correction_fraction'] == 0 for level in gates['zero_aux']['levels'])
        observer.close()
        registry(work)
        report = dict(status='COMPLETE_REAL_GPU_NATIVE_TRAIN_AND_B1_INFERENCE_ENGINEERING',
            metadata=metadata, updates=2, delivered_Train_frames=8, completed_full_Train_epochs=0,
            optimizer_rows=rows, source_and_missing_identity=identities, gate_statistics=gates,
            frozen_RGB_weights_and_BN_unchanged_before_head_precision_conversion=True,
            production_inference_precision='Original RGB backbone/head FP16, native module FP32, sequentialB1',
            image_reader_statistics=reader.statistics(), seconds=time.monotonic()-started,
            saved_weights=False, Dev_rows=0, Test_rows=0, goal_met=False, supported_innovations=0)
        dump(probe_path, report)
        print(json.dumps(dict(status=report['status'], fit=args.fit_id, seconds=report['seconds'])), flush=True)
    else:
        assert args.through_epoch in study['budget_and_Dev']['stages']
        proof = read(probe_path)
        assert proof['status'] == 'COMPLETE_REAL_GPU_NATIVE_TRAIN_AND_B1_INFERENCE_ENGINEERING'
        assert proof['metadata']['fit_config_sha256'] == sha256(arm_path)
        for normal in study['normal_fits']:
            proof = read(work / 'native_pyramid_engineering' / normal['id'] / 'proof.json')
            assert proof['status'] == 'COMPLETE_REAL_GPU_NATIVE_TRAIN_AND_B1_INFERENCE_ENGINEERING'
            assert proof['metadata']['registry_sha256'] == metadata['registry_sha256']
        authorization = read(work / 'native_pyramid_stage_authorized.json')
        assert authorization['through_epoch'] == args.through_epoch and authorization['study_sha256'] == sha256(STUDY)
        assert args.fit_id in authorization['fit_ids']
        run_training_to_epoch(module, teacher.backbone, teacher.head, optimizer, provider, arm, metadata,
                              arm_path, work / 'native_pyramid_fits' / args.fit_id, args.through_epoch, work)
        registry(work)


if __name__ == '__main__':
    main()
