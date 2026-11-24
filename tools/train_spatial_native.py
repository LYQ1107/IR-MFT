#!/usr/bin/env python3
"""Source-pinned spatial worker; complete registration/release before CUDA.

The callable canary is also used by an explicitly CPU-only engineering probe.
CPU checks cannot produce the GPU eligibility artifact accepted for fitting.
"""
import argparse
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from tools.native_pyramid_experiment import gpu_settings, load_teacher, read
from tools.native_pyramid_fit import optimizer_step
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_train_data import OrderedTrainPixels, epoch_clips, preprocess_observation
from tools.native_pyramid_training_loop import run_training_to_epoch, state_sha256
from tools.neck72_runtime import require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.spatial_native_fusion import SpatialNativeFusion
from tools.spatial_native_observer import SpatialObserver
from tools.spatial_native_study import (STUDY, REGISTRY, arm_path, bounded_image_reader,
    computation_key, context, predecessor_ready, registered, validate_history)


def sums(module, fragment):
    return sum(float(p.grad.detach().abs().sum()) for name, p in module.named_parameters()
        if fragment in name and p.grad is not None)


def canary(module, teacher, provider, arm, inference_frames):
    """Actual registered optimizer and B1 image path, no GT in inference."""
    import numpy as np
    from tools.extract_structural_relative import detections
    from yolox.utils import postprocess
    frozen = dict(encoder=state_sha256(teacher.backbone.state_dict()), head=state_sha256(teacher.head.state_dict()))
    optimizer = torch.optim.AdamW(module.parameters(), lr=arm['fit']['learning_rate'],
                                 weight_decay=arm['fit']['weight_decay'])
    provider.begin_epoch(1)
    clips = list(epoch_clips(provider.lengths, 1))[:3]
    assert len(clips) == 3 and len({c.sequence for c in clips}) == 1
    assert [f for c in clips for f in c.frames] == list(range(1, 13))
    rows, state, previous = [], None, None
    for position, clip in enumerate(clips):
        batch = provider.prepare(clip, arm['cue'])
        row, state, previous = optimizer_step(module, teacher.backbone, teacher.head,
            batch, optimizer, arm, position, state, previous)
        row['gradients_L1'] = {key: sums(module, key) for key in ['output.', 'quality.',
            'encoder.blocks.0.0', 'spatial.4', 'redundancy.2', 'rgb_projection', 'aux_projection']}
        rows.append(row)
    assert state is None and rows[0]['initial_full_RGB_exact']
    assert all(r['optimizer_updated'] for r in rows)
    assert rows[0]['gradients_L1']['encoder.blocks.0.0'] == 0 and rows[0]['gradients_L1']['output.'] > 0
    assert rows[1]['gradients_L1']['encoder.blocks.0.0'] > 0
    for enabled, key in [(module.use_quality, 'quality.'), (module.use_spatial, 'spatial.4'),
                         (module.use_redundancy, 'redundancy.2')]:
        if enabled:
            assert rows[1]['gradients_L1'][key] > 0
    if module.use_spatial or module.use_redundancy:
        assert rows[2]['gradients_L1']['rgb_projection'] > 0 and rows[2]['gradients_L1']['aux_projection'] > 0
    assert frozen == dict(encoder=state_sha256(teacher.backbone.state_dict()), head=state_sha256(teacher.head.state_dict()))
    assert all(p.grad is None for p in teacher.parameters())
    device = next(teacher.parameters()).device
    if device.type == 'cuda':
        teacher.head.half()
    module.eval().requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        zero = SpatialNativeFusion(**arm['module_kwargs']).to(device).float().eval().requires_grad_(False)
    dtype = next(teacher.parameters()).dtype
    frozen_inference = state_sha256(teacher.state_dict())
    observations = [dict(id='RGB', cue='rgb', baseline=True), dict(id='trained', cue=arm['cue']),
        dict(id='zero', cue=arm['cue'], expect_RGB_exact=True),
        dict(id='missing', cue='ir', intervention='unavailable')]
    observer = SpatialObserver()
    observer.attach('trained', module)
    observer.attach('zero', zero)
    engine = OnlineNativeDetector(teacher, observations, dict(RGB=None, trained=module, zero=zero, missing=module),
        arm['inference_input_hw'], provider.calibration)
    checks = []
    try:
        for frame in range(1, inference_frames+1):
            sequence = clips[0].sequence
            rgb = provider.reader.read(sequence, frame, 'rgb')
            ir = provider.reader.read(sequence, frame, 'ir') if arm['cue'] == 'ir' else None
            with torch.no_grad():
                values, information = engine.step(rgb, ir, sequence, frame)
                pixels, _, _, _, ratio, _ = preprocess_observation(rgb, None, 'rgb', arm['inference_input_hw'], provider.calibration)
                direct = teacher(torch.from_numpy(pixels[None]).to(device=device, dtype=dtype))
                direct = detections(postprocess(direct.clone(), 1, .1, .7)[0], ratio)
            np.testing.assert_array_equal(values['RGB'], direct)
            np.testing.assert_array_equal(values['RGB'], values['zero'])
            np.testing.assert_array_equal(values['RGB'], values['missing'])
            checks.append(dict(sequence=sequence, frame=frame, fresh_direct_packets_exact=True,
                zero_full_head_exact=information['full_head_exact_to_fresh_RGB']['zero'],
                learned_missing_full_head_exact=information['full_head_exact_to_fresh_RGB']['missing']))
        gates = {name: observer.summary(name) for name in observer.rows}
    finally:
        observer.close()
    assert frozen_inference == state_sha256(teacher.state_dict())
    return dict(optimizer_rows=rows, actual_updates=3, delivered_Train_frames=12, completed_full_Train_epochs=0,
        inference_B1_frames=inference_frames, source_zero_missing_checks=checks, gate_statistics=gates,
        frozen_RGB_weights_BN_unchanged_before_explicit_head_precision_conversion=True,
        inference_RGB_state_unchanged=True, inference_precision=str(dtype), saved_weights=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['probe', 'fit'])
    parser.add_argument('--fit-id', required=True)
    parser.add_argument('--through-epoch', type=int, choices=[6, 9, 12])
    args = parser.parse_args()
    plan, study, protocol, calibration, work, checkpoint, base = context()
    registration = registered(work)
    assert predecessor_ready(work), 'Wait for complete prior exports and actual process exits before CUDA'
    path = arm_path(base, args.fit_id)
    arm = read(path)
    assert arm['fit_id'] == args.fit_id and arm['parent_study_sha256'] == sha256(STUDY)
    arms = dict(registration['normal_arms'])
    if arm.get('removal'):
        removal_registration = read(base/'removals_registered.json')
        assert removal_registration['study_sha256'] == sha256(STUDY)
        arms.update(removal_registration['new_arms'])
    assert arms.get(args.fit_id) == sha256(path)
    require_host_memory(8*2**30, maximum_workers=1)
    require_storage(work, growth_bytes=int(.5*2**30))
    physical, uuid = gpu_settings()
    teacher = load_teacher(checkpoint, training=True)
    torch.manual_seed(0)
    module = SpatialNativeFusion(**arm['module_kwargs']).cuda().float()
    assert sum(p.numel() for p in module.parameters()) == study['architecture']['quality_parameters']
    reader, reference = bounded_image_reader(work, plan, protocol, arm['Train_delivery']['lengths'], arm['cue'])
    provider = OrderedTrainPixels(plan, protocol, arm['Train_delivery']['lengths'], calibration,
        reader, arm['training_input_hw'], 'cuda')
    metadata = dict(fit_id=args.fit_id, variant=arm['variant'], cue=arm['cue'], removal=arm['removal'], seed=0,
        fit_config_sha256=sha256(path), study_sha256=sha256(STUDY), registry_sha256=sha256(base/REGISTRY),
        computation_key=computation_key(arm), source_hashes=registration['source_hashes'],
        source_RGB_sha256=study['source_RGB_sha256'], protocol_sha256=study['protocol_sha256'],
        calibration_sha256=study['calibration_sha256'], Train_GT_sha256=provider.gt_hashes,
        initial_native_sha256=state_sha256(module.state_dict()), input_content_reference=reference,
        physical_gpu=physical, GPU_uuid=uuid, original_RGB_frozen=True, Dev_fit_rows=0, Test_fit_rows=0)
    proof_path = base/'engineering'/args.fit_id/'proof.json'
    started = time.monotonic()
    if args.mode == 'probe':
        assert not proof_path.exists()
        record = canary(module, teacher, provider, arm, inference_frames=12)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 8*2**30
        registered(work)
        dump(proof_path, dict(status='COMPLETE_SPATIAL_GPU_TRAIN_B1_ENGINEERING', metadata=metadata,
            record=record, image_reader_statistics=reader.statistics(), seconds=time.monotonic()-started,
            GPU_work=True, Dev_rows=0, Test_rows=0, goal_met=False, supported_innovations=0))
    else:
        assert args.through_epoch in study['budget_and_Dev']['stages']
        for fit_id in [r['id'] for r in study['normal_fits']]+[args.fit_id]:
            proof = read(base/'engineering'/fit_id/'proof.json')
            assert proof['status'] == 'COMPLETE_SPATIAL_GPU_TRAIN_B1_ENGINEERING' and proof['GPU_work']
            assert proof['metadata']['registry_sha256'] == metadata['registry_sha256']
            assert proof['metadata']['fit_config_sha256'] == sha256(arm_path(base, fit_id))
        authorization = read(base/'stage_authorized.json')
        assert authorization['study_sha256'] == sha256(STUDY) and authorization['through_epoch'] == args.through_epoch
        assert args.fit_id in authorization['fit_ids'] and authorization['arm_hashes'][args.fit_id] == sha256(path)
        optimizer = torch.optim.AdamW(module.parameters(), lr=arm['fit']['learning_rate'], weight_decay=arm['fit']['weight_decay'])
        run_training_to_epoch(module, teacher.backbone, teacher.head, optimizer, provider, arm, metadata,
            path, base/'fits'/args.fit_id, args.through_epoch, work)
        progress = read(base/'fits'/args.fit_id/'progress.json')
        validate_history(dict(completed_epochs=progress['completed_full_Train_epochs'],
            history=progress['history'], metadata=progress['metadata']), args.through_epoch, arm)
        registered(work)


if __name__ == '__main__':
    main()
