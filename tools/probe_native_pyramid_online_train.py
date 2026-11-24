#!/usr/bin/env python3
"""Actual Train pixels through the shared online fit, CPU engineering only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import torch

from tools.native_pyramid_fit import optimizer_step
from tools.native_pyramid_train_data import OrderedTrainPixels, epoch_clips, schedule_proof
from tools.probe_native_pyramid_train import state_digest
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.structural_train_image_reader import TrainImageReader
from yolox.exp import get_exp
from yolox.models.frozen_head_repair import head_predictions
from yolox.models.native_pyramid_fusion import NativePyramidFusion


class AuditedReader(TrainImageReader):
    def read(self, sequence, frame, modality):
        result = super().read(sequence, frame, modality)
        self.delivered.append((sequence, frame, modality))
        return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Do not use a third GPU for CPU engineering'
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    study_path = ROOT / 'configs/perceptual_native_pyramid_study_20261124.json'
    study = json.loads(study_path.read_text())
    plan_path = ROOT / 'configs/perceptual_mft_20261122.json'
    assert sha256(plan_path) == study['main_plan_sha256']
    plan = json.loads(plan_path.read_text())
    protocol_path = ROOT / plan['protocol']
    assert sha256(protocol_path) == study['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    work = Path(plan['work_root'])
    calibration_path = work / 'correlation_cache/calibration.json'
    assert sha256(calibration_path) == study['calibration_sha256']
    calibration = json.loads(calibration_path.read_text())
    length_path = work / study['Train_length_source']
    assert sha256(length_path) == study['Train_length_source_sha256']
    lengths = json.loads(length_path.read_text())['lengths']
    assert lengths == study['Train_delivery']['lengths']
    policy_path = ROOT / 'configs/perceptual_single_seed_policy_20261123.json'
    assert sha256(policy_path) == study['single_seed_policy_sha256']
    assert json.loads(policy_path.read_text())['fixed_seed'] == study['fixed_seed'] == 0
    output = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_PYRAMID_ONLINE_ENGINEERING.json'
    assert not output.exists(), 'Never replace the prior engineering evidence'
    assert shutil.disk_usage(work).free >= study['storage']['minimum_independent_filesystem_free_GiB'] * 2**30
    parent = json.loads((ROOT / 'configs/perceptual_correlation_repair_20261123.json').read_text())
    checkpoint = Path(parent['source_RGB_checkpoint'])
    assert sha256(checkpoint) == study['source_RGB_sha256']
    source_names = ['tools/probe_native_pyramid_online_train.py', 'tools/native_pyramid_train_data.py',
        'tools/native_pyramid_fit.py', 'yolox/models/native_pyramid_fusion.py',
        'tools/structural_train_image_reader.py', 'tools/build_correlation_cache.py',
        'yolox/models/frozen_head_repair.py', 'yolox/tracker/foveated_regions.py',
        'yolox/models/yolo_head.py', 'tools/probe_native_pyramid_train.py']
    sources = {name: sha256(ROOT / name) for name in source_names}
    started = time.monotonic()
    teacher = get_exp(str(ROOT / 'exps/yolox_mft26ir_visible.py'), None).get_model().float().eval().requires_grad_(False)
    original = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    teacher.load_state_dict(original, strict=True)
    reader = AuditedReader(plan['data_root'], plan['generated_ir_root'], lengths,
                           max_bytes=256 * 2**20, reserve_bytes=40 * 2**30)
    reader.delivered = []
    clips = list(epoch_clips(lengths, 1))[:study['engineering_canary']['clips_each']]
    assert len(clips) == 2 and clips[0].sequence == clips[1].sequence
    assert clips[0].frames == (1, 2, 3, 4) and clips[1].frames == (5, 6, 7, 8)
    proof = schedule_proof(lengths, 1)
    assert proof['frames'] == 27251 and proof['clips'] == 6814
    rows, initial_by_mode = [], {}
    for mode in study['modes']:
        for cue in study['matched_cues']:
            provider = OrderedTrainPixels(plan, protocol, lengths, calibration, reader,
                                          study['training_input_hw'], 'cpu')
            provider.begin_epoch(1)
            reader.delivered = []
            torch.manual_seed(0)
            module = NativePyramidFusion(mode=mode, bound=study['architecture']['feature_residual_bound']).float()
            initial = state_digest(module.state_dict())
            assert initial_by_mode.setdefault(mode, initial) == initial
            optimizer = torch.optim.AdamW(module.parameters(), lr=study['fit']['learning_rate'],
                                         weight_decay=study['fit']['weight_decay'])
            losses, state, gray = [], None, None
            for position, clip in enumerate(clips):
                batch = provider.prepare(clip, cue)
                if position == 1 and mode == 'temporal':
                    assert state is not None and state.frame == 4
                    assert all(not v.requires_grad for v in state.latent + state.covariance)
                row, state, gray = optimizer_step(module, teacher.backbone, teacher.head, batch,
                                                  optimizer, study, position, state, gray)
                losses.append(row)
            assert losses[0]['initial_full_RGB_exact']
            assert losses[0]['native_stem_gradient_L1'] == 0
            assert losses[1]['native_stem_gradient_L1'] > 0
            assert losses[1]['decoded_box_max_change'] > 0
            assert all(row['optimizer_updated'] for row in losses)
            if mode == 'temporal':
                assert state.frame == 8 and state.sequence == clips[0].sequence
            # Assert actual I/O isolation of RGB/gray controls, not just flags.
            assert len([v for v in reader.delivered if v[2] == 'rgb']) == 8
            assert len([v for v in reader.delivered if v[2] == 'ir']) == (8 if cue == 'ir' else 0)
            assert state_digest(module.state_dict()) != initial
            with torch.no_grad():
                features = teacher.backbone(batch.rgb.float())
                reference, metadata = head_predictions(teacher.head, features)
                unavailable, empty_state = [], None
                for index, frame in enumerate(batch.address.frames):
                    value, empty_state, _ = module(tuple(v[index:index+1] for v in features), None,
                        tuple(v[index:index+1] for v in metadata['scores']), available=False,
                        state=empty_state, sequence=batch.address.sequence, frame=frame)
                    assert empty_state is None
                    unavailable.append(value)
                prediction, _ = head_predictions(teacher.head,
                    tuple(torch.cat([v[level] for v in unavailable]) for level in range(3)))
                assert torch.equal(prediction, reference)
            row = dict(mode=mode, cue=cue, parameter_count=sum(p.numel() for p in module.parameters()),
                initial_sha256=initial, delivered_actual_frames=8, actual_updates=2, losses=losses,
                actual_prefix_sha256=provider.digest.hexdigest(),
                current_train_sequence=clips[0].sequence,
                actual_IR_image_reads=8 if cue == 'ir' else 0,
                temporal_state_continued_across_optimizer_updates=(mode == 'temporal'),
                no_gradient_through_past_state=True, trained_unavailable_full_head_exact_RGB=True)
            assert row['parameter_count'] == study['architecture'][mode + '_parameters']
            rows.append(row)
            print(json.dumps(dict(mode=mode, cue=cue, status='TWO_ORDERED_TRAIN_CLIPS_PASSED',
                                  temporal_final_frame=state.frame if state else None)), flush=True)
            del module, optimizer, provider
    assert all(value.grad is None for value in teacher.parameters())
    assert all(torch.equal(value, original[name].to(dtype=value.dtype)) for name, value in teacher.state_dict().items())
    assert all(sha256(ROOT / name) == digest for name, digest in sources.items())
    addresses = [(c.sequence, f) for c in clips for f in c.frames]
    source_frames = [dict(sequence=s, frame=f,
        RGB_sha256=sha256(Path(plan['data_root']) / 'train' / s / 'visible' / f'{f:06d}.jpg'),
        November_IR_sha256=sha256(Path(plan['generated_ir_root']) / 'train' / s / 'infrared_diffv2ir' / f'{f:06d}.png'))
        for s, f in addresses]
    report = dict(status='COMPLETE_REAL_TRAIN_CPU_ONLINE_NATIVE_FIT_ENGINEERING_ONLY',
        study_sha256=sha256(study_path), source_hashes=sources, source_RGB_sha256=sha256(checkpoint),
        source_frames=source_frames, Train_GT_sha256={s: sha256(Path(plan['evaluation']['GT_root']) /
            'train' / s / 'gt/gt.txt') for s in lengths},
        full_epoch_schedule_proof=proof, completed_full_Train_epochs=0,
        calibration_sha256=sha256(calibration_path), fixed_seed=0, additional_seed_work=False,
        rows=rows, seconds=time.monotonic()-started, image_reader_statistics=reader.statistics(),
        frozen_RGB_encoder_head_weights_and_BN_exact_unchanged=True,
        GPU_work=False, device='CPU FP32; actual GPU FP16 encoder proof remains required',
        Dev_rows=0, Test_rows=0, saved_model_or_image_or_FPN_disk_cache=False,
        formal_fit_launched=False, complete_Dev_Test_pipeline_ready=False,
        innovation_or_HOTA_evidence=False, goal_met=False, supported_innovations=0)
    dump(output, report)
    print(json.dumps(dict(status=report['status'], cases=len(rows), seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
