#!/usr/bin/env python3
"""CPU-only real-Train gradient/identity canary, never a performance experiment."""
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
import numpy as np
import torch

from tools.build_correlation_cache import auxiliary_input, read_pair
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.exp import get_exp
from yolox.models.frozen_head_repair import head_predictions, original_supervised_loss, preserve_reference
from yolox.models.native_pyramid_fusion import NativePyramidFusion
from yolox.tracker.foveated_regions import exact_half_preproc


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU engineering must not allocate another GPU'
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    config_path = ROOT / 'configs/perceptual_native_pyramid_prototype_20261124.json'
    config = json.loads(config_path.read_text())
    plan_path = ROOT / 'configs/perceptual_mft_20261122.json'
    assert sha256(plan_path) == config['main_plan_sha256']
    plan = json.loads(plan_path.read_text())
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    work = Path(plan['work_root'])
    output = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_PYRAMID_ENGINEERING.json'
    assert not output.exists()
    assert shutil.disk_usage(work).free >= 40 * 2**30
    sequence, frames = config['sequence'], config['frames']
    assert sequence in protocol['train'] and frames == [1, 2] and config['fixed_seed'] == 0
    parent = json.loads((ROOT / 'configs/perceptual_correlation_repair_20261123.json').read_text())
    checkpoint = Path(parent['source_RGB_checkpoint'])
    assert sha256(checkpoint) == parent['source_RGB_sha256'] == config['source_RGB_sha256']
    calibration_path = work / 'correlation_cache/calibration.json'
    calibration = json.loads(calibration_path.read_text())
    assert calibration['Dev_rows'] == calibration['Test_rows'] == 0
    gt_path = Path(plan['evaluation']['GT_root']) / 'train' / sequence / 'gt/gt.txt'
    assert sha256(gt_path) == protocol['derived_gt_sha256']['train/' + sequence]
    gt = np.loadtxt(gt_path, delimiter=',', ndmin=2)
    gt = gt[(gt[:, 6] > 0) & (gt[:, 7] == 1)]
    height, width = config['input_hw']
    inputs = {cue: [] for cue in config['matched_cues']}
    targets, masks, gray_for_flow, source_rows = [], [], [], []
    for frame in frames:
        rgb, ir = read_pair(plan, sequence, frame)
        assert ir.max() != ir.min(), 'Registered real engineering observations must contain available IR'
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        value, ratio = exact_half_preproc(rgb, (height, width), (.485, .456, .406), (.229, .224, .225))
        inputs['rgb'].append(value.astype(np.float32))
        for cue, image in [('gray', gray), ('ir', ir)]:
            norm = calibration['statistics'][cue]
            value, _ = auxiliary_input(image, (height, width), norm['mean'], norm['std'])
            inputs[cue].append(value.astype(np.float32))
        actual_h, actual_w = int(rgb.shape[0] * ratio), int(rgb.shape[1] * ratio)
        valid = np.zeros((1, height, width), bool)
        valid[:, :actual_h, :actual_w] = True
        masks.append(valid)
        padded = np.zeros((height, width), np.uint8)
        padded[:actual_h, :actual_w] = cv2.resize(gray, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)
        gray_for_flow.append(cv2.resize(padded, (width // 4, height // 4), interpolation=cv2.INTER_AREA))
        target = np.zeros((500, 5), np.float32)
        boxes = gt[gt[:, 0] == frame, 2:6].astype(np.float32).copy()
        boxes[:, :2] += boxes[:, 2:] / 2
        boxes *= ratio
        assert 0 < len(boxes) <= 500
        target[:len(boxes), 1:] = boxes
        targets.append(target)
        rgb_path = Path(plan['data_root']) / 'train' / sequence / 'visible' / f'{frame:06d}.jpg'
        ir_path = Path(plan['generated_ir_root']) / 'train' / sequence / 'infrared_diffv2ir' / f'{frame:06d}.png'
        source_rows.append(dict(sequence=sequence, frame=frame, RGB_sha256=sha256(rgb_path),
                                November_IR_path=str(ir_path), November_IR_sha256=sha256(ir_path), targets=len(boxes)))
    # Farneback maps its FIRST image to its SECOND. Current first gives C->P.
    flow = cv2.calcOpticalFlowFarneback(gray_for_flow[1], gray_for_flow[0], None,
                                      .5, 3, 15, 3, 5, 1.2, 0)
    flow = torch.from_numpy(flow.transpose(2, 0, 1)[None].copy())
    pixels = {cue: torch.from_numpy(np.stack(values)) for cue, values in inputs.items()}
    valid = torch.from_numpy(np.stack(masks))
    target = torch.from_numpy(np.stack(targets))
    started = time.monotonic()
    torch.manual_seed(0)
    teacher = get_exp(str(ROOT / 'exps/yolox_mft26ir_visible.py'), None).get_model().float().eval().requires_grad_(False)
    original = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    teacher.load_state_dict(original, strict=True)
    with torch.no_grad():
        rgb_features = teacher.backbone(pixels['rgb'])
        reference, metadata = head_predictions(teacher.head, rgb_features)
    assert all(not v.is_inference() and not v.requires_grad for v in rgb_features)
    source_names = ['tools/probe_native_pyramid_train.py', 'yolox/models/native_pyramid_fusion.py',
        'yolox/models/frozen_head_repair.py', 'tools/build_correlation_cache.py',
        'yolox/tracker/foveated_regions.py', 'yolox/models/yolo_head.py']
    sources = {name: sha256(ROOT / name) for name in source_names}
    rows, initial_by_mode = [], {}
    for mode in config['modes']:
        for cue in config['matched_cues']:
            torch.manual_seed(0)
            module = NativePyramidFusion(mode=mode, bound=config['feature_residual_bound']).float().train()
            initial = state_digest(module.state_dict())
            if mode in initial_by_mode:
                assert initial == initial_by_mode[mode]
            initial_by_mode[mode] = initial
            optimizer = torch.optim.AdamW(module.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            losses, stem_gradients, projection_gradients = [], [], []
            for step in range(config['engineering_updates_each']):
                fused, state = [], None
                for index, frame in enumerate(frames):
                    result, state, _ = module(tuple(v[index:index+1] for v in rgb_features),
                        pixels[cue][index:index+1], tuple(v[index:index+1] for v in metadata['scores']),
                        valid=valid[index:index+1], state=state, sequence=sequence, frame=frame,
                        flow_to_previous=flow if index else None)
                    fused.append(result)
                predictions, _ = head_predictions(teacher.head,
                    tuple(torch.cat([row[level] for row in fused]) for level in range(3)))
                if step == 0:
                    assert torch.equal(predictions, reference), 'Initial full detector outputs must equal RGB'
                original_loss = original_supervised_loss(teacher.head, predictions, metadata, target, (height, width))[0]
                preservation = preserve_reference(predictions, reference, metadata['strides'], reliable_only=True)
                loss = original_loss + config['lambda_reliable_RGB'] * preservation
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                stem_gradients.append(float(module.encoder.blocks[0][0].weight.grad.abs().sum()))
                projection_gradients.append(sum(float(layer.weight.grad.abs().sum()) for layer in module.output))
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                losses.append(dict(step=step+1, total=float(loss.detach()), full_YOLOX=float(original_loss.detach()),
                    reliable_RGB_preservation=float(preservation.detach()), gradient_norm=float(norm),
                    decoded_box_max_change_from_RGB=float((predictions[:, :, :4] - reference[:, :, :4]).detach().abs().max())))
            assert stem_gradients[0] == 0 and stem_gradients[1] > 0 and projection_gradients[0] > 0
            assert state_digest(module.state_dict()) != initial
            assert losses[1]['decoded_box_max_change_from_RGB'] > 0
            # Test the learned module's unavailable route, not just its zero init.
            missing = []
            for index, frame in enumerate(frames):
                result, state, _ = module(tuple(v[index:index+1] for v in rgb_features), None,
                    tuple(v[index:index+1] for v in metadata['scores']), available=False,
                    sequence=sequence, frame=frame)
                assert state is None
                missing.append(result)
            with torch.no_grad():
                unavailable, _ = head_predictions(teacher.head,
                    tuple(torch.cat([row[level] for row in missing]) for level in range(3)))
            assert torch.equal(unavailable, reference)
            row = dict(mode=mode, cue=cue, updates=2, parameter_count=sum(p.numel() for p in module.parameters()),
                initial_state_sha256=initial, initial_full_detector_RGB_exact=True,
                native_encoder_stem_gradient_L1=stem_gradients, projection_gradient_L1=projection_gradients,
                trained_unavailable_full_detector_RGB_exact=True, losses=losses)
            rows.append(row)
            print(json.dumps(dict(mode=mode, cue=cue, updates=2, native_encoder_gradient_verified=True,
                                  missing_full_RGB_exact=True)), flush=True)
            del module, optimizer
    assert all(value.grad is None for value in teacher.parameters())
    assert all(torch.equal(value, original[name].to(dtype=value.dtype)) for name, value in teacher.state_dict().items())
    assert all(sha256(ROOT / name) == digest for name, digest in sources.items())
    report = dict(status='COMPLETE_REAL_TRAIN_CPU_NATIVE_PYRAMID_ENGINEERING_ONLY', rows=rows,
        source_frames=source_rows, configuration_sha256=sha256(config_path), main_plan_sha256=sha256(plan_path),
        source_RGB_sha256=sha256(checkpoint), Train_GT_sha256=sha256(gt_path), calibration_sha256=sha256(calibration_path),
        source_hashes=sources, fixed_seed=0, additional_seed_work=False, GPU_work=False,
        RGB_encoder_and_head_weights_and_BN_exact_unchanged=True, seconds=time.monotonic()-started,
        precision='CPU FP32; not bit-equivalence to the production GPU FP16 baseline',
        saved_weights=False, new_disk_image_or_FPN_cache=False, Dev_rows=0, Test_rows=0,
        full_training_completed=False, formal_current_IR_architecture_validation_completed=False,
        scope=config['scope'], formal_work_remaining=config['formal_work_remaining'],
        goal_met=False, supported_innovations=0)
    dump(output, report)
    print(json.dumps(dict(status=report['status'], rows=len(rows), seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
