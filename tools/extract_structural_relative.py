#!/usr/bin/env python3
"""Original November IR, frozen RGB geometry and single-seed relative semantics."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from tools.build_correlation_cache import auxiliary_input
from tools.indexed_postprocess import indexed_postprocess, corner_boxes
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.structural_grid import structural_grid
from tools.structural_relative_inference import decoded_relative_response
from yolox.exp import get_exp
from yolox.models.frozen_head_repair import head_predictions
from yolox.models.structural_relative_repair import StructuralRepair
from yolox.tracker.foveated_regions import exact_half_preproc
from yolox.tracker.priority_observations import priority_union
from yolox.utils import postprocess


def detections(output, ratio):
    if output is None:
        return np.empty((0, 5), np.float32)
    values = output.float().cpu().numpy()
    return np.c_[values[:, :4]/ratio, values[:, 4]*values[:, 5]].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--specification')
    parser.add_argument('--source-check', type=int, default=0)
    parser.add_argument('--limit-frames', type=int, default=0)
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES'); assert physical in ['2', '3']
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    config_path = ROOT/'configs/perceptual_structural_relative_20261123.json'
    policy_path = ROOT/'configs/perceptual_single_seed_policy_20261123.json'
    plan = json.loads(plan_path.read_text()); config = json.loads(config_path.read_text())
    policy = json.loads(policy_path.read_text()); assert policy['fixed_seed'] == 0 and not policy['additional_seed_inference']
    assert policy['structural_base_config_sha256'] == sha256(config_path)
    parent_path = ROOT/'configs/perceptual_correlation_repair_20261123.json'
    parent = json.loads(parent_path.read_text()); assert sha256(parent_path) == config['parent_plan_sha256']
    work = Path(plan['work_root']); protocol = json.loads((ROOT/plan['protocol']).read_text())
    assert args.sequence in protocol['dev' if args.split == 'dev' else 'historical_test']
    assert args.limit_frames >= 0 and (not args.limit_frames or args.split == 'dev')
    if args.split == 'test':
        assert (work/'structural_relative_selection_frozen.json').exists()
    calibration_path = work/'correlation_cache/calibration.json'
    calibration = json.loads(calibration_path.read_text()); assert calibration['config_sha256'] == sha256(parent_path)
    if args.source_check:
        spec = dict(models=[dict(id='CG3_source_RGB', cue='rgb', baseline=True)]+
            [dict(id='SR_zero_'+cue, cue=cue, zero_adapter=True, seed=0, variant='full') for cue in ['rgb', 'gray', 'ir']])
        spec_sha256 = None
    else:
        spec = json.loads(Path(args.specification).read_text()); spec_sha256 = sha256(args.specification)
        assert spec['config_sha256'] == sha256(config_path)
    assert all(row.get('seed', 0) == 0 for row in spec['models'])
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    checkpoint = Path(parent['source_RGB_checkpoint']); assert sha256(checkpoint) == parent['source_RGB_sha256']
    model = get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'), None).get_model()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True); del saved
    model = model.cuda().half().eval().requires_grad_(False)
    architecture = config['architecture']; loaded = {}; model_info = {}; needed = {'rgb'}
    for row in spec['models']:
        name = row['id']; cue = row['cue']; assert name not in loaded
        view = 'ir_image_roll' if row.get('intervention') == 'IR_image_spatial_roll' else cue
        needed.add(view)
        if row.get('baseline'):
            loaded[name] = None; model_info[name] = dict(row); continue
        adapter = StructuralRepair(latent=architecture['latent_channels'], radius=architecture['local_radius'],
            temperature=architecture['structural_temperature'], bound=architecture['feature_residual_bound_RMS_fraction'],
            risk_threshold=architecture['teacher_risk_threshold'], variant=row['variant'],
            tolerance=architecture['forward_backward_tolerance_cells'])
        info = dict(row)
        if not row.get('zero_adapter'):
            path = Path(row['checkpoint']); saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['config_sha256'] == sha256(config_path) and saved['completed_epochs'] == row['epoch']
            meta = saved['metadata']; assert meta['cue'] == cue and meta['variant'] == row['variant'] and meta['seed'] == 0
            assert meta['source_RGB_sha256'] == parent['source_RGB_sha256']
            assert meta['single_seed_user_policy_sha256'] == sha256(policy_path)
            adapter.load_state_dict(saved['adapter'], strict=True)
            info.update(checkpoint_sha256=sha256(path), metadata=meta)
        loaded[name] = adapter.cuda().float().eval().requires_grad_(False); model_info[name] = info
    produce(args, physical, plan, config, config_path, policy_path, spec, spec_sha256,
            calibration, calibration_path, work, model, loaded, model_info, needed)


def produce(args, physical, plan, config, config_path, policy_path, spec, spec_sha256,
            calibration, calibration_path, work, model, loaded, model_info, needed):
    source = Path(plan['frozen_detection_sources']['rgb_'+args.split])/'detections'/(args.sequence+'.npz')
    with np.load(source) as packet:
        reference = {key: packet[key] for key in ['frame_ids', 'offsets', 'boxes', 'scores']}
    limit = min(len(reference['frame_ids']), args.source_check or args.limit_frames or len(reference['frame_ids']))
    out = work/'structural_relative_predictions'/args.split/args.sequence
    if args.source_check:
        out = out/f'source_check_{args.source_check}'
    elif args.limit_frames:
        out = out/f'prefix_check_{args.limit_frames}'
    out.mkdir(parents=True, exist_ok=True); assert not (out/'complete.json').exists()
    buffers = {}; anchors = {}; reference_scores = {}
    for row in spec['models']:
        key = row['id']; buffers[key] = []
        if not row.get('baseline'):
            buffers[key+'_raw'] = []; anchors[key+'_raw'] = []; reference_scores[key+'_raw'] = []
    parity = []; started = time.perf_counter()
    with torch.inference_mode():
        for first in range(1, limit+1, 4):
            frames = list(range(first, min(first+4, limit+1)))
            inputs = {view: [] for view in needed}; available = {view: [] for view in needed}
            structures = {view: [] for view in needed if view != 'gray'}
            structure_valid = {view: [] for view in needed if view != 'gray'}; ratios = []
            for frame in frames:
                rgb_path = Path(plan['data_root'])/('train' if args.split == 'dev' else 'test')/args.sequence/'visible'/f'{frame:06d}.jpg'
                ir_path = Path(plan['generated_ir_root'])/args.split/args.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
                rgb = cv2.imread(str(rgb_path))
                ir = cv2.imread(str(ir_path), -1) if any(v.startswith('ir') for v in needed) else None
                assert rgb is not None
                if ir is not None:
                    assert ir.ndim == 2 and ir.shape == rgb.shape[:2] and ir.dtype == np.uint8
                elif any(v.startswith('ir') for v in needed):
                    raise ValueError('Required actual November IR is unavailable')
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                scalar = dict(rgb=gray, gray=gray, ir=ir)
                if 'ir_image_roll' in needed:
                    scalar['ir_image_roll'] = np.roll(ir, (ir.shape[0]//4, ir.shape[1]//4), (0, 1))
                value, ratio = exact_half_preproc(rgb, config['inference_input_hw'], (.485, .456, .406), (.229, .224, .225))
                inputs['rgb'].append(value); ratios.append(ratio); available['rgb'].append(True)
                for view in needed:
                    image = scalar[view]
                    if view != 'rgb':
                        norm = calibration['statistics']['ir' if view.startswith('ir') else 'gray']
                        value, _ = auxiliary_input(image, config['inference_input_hw'], norm['mean'], norm['std'])
                        inputs[view].append(value); available[view].append(bool(image.max() != image.min()))
                    if view != 'gray':
                        values, valid = structural_grid(image, config['inference_input_hw'])
                        structures[view].append(values); structure_valid[view].append(valid)
            features = {view: model.backbone(torch.from_numpy(np.stack(values)).cuda()) for view, values in inputs.items()}
            descriptors = {view: torch.from_numpy(np.stack(values)).cuda() for view, values in structures.items()}
            validity = {view: torch.from_numpy(np.stack(values)).cuda() for view, values in structure_valid.items()}
            teacher = model.head(features['rgb']); teacher_raw, _ = head_predictions(model.head, features['rgb'])
            teacher_corners = corner_boxes(teacher).float().cpu().numpy()
            teacher_scores = []; offset = 0
            for value in features['rgb']:
                count = value.shape[-2]*value.shape[-1]
                score = teacher[:, offset:offset+count, 4]*teacher[:, offset:offset+count, 5]
                teacher_scores.append(score.reshape(len(frames), 1, *value.shape[-2:])); offset += count
            fresh = [detections(value, ratio) for value, ratio in zip(postprocess(teacher.clone(), 1, .1, .7), ratios)]
            originals = []
            for frame, current in zip(frames, fresh):
                lo, hi = reference['offsets'][frame-1:frame+1]
                original = np.c_[reference['boxes'][lo:hi], reference['scores'][lo:hi]]; originals.append(original)
                parity.append(dict(frame=frame, count_equal=len(current) == len(original),
                    array_exact=bool(np.array_equal(current, original))))
            for row in spec['models']:
                name = row['id']; adapter = loaded[name]
                if adapter is None:
                    buffers[name].extend(fresh); continue
                view = 'ir_image_roll' if row.get('intervention') == 'IR_image_spatial_roll' else row['cue']
                descriptor_view = 'rgb' if view == 'gray' else view
                is_available = available[view]
                if row.get('intervention') == 'unavailable':
                    is_available = [False]*len(frames)
                is_available = torch.tensor(is_available, device='cuda')
                common = (features['rgb'], features[view], teacher_scores, descriptors['rgb'], descriptors[descriptor_view],
                          validity['rgb'], validity[descriptor_view], is_available)
                current = head_predictions(model.head, adapter(*common))[0]
                null = head_predictions(model.head, adapter(*common, null=True))[0]
                prediction = decoded_relative_response(current, null, teacher_raw, teacher,
                    centered=row['variant'] != 'no_centering', bound=config['architecture']['semantic_logit_bound'])
                assert torch.equal(prediction[:, :, :4], teacher[:, :, :4])
                outputs, indices = indexed_postprocess(prediction, 1, .1, .7)
                values = [detections(value, ratio) for value, ratio in zip(outputs, ratios)]
                for batch, (value, index, ratio) in enumerate(zip(values, indices, ratios)):
                    ids = np.empty(0, np.int32) if index is None else index.cpu().numpy().astype(np.int32)
                    refs = (np.empty(0, np.float32) if index is None else
                        (teacher[batch, index, 4].float()*teacher[batch, index, 5].float()).cpu().numpy())
                    np.testing.assert_array_equal(value[:, :4], teacher_corners[batch, ids]/ratio)
                    anchors[name+'_raw'].append(ids); reference_scores[name+'_raw'].append(refs)
                if row.get('zero_adapter') or row.get('intervention') == 'unavailable':
                    for value, original in zip(values, fresh):
                        np.testing.assert_array_equal(value, original)
                buffers[name+'_raw'].extend([value.copy() for value in values])
                merged = [original.copy() if not bool(ok) or row.get('zero_adapter') else
                    priority_union(original, [(value, 1.)])[0] for original, value, ok in zip(originals, values, is_available)]
                buffers[name].extend(merged)
            if frames[-1] % 100 < 4 or frames[-1] == limit:
                progress = dict(frame=frames[-1], frames=limit, models=len(loaded), physical_gpu=physical, seconds=time.perf_counter()-started)
                dump(out/'progress.json', progress); print(json.dumps(progress), flush=True)
    publish(args, physical, plan, config_path, policy_path, spec_sha256, calibration_path,
            out, source, buffers, anchors, reference_scores, model_info, parity, started)


def publish(args, physical, plan, config_path, policy_path, spec_sha256, calibration_path,
            out, source, buffers, anchors, reference_scores, model_info, parity, started):
    outputs = {}; limit = len(parity)
    for key, values in buffers.items():
        merged = np.concatenate(values).reshape(-1, 5)
        path = out/(key+'.npz'); temporary = out/(key+'.tmp.npz')
        extra = {}
        if key in anchors:
            extra = dict(anchor_indices=np.concatenate(anchors[key]), rgb_reference_scores=np.concatenate(reference_scores[key]))
            assert len(merged) == len(extra['anchor_indices']) == len(extra['rgb_reference_scores'])
        np.savez_compressed(temporary, frame_ids=np.arange(1, limit+1),
            offsets=np.r_[0, np.cumsum([len(value) for value in values])], boxes=merged[:, :4], scores=merged[:, 4], **extra)
        os.replace(temporary, path)
        outputs[key] = dict(path=str(path), sha256=sha256(path), frames=limit, detections=len(merged), bytes=path.stat().st_size)
    names = ['tools/extract_structural_relative.py', 'tools/structural_relative_inference.py',
        'tools/build_correlation_cache.py', 'yolox/models/yolo_head.py',
        'tools/indexed_postprocess.py', 'tools/structural_grid.py', 'tools/structural_correspondence.py',
        'yolox/models/structural_relative_repair.py', 'yolox/models/frozen_head_repair.py',
        'yolox/tracker/foveated_regions.py', 'yolox/tracker/priority_observations.py']
    meta = dict(status='SMOKE_COMPLETE' if args.source_check or args.limit_frames else 'COMPLETE_STRUCTURAL_RELATIVE_OBSERVATIONS',
        sequence=args.sequence, split=args.split, frames=limit, outputs=outputs, models=model_info,
        config_sha256=sha256(config_path), specification_sha256=spec_sha256,
        single_seed_policy_sha256=sha256(policy_path), calibration_sha256=sha256(calibration_path),
        source_RGB_sha256=json.loads((ROOT/'configs/perceptual_correlation_repair_20261123.json').read_text())['source_RGB_sha256'],
        global_reference=dict(path=str(source), sha256=sha256(source)),
        source_hashes={name: sha256(ROOT/name) for name in names},
        source_sha256=sha256(__file__), GT_read=False, future_frame_input=False,
        previous_outputs_rewritten=False, independent_sample_batch_size=4, fixed_fitted_seed=0,
        fresh_RGB_parity=parity if args.source_check else
            dict(frames=len(parity), count_equal=sum(row['count_equal'] for row in parity), exact=sum(row['array_exact'] for row in parity)),
        dense_geometry='Every raw emitted box was checked against the exact original RGB corner box at its selected original dense anchor, before publication.',
        probability='Zero semantic influence preserves original decoded probabilities exactly; nonzero bounded logit influence uses FP32 then original FP16 decoding dtype.',
        intervention='IR_image_spatial_roll rolls actual current IR by quarter image in each axis, before encoding and descriptor extraction; no source image file is changed. It is an image intervention, distinct from the model-internal quarter-FPN-roll null reference.',
        physical_gpu=physical, seconds=time.perf_counter()-started,
        peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,
        runtime_scope='Shared production of all declared snapshots and interventions, not per-method end-to-end FPS. Normal gray/IR each use a second shared encoder; RGB-repeat reuses RGB; image-roll diagnosis adds another shared encoder pass only in this diagnostic bundle.')
    dump(out/'complete.json', meta)


if __name__ == '__main__':
    main()
