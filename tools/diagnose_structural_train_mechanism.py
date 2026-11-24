#!/usr/bin/env python3
"""Train-only read-only diagnostics of completed full-Train structural adapters."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F

from tools.neck72_runtime import require_host_memory, require_storage
from tools.native_pyramid_training_loop import state_sha256
from yolox.models.frozen_head_repair import head_predictions
from yolox.models.structural_factorial_repair import StructuralFactorial
from yolox.models.structural_relative_repair import structural_weights, relative_semantics
from yolox.models.yolo_head import YOLOXHead

SPEC = ROOT/'configs/perceptual_mechanism_diagnosis_20261125.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class Moments:
    def __init__(self):
        self.n, self.total, self.square, self.low, self.high = 0, 0., 0., None, None

    def add(self, values):
        x = values.detach().double().reshape(-1)
        assert bool(torch.isfinite(x).all())
        if not len(x):
            return
        self.n += len(x)
        self.total += float(x.sum())
        self.square += float(x.square().sum())
        lo, hi = float(x.min()), float(x.max())
        self.low = lo if self.low is None else min(self.low, lo)
        self.high = hi if self.high is None else max(self.high, hi)

    def summary(self):
        mean = None if not self.n else self.total/self.n
        return dict(n=self.n, mean=mean,
            std=None if not self.n else max(0., self.square/self.n-mean**2)**.5,
            minimum=self.low, maximum=self.high)


def regions(target, feature, resized_hw, stride):
    height, width = feature.shape[-2:]
    yy, xx = torch.meshgrid((torch.arange(height)+.5)*stride,
                            (torch.arange(width)+.5)*stride, indexing='ij')
    inside = (yy < resized_hw[0]) & (xx < resized_hw[1])
    count = torch.zeros((height, width), dtype=torch.int64)
    for box in target:
        x, y, w, h = box[1:]
        count += (xx >= x-w/2) & (xx < x+w/2) & (yy >= y-h/2) & (yy < y+h/2)
    return inside, count


def anchor_regions(reference, target):
    boxes = reference[0, :, :4]
    ground = target[:, 1:]
    low, high = boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2
    gl, gh = ground[:, :2]-ground[:, 2:]/2, ground[:, :2]+ground[:, 2:]/2
    intersection = (torch.minimum(high[:, None], gh[None])-torch.maximum(low[:, None], gl[None])).clamp_min(0).prod(-1)
    union = boxes[:, 2:].prod(-1)[:, None]+ground[:, 2:].prod(-1)[None]-intersection
    overlaps = intersection/union.clamp_min(1e-12)
    maximum = overlaps.amax(1) if len(ground) else boxes.new_zeros(len(boxes))
    return dict(foreground=maximum >= .5, far_background=maximum <= .2), overlaps >= .5


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.monotonic()
    specification = read(SPEC)
    spec_sha = sha(SPEC)
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    assert sha(plan_path) == specification['main_plan_sha256']
    plan = read(plan_path)
    work = Path(plan['work_root'])
    require_host_memory(4*2**30, maximum_workers=1)
    require_storage(work)
    output = work/'structural_train_mechanism_20261125.json'
    partial = work/'structural_train_mechanism_20261125.state.json'
    assert not output.exists() and not partial.exists()
    full_path = ROOT/'configs/perceptual_structural_fulltrain_20261123.json'
    assert sha(full_path) == specification['structural_fulltrain_specification_sha256']
    config = read(full_path)
    a = config['architecture']
    names = ['tools/diagnose_structural_train_mechanism.py',
        'yolox/models/structural_factorial_repair.py', 'yolox/models/structural_relative_repair.py',
        'yolox/models/frozen_head_repair.py', 'yolox/models/yolo_head.py',
        'tools/native_pyramid_training_loop.py', 'tools/neck72_runtime.py']
    sources = {name: sha(ROOT/name) for name in names}
    state = dict(status='RUNNING_TRAIN_ONLY_READ_ONLY_MECHANISM_DIAGNOSIS',
        specification_sha256=spec_sha, source_hashes=sources, completed_sequences=[],
        completed_frames=0, physical_GPU_work=False, model_fit=False)
    def save_state():
        temporary = partial.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(partial)
    save_state()
    source_checkpoint = Path(specification['RGB_source_checkpoint'])
    assert sha(source_checkpoint) == specification['RGB_source_sha256']
    saved = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
    head = YOLOXHead(1, width=1.25).float().eval().requires_grad_(False)
    head.load_state_dict({k.removeprefix('head.'): v for k, v in saved['model'].items() if k.startswith('head.')}, strict=True)
    del saved
    head_sha = state_sha256(head.state_dict())
    models = {}
    for case in specification['train']['cases']:
        assert sha(case['checkpoint']) == case['checkpoint_sha256']
        assert sha(case['complete_manifest']) == case['complete_manifest_sha256']
        payload = torch.load(case['checkpoint'], map_location='cpu', weights_only=False)
        assert payload['completed_epochs'] == case['macro_snapshot']
        assert payload['config_sha256'] == sha(full_path)
        module = StructuralFactorial(latent=a['latent_channels'], radius=a['local_radius'],
            temperature=a['structural_temperature'], bound=a['feature_residual_bound_RMS_fraction'],
            risk_threshold=a['teacher_risk_threshold'], tolerance=a['forward_backward_tolerance_cells'],
            components=case['components']).float().eval().requires_grad_(False)
        module.load_state_dict(payload['adapter'], strict=True)
        del payload
        assert sum(p.numel() for p in module.parameters()) == 145600
        models[case['id']] = module
    before = {key: state_sha256(module.state_dict()) for key, module in models.items()}
    ir_case = next(c for c in specification['train']['cases'] if c['id'] == 'ir_111_e72')
    content_info = read(ir_case['complete_manifest'])['source_image_content_manifest']
    assert sha(content_info['path']) == content_info['sha256']
    content = {(r['sequence'], r['frame'], r['modality']): r for r in read(content_info['path'])}
    parity_path = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_TRAIN_STREAM_INPUT_PARITY_GPU5.json'
    parity = read(parity_path)
    assert parity['status'] == 'COMPLETE_TRAIN_INPUT_PARITY_PROBE'
    assert len(parity['rows']) == 12 and all(r['all3cue_3level_B1_FPNs_exact'] for r in parity['rows'])
    original_parity = {(r['sequence'], r['frame']): r for r in parity['rows']}
    sample_source_images, previously_verified_sample_points = [], 0
    stats = defaultdict(Moments)
    support = defaultdict(lambda: dict(GT=0, GT_without_eligible_RGB_anchor=0,
        thresholds={str(t): dict(reference=0, current=0, gained=0, lost=0) for t in [.1, .6, .7]}))
    cache_sources, verified_arrays, frame_records = {}, {}, []
    with torch.no_grad():
        for sample in specification['train']['sample']:
            sequence = sample['sequence']
            assert sha(sample['feature_manifest']) == sample['feature_manifest_sha256']
            assert sha(sample['structure_manifest']) == sample['structure_manifest_sha256']
            meta, structural = read(sample['feature_manifest']), read(sample['structure_manifest'])
            assert not meta['Dev_rows'] and not meta['Test_rows'] and not structural['GT_read']
            assert structural['parent_cache_manifest_sha256'] == sample['feature_manifest_sha256']
            assert meta['source_RGB_sha256'] == specification['RGB_source_sha256']
            maps = {}
            for key, info in {**meta['outputs'], 'targets': meta['targets'], **structural['outputs']}.items():
                assert sha(info['path']) == info['sha256']
                maps[key] = np.load(info['path'], mmap_mode='r')
                verified_arrays[info['path']] = info['sha256']
            cache_sources[sequence] = dict(feature_manifest_sha256=sample['feature_manifest_sha256'],
                structure_manifest_sha256=sample['structure_manifest_sha256'], verified_arrays=len(maps))
            for index, frame in zip(sample['indices'], sample['frame_ids']):
                assert meta['frame_ids'][index] == frame
                for modality, image in [('rgb', Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'),
                                        ('ir', Path(plan['generated_ir_root'])/'train'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png')]:
                    recorded = content[sequence, frame, modality]
                    assert image.stat().st_size == recorded['bytes'] and sha(image) == recorded['sha256']
                    sample_source_images.append(dict(sequence=sequence, frame=frame, modality=modality,
                        bytes=recorded['bytes'], sha256=recorded['sha256']))
                if (sequence, frame) in original_parity:
                    old = original_parity[sequence, frame]
                    assert old['cache_manifest_sha256'] == sample['feature_manifest_sha256']
                    assert old['structure_manifest_sha256'] == sample['structure_manifest_sha256']
                    previously_verified_sample_points += 1
                def tensor(name):
                    return torch.from_numpy(np.array(maps[name][index], copy=True)).float()
                inputs = {cue: tuple(tensor(f'{cue}_P{level}')[None] for level in [3, 4, 5]) for cue in ['rgb', 'gray', 'ir']}
                structure = {cue: (tensor(cue+'_descriptor')[None], tensor(cue+'_valid')[None]) for cue in ['rgb', 'ir']}
                target = tensor('targets')
                target = target[(target[:, 3] > 0) & (target[:, 4] > 0)]
                assert len(target)
                reference, metadata = head_predictions(head, inputs['rgb'])
                masks, eligible = anchor_regions(reference, target)
                reference_probability = reference[0, :, 4:].sigmoid().prod(-1)
                reference_support = torch.where(eligible, reference_probability[:, None], 0.).amax(0)
                resized_hw = [int(v*meta['ratios'][index]) for v in meta['image_shapes'][index]]
                confidence_cache = {}
                for case in specification['train']['cases']:
                    identifier, cue = case['id'], case['cue']
                    module = models[identifier]
                    query, qv = structure['rgb']
                    key, kv = structure['ir' if cue == 'ir' else 'rgb']
                    rgb, auxiliary = inputs['rgb'], inputs[cue]
                    current = module(rgb, auxiliary, metadata['scores'], query, key, qv, kv)
                    null = module(rgb, auxiliary, metadata['scores'], query, key, qv, kv, null=True)
                    raw_current = head_predictions(head, current)[0]
                    raw_null = head_predictions(head, null)[0]
                    prediction = relative_semantics(raw_current, raw_null, reference,
                        centered=module.centered, bound=a['semantic_logit_bound'])
                    assert torch.equal(prediction[:, :, :4], reference[:, :, :4])
                    probability = prediction[0, :, 4:].sigmoid().prod(-1)
                    assert bool(torch.isfinite(probability).all())
                    current_support = torch.where(eligible, probability[:, None], 0.).amax(0)
                    for scope in [sequence, 'ALL_TRAIN_SAMPLE']:
                        row = support[identifier, scope]
                        row['GT'] += len(target)
                        row['GT_without_eligible_RGB_anchor'] += int((~eligible.any(0)).sum())
                        stats[identifier, scope, 'GT_anchor_support', 'reference_probability'].add(reference_support)
                        stats[identifier, scope, 'GT_anchor_support', 'current_probability'].add(current_support)
                        for threshold in [.1, .6, .7]:
                            old, new = reference_support >= threshold, current_support >= threshold
                            counts = row['thresholds'][str(threshold)]
                            counts['reference'] += int(old.sum())
                            counts['current'] += int(new.sum())
                            counts['gained'] += int((new & ~old).sum())
                            counts['lost'] += int((old & ~new).sum())
                        for region, mask in masks.items():
                            values = dict(probability_delta=probability-reference_probability,
                                objectness_logit_delta=prediction[0, :, 4]-reference[0, :, 4],
                                class_logit_delta=prediction[0, :, 5]-reference[0, :, 5],
                                absolute_current_null_objectness_delta=(raw_current[0, :, 4]-raw_null[0, :, 4]).abs(),
                                absolute_current_null_class_delta=(raw_current[0, :, 5]-raw_null[0, :, 5]).abs())
                            for name, value in values.items():
                                stats[identifier, scope, 'anchors_'+region, name].add(value[mask])
                    for level, (layer, source, modified, score) in enumerate(zip(module.levels, rgb, current, metadata['scores']), 3):
                        probability_grid = score[0, 0]
                        inside, occupancy = regions(target, source, resized_hw, 2**level)
                        spatial_masks = dict(in_image=inside, background=inside & (occupancy == 0),
                            single_box=inside & (occupancy == 1), overlap_boxes=inside & (occupancy >= 2),
                            foreground_teacher_lt06=inside & (occupancy > 0) & (probability_grid < .6),
                            foreground_teacher_ge06=inside & (occupancy > 0) & (probability_grid >= .6))
                        assert int(spatial_masks['background'].sum()+spatial_masks['single_box'].sum()+spatial_masks['overlap_boxes'].sum()) == int(inside.sum())
                        ck = cue, case['components'], level
                        if ck not in confidence_cache:
                            size = source.shape[-2:]
                            _, confidence = structural_weights(F.adaptive_avg_pool2d(query, size),
                                F.adaptive_avg_pool2d(key, size), F.adaptive_avg_pool2d(qv, size) >= .5,
                                F.adaptive_avg_pool2d(kv, size) >= .5, layer.radius, layer.temperature,
                                layer.cycle, layer.use_structure, layer.tolerance)
                            confidence_cache[ck] = confidence[0, 0]
                        confidence = confidence_cache[ck]
                        risk = (1-probability_grid)*(probability_grid < layer.risk_threshold)
                        trust = confidence*risk
                        delta = modified[0]-source[0]
                        scale = source[0].square().mean(0).add(1e-6).sqrt()
                        correction = delta.square().mean(0).sqrt()/scale
                        assert bool((delta.abs().amax(0)/scale <= layer.bound*trust+2e-5).all())
                        protected = probability_grid >= layer.risk_threshold
                        assert not bool(delta[:, protected].any())
                        values = dict(structural_confidence=confidence, structural_confidence_lt01=(confidence < .1).float(),
                            teacher_risk=risk, effective_trust=trust, correction_RGB_RMS=correction,
                            nonzero_correction=(delta.abs().amax(0) > 0).float())
                        for region, mask in spatial_masks.items():
                            for scope in [sequence, 'ALL_TRAIN_SAMPLE']:
                                for name, value in values.items():
                                    stats[identifier, scope, f'P{level}_'+region, name].add(value[mask])
                frame_records.append(dict(sequence=sequence, cache_index=index, frame=frame, GT_boxes=len(target)))
                state['completed_frames'] += 1
                state['seconds'] = time.monotonic()-started
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 4*2**30
                if state['completed_frames'] % 3 == 0:
                    save_state()
                    print(json.dumps(dict(sequence=sequence, frames=state['completed_frames'], total=72,
                        seconds=state['seconds'])), flush=True)
            state['completed_sequences'].append(sequence)
            save_state()
            del maps
    assert len(frame_records) == 72 and not torch.cuda.is_initialized()
    assert head_sha == state_sha256(head.state_dict())
    assert all(before[key] == state_sha256(module.state_dict()) for key, module in models.items())
    assert all(p.grad is None for module in [head, *models.values()] for p in module.parameters())
    assert spec_sha == sha(SPEC) and all(sha(ROOT/name) == value for name, value in sources.items())
    require_storage(work)
    rows = []
    for (identifier, scope, region, variable), values in sorted(stats.items()):
        rows.append(dict(case=identifier, scope=scope, region=region, variable=variable, **values.summary()))
    result = dict(status='COMPLETE_READ_ONLY_STRUCTURAL_FULL_TRAIN_MECHANISM_DIAGNOSIS',
        specification_sha256=spec_sha, source_hashes=sources, cases=specification['train']['cases'],
        Train_frame_samples=frame_records, Train_unique_frames=72, Train_domains=6,
        cache_sources=cache_sources, verified_input_arrays=verified_arrays,
        current_raw_source_images_matched_to_completed_IR_fit=sample_source_images,
        completed_IR_fit_content_manifest_sha256=content_info['sha256'],
        historical_GPU_B1_cache_parity_proof_sha256=sha(parity_path),
        historical_GPU_B1_cache_parity_total_frames=12,
        current_sample_frames_already_in_historical_GPU_parity=previously_verified_sample_points,
        new_full72_frame_encoder_parity=False,
        pixel_and_anchor_statistics=rows,
        GT_anchor_support=[dict(case=k[0], scope=k[1], **v) for k, v in sorted(support.items())],
        original_head_and_all_adapters_unchanged=True, parameters_and_gradients_unchanged=True,
        fixed_seed=0, additional_seed_work=False, physical_GPU_work=False,
        actual_new_image_encoding=False, new_image_or_FPN_disk_cache=False, saved_weights=False,
        model_fit=False, Dev_read=False, Test_read=False, new_tracking_metrics=False,
        CPU_FP32_not_GPU_parity=True, seconds=time.monotonic()-started,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        scope='72 existing sampled Train frames, not held-out efficacy or full-Train measurements. '
            'Box occupancy is not segmentation/true occlusion. Fixed RGB dense anchor support beforeNMS '
            'is not detection Recall/AP/HOTA. Foreground confidence/activation describes mechanisms '
            'without proving semantic correctness, generalization, independent novelty or +5Test.',
        supported_innovations=0, goal_met=False)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    state.update(status=result['status'], seconds=result['seconds'], output_sha256=sha(output))
    save_state()
    print(json.dumps(dict(status=result['status'], frames=72, seconds=result['seconds'],
        peak_process_RSS_GiB=result['peak_process_RSS_GiB'])), flush=True)


if __name__ == '__main__':
    main()
