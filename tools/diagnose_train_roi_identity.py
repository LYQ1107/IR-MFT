#!/usr/bin/env python3
"""Read-only Train oracle-ROI retrieval for all18completed structural snapshots."""
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
import cv2
import numpy as np
import torch

from tools.fusion_memory_features import pool_p3
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_storage, require_host_memory
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.train_roi_identity import identities_and_boxes, p3_branch, normalized, geometry_scores, retrieval, add_stats, summary
from yolox.models.frozen_head_repair import head_predictions
from yolox.models.structural_factorial_repair import StructuralFactorial
from yolox.models.yolo_head import YOLOXHead

SPEC = ROOT / 'configs/perceptual_roi_identity_diagnosis_20261125.json'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('This diagnostic is CPU only')
    started = time.monotonic()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    spec = read(SPEC)
    spec_sha = sha256(SPEC)
    planpath = ROOT / 'configs/perceptual_mft_20261122.json'
    if sha256(planpath) != spec['main_plan_sha256']:
        raise ValueError('Fixed original image plan required')
    plan = read(planpath)
    protocolpath = ROOT / plan['protocol']
    protocol = read(protocolpath)
    if sha256(protocolpath) != spec['protocol_sha256']:
        raise ValueError('Only the original six Train sequences are permitted')
    work = Path(plan['work_root'])
    require_storage(work, growth_bytes=4 * 2**20)
    require_host_memory(8 * 2**30, maximum_workers=1)
    output = work / 'train_roi_identity_20261125'
    output.mkdir(exist_ok=False)
    sources = {p: sha256(ROOT / p) for p in spec['source_files']}
    state = dict(status='RUNNING_TRAIN_ONLY_ORACLE_ROI_IDENTITY_DIAGNOSIS',
        specification_sha256=spec_sha, source_hashes=sources, completed_frames=0,
        completed_sequences=[], GPU_work=False, model_fit=False, GT_split='train')
    def save():
        dump(output / 'state.json', state)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 2**30:
            raise RuntimeError('Own read-only CPU diagnostic exceeded8GiB')
    save()
    files = {str(SPEC): spec_sha}
    def bind(path, expected=None):
        path = Path(path)
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('Original diagnostic source changed: ' + str(path))
        files[str(path)] = digest
    for p in [planpath, protocolpath]:
        bind(p)
    bind(spec['training_ledger'], spec['training_ledger_sha256'])
    configpath = ROOT / 'configs/perceptual_structural_fulltrain_20261123.json'
    bind(configpath, spec['structural_config_sha256'])
    config = read(configpath)
    architecture = config['architecture']
    bind(spec['original_sample_specification'], spec['original_sample_specification_sha256'])
    samples = read(spec['original_sample_specification'])['train']['sample']
    if [s['sequence'] for s in samples] != protocol['train']:
        raise ValueError('All original six Train domains must remain')
    bind(spec['RGB_source_checkpoint'], spec['RGB_source_sha256'])
    saved = torch.load(spec['RGB_source_checkpoint'], map_location='cpu', weights_only=False)
    head = YOLOXHead(1, width=1.25).float().eval().requires_grad_(False)
    head.load_state_dict({k.removeprefix('head.'): v for k, v in saved['model'].items() if k.startswith('head.')}, strict=True)
    del saved
    head_before = state_sha256(head.state_dict())
    modules = {}
    for case in spec['cases']:
        bind(case['checkpoint'], case['checkpoint_sha256'])
        payload = torch.load(case['checkpoint'], map_location='cpu', weights_only=False)
        if (payload['completed_epochs'] != case['macro_snapshot'] or payload['config_sha256'] != sha256(configpath) or
                payload['metadata']['seed'] != 0 or payload['metadata']['cue'] != case['cue'] or
                payload['metadata']['variant'] != case['components']):
            raise ValueError('The actual registered structural snapshot must match')
        module = StructuralFactorial(latent=architecture['latent_channels'], radius=architecture['local_radius'],
            temperature=architecture['structural_temperature'], bound=architecture['feature_residual_bound_RMS_fraction'],
            risk_threshold=architecture['teacher_risk_threshold'], tolerance=architecture['forward_backward_tolerance_cells'],
            components=case['components']).float().eval().requires_grad_(False)
        module.load_state_dict(payload['adapter'], strict=True)
        modules[case['id']] = module
        del payload
    if len(modules) != 18:
        raise ValueError('All six fitted arms and all three original snapshots are mandatory')
    before = {key: state_sha256(m.state_dict()) for key, m in modules.items()}
    bind(spec['full_Train_IR_complete'], spec['full_Train_IR_complete_sha256'])
    info = read(spec['full_Train_IR_complete'])['source_image_content_manifest']
    bind(info['path'], info['sha256'])
    selected = {(s['sequence'], f, m) for s in samples for f in s['frame_ids'] for m in ['rgb', 'ir']}
    content = {(r['sequence'], r['frame'], r['modality']): r
        for r in read(info['path']) if (r['sequence'], r['frame'], r['modality']) in selected}
    total = defaultdict(dict)
    changes = defaultdict(dict)
    sequence_reports = []
    frame_rows = []
    with torch.inference_mode():
        for sample in samples:
            sequence = sample['sequence']
            bind(sample['feature_manifest'], sample['feature_manifest_sha256'])
            bind(sample['structure_manifest'], sample['structure_manifest_sha256'])
            meta, structure = read(sample['feature_manifest']), read(sample['structure_manifest'])
            if meta['Dev_rows'] or meta['Test_rows'] or structure['GT_read']:
                raise ValueError('Only fixed Train cached features and descriptors may be read')
            if (meta['source_RGB_sha256'] != spec['RGB_source_sha256'] or
                    structure['parent_cache_manifest_sha256'] != sample['feature_manifest_sha256']):
                raise ValueError('Original frozen encoder or structural cache provenance changed')
            maps = {}
            for key, value in {**meta['outputs'], 'targets': meta['targets'], **structure['outputs']}.items():
                bind(value['path'], value['sha256'])
                maps[key] = np.load(value['path'], mmap_mode='r')
            if any(maps[f'{cue}_P{level}'].dtype != np.float16 for cue in ['rgb', 'gray', 'ir'] for level in [3, 4, 5]):
                raise ValueError('Use the original pinned FP16FPNcache')
            gtpath = Path(plan['evaluation']['GT_root']) / 'train' / sequence / 'gt/gt.txt'
            bind(gtpath, protocol['derived_gt_sha256']['train/' + sequence])
            if files[str(gtpath)] != meta['GT_sha256']:
                raise ValueError('Cached and current canonical Train GT differ')
            gt = np.loadtxt(gtpath, delimiter=',', ndmin=2)
            frames = []
            for index, frame in zip(sample['indices'], sample['frame_ids']):
                if meta['frame_ids'][index] != frame:
                    raise ValueError('Do not change the previously frozen72Train samples')
                for modality, path in [('rgb', Path(plan['data_root']) / 'train' / sequence / 'visible' / f'{frame:06d}.jpg'),
                        ('ir', Path(plan['generated_ir_root']) / 'train' / sequence / 'infrared_diffv2ir' / f'{frame:06d}.png')]:
                    row = content[sequence, frame, modality]
                    bind(path, row['sha256'])
                ir = cv2.imread(str(Path(plan['generated_ir_root']) / 'train' / sequence / 'infrared_diffv2ir' / f'{frame:06d}.png'), -1)
                availability = bool(ir.max() != ir.min())
                del ir
                def tensor(name):
                    return torch.from_numpy(np.array(maps[name][index], copy=True)).float()
                inputs = {cue: tuple(tensor(f'{cue}_P{level}')[None] for level in [3, 4, 5])
                    for cue in ['rgb', 'gray', 'ir']}
                descriptors = {cue: (tensor(cue + '_descriptor')[None], tensor(cue + '_valid')[None])
                    for cue in ['rgb', 'ir']}
                ids, boxes = identities_and_boxes(gt, frame, meta['ratios'][index], np.array(maps['targets'][index]))
                embeddings = {}
                def pool(value):
                    pooled, valid = pool_p3(value, boxes, list(meta['image_shapes'][index]), config['training_input_hw'],
                        meta['ratios'][index], normalize=False)
                    return pooled.numpy().copy(), valid.numpy().copy()
                for cue in ['rgb', 'gray', 'ir']:
                    pooled, valid = pool(inputs[cue][0])
                    values, norm_valid = normalized(pooled)
                    embeddings['raw_' + cue] = values, valid & norm_valid
                rgb_pooled = pool(inputs['rgb'][0])[0]
                _, prediction_info = head_predictions(head, inputs['rgb'])
                query, qvalid = descriptors['rgb']
                for case in spec['cases']:
                    identifier, cue = case['id'], case['cue']
                    key, kvalid = descriptors['ir' if cue == 'ir' else 'rgb']
                    args = (modules[identifier], inputs['rgb'][0], inputs[cue][0], prediction_info['scores'][0],
                        query, key, qvalid, kvalid, availability if cue == 'ir' else True)
                    current, current_valid = pool(p3_branch(*args, null=False))
                    null, null_valid = pool(p3_branch(*args, null=True))
                    for kind, vector, valid in [('current', current, current_valid), ('null', null, null_valid),
                            ('pooled_delta_proxy', current - null, current_valid & null_valid)]:
                        values, norm_valid = normalized(vector)
                        embeddings[identifier + ':' + kind] = values, valid & norm_valid
                    rgb_norm = np.linalg.norm(rgb_pooled, axis=1)
                    good = rgb_norm > 1e-12
                    stats = dict(ROI_count=int(good.sum()),
                        current_change_relative_RGB_norm_sum=float((np.linalg.norm(current[good]-rgb_pooled[good], axis=1)/rgb_norm[good]).sum()),
                        current_null_relative_RGB_norm_sum=float((np.linalg.norm(current[good]-null[good], axis=1)/rgb_norm[good]).sum()))
                    add_stats(changes[sequence, identifier], stats)
                    add_stats(changes['ALL_TRAIN_SAMPLE', identifier], stats)
                frames.append(dict(frame=frame, ids=ids, boxes=boxes, embeddings=embeddings))
                frame_rows.append(dict(sequence=sequence, frame=frame, oracle_GT_ROIs=len(ids), IR_available=availability))
                state['completed_frames'] += 1
                save()
            local = defaultdict(dict)
            pairs = []
            for lag in spec['lag_in_registered_sample_positions']:
                for t in range(lag, len(frames)):
                    old, current = frames[t-lag], frames[t]
                    if old['frame'] >= current['frame']:
                        raise ValueError('Gallery must come strictly before the query frame')
                    all_q = np.ones(len(current['ids']), bool)
                    all_g = np.ones(len(old['ids']), bool)
                    geometry, geometric_correct, known = retrieval(geometry_scores(current['boxes'], old['boxes']),
                        current['ids'], old['ids'], all_q, all_g)
                    _, rgb_correct, rgb_known = retrieval(current['embeddings']['raw_rgb'][0] @ old['embeddings']['raw_rgb'][0].T,
                        current['ids'], old['ids'], current['embeddings']['raw_rgb'][1], old['embeddings']['raw_rgb'][1])
                    np.testing.assert_array_equal(known, rgb_known)
                    values = {'geometry_only_center_distance': (geometry, geometric_correct)}
                    for name, (query_values, query_valid) in current['embeddings'].items():
                        gallery_values, gallery_valid = old['embeddings'][name]
                        result, correct, eligible = retrieval(query_values @ gallery_values.T,
                            current['ids'], old['ids'], query_valid, gallery_valid)
                        np.testing.assert_array_equal(eligible, known)
                        values[name] = result, correct
                    for name, (result, correct) in values.items():
                        result.update(corrected_RGB_errors=int((correct & ~rgb_correct).sum()),
                            reversed_RGB_correct=int((~correct & rgb_correct).sum()),
                            geometry_errors=int((~geometric_correct).sum()),
                            correct_where_geometry_wrong=int((correct & ~geometric_correct).sum()),
                            wrong_where_geometry_correct=int((~correct & geometric_correct).sum()))
                        add_stats(local[lag, name], result)
                        add_stats(total[lag, name], result)
                    pairs.append(dict(lag_in_registered_samples=lag, past_frame=old['frame'], current_frame=current['frame'],
                        actual_frame_gap=current['frame']-old['frame'], gallery_ROIs=len(old['ids']),
                        query_ROIs=len(current['ids']), known_identity_queries=len(known)))
            rows = [dict(sequence=sequence, lag_in_registered_samples=lag, descriptor=name, **summary(v))
                for (lag, name), v in sorted(local.items())]
            path = output / (sequence + '.json')
            dump(path, dict(sequence=sequence, rows=rows, actual_frame_pairs=pairs))
            sequence_reports.append(dict(sequence=sequence, path=str(path), sha256=sha256(path)))
            state['completed_sequences'].append(sequence)
            print(json.dumps(dict(sequence=sequence, frames=len(frames), rows=len(rows), seconds=time.monotonic()-started)), flush=True)
            del maps, frames, inputs, embeddings
            save()
    if (head_before != state_sha256(head.state_dict()) or any(before[k] != state_sha256(m.state_dict()) for k, m in modules.items()) or
            any(p.grad is not None for m in [head, *modules.values()] for p in m.parameters())):
        raise ValueError('The learned weights, BN or gradients changed in this read-only diagnostic')
    for path, digest in files.items():
        if sha256(path) != digest:
            raise ValueError('Original source changed across the read-only diagnostic')
    if sha256(SPEC) != spec_sha or any(sha256(ROOT / p) != s for p, s in sources.items()) or torch.cuda.is_initialized():
        raise ValueError('Diagnostic source or CPU-only scope changed')
    results = [dict(lag_in_registered_samples=lag, descriptor=name, **summary(v)) for (lag, name), v in sorted(total.items())]
    result = dict(status='COMPLETE_TRAIN_ONLY_ORACLE_ROI_IDENTITY_DIAGNOSIS_NOT_ONLINE_TRACKING',
        specification_sha256=spec_sha, source_hashes=sources, original_source_file_hashes=files,
        actual_fitted_snapshot_count=len(modules), actual_Train_sample_frames=state['completed_frames'],
        actual_frame_records=frame_rows, aggregate_retrieval=results, sequence_reports=sequence_reports,
        ROI_change_statistics=[dict(sequence=s, model=m, **v) for (s, m), v in sorted(changes.items())],
        GT_read=True, GT_split='train', GT_role='Oracle boxes for ROI pooling and identity labels for diagnostic retrieval only',
        GT_as_fusion_network_input=False, Dev_rows=0, Test_rows=0, weights_BN_gradients_unchanged=True,
        actual_device='cpu', feature_and_head_precision='FP32 CPU from pinned FP16FPNcache and restored checkpoint values',
        source_RGB_gray_old_Train_byte_delivery_not_retroactively_certified=True,
        new_fit=False, new_GPU_work=False, new_image_FPN_or_ROI_disk_cache=False, new_tracking_HOTA=False,
        fixed_seed=0, additional_seed_work=False, goal_met=False, supported_innovations=0,
        seconds=time.monotonic()-started, peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        scope=spec['scope'])
    dump(output / 'complete.json', result)
    state.update(status=result['status'], result_sha256=sha256(output/'complete.json'))
    save()
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'], RSS_GiB=result['peak_process_RSS_GiB'])), flush=True)


if __name__ == '__main__':
    main()
