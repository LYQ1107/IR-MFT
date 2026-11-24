#!/usr/bin/env python3
"""Train-only current-versus-past cue utility, never a tracking evaluation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
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

from tools.diagnose_structural_train_mechanism import Moments, anchor_regions, read, sha
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage
from yolox.models.frozen_head_repair import head_predictions
from yolox.models.structural_factorial_repair import StructuralFactorial
from yolox.models.structural_relative_repair import relative_semantics
from yolox.models.yolo_head import YOLOXHead


def contrast_auc(positive, negative):
    """P(positive > negative) + half ties, using every finite anchor contrast."""
    positive, negative = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    if positive.ndim != 1 or negative.ndim != 1 or not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError('Finite one-dimensional contrast samples required')
    if not len(positive) or not len(negative):
        return None
    ordered = np.sort(negative)
    low = np.searchsorted(ordered, positive, side='left')
    high = np.searchsorted(ordered, positive, side='right')
    return float((low.astype(np.float64) + .5*(high-low)).sum()/(len(positive)*len(negative)))


def support_counts(reference, current, past, threshold):
    values = [np.asarray(v, dtype=np.float64) for v in [reference, current, past]]
    if not 0 <= threshold <= 1 or any(v.ndim != 1 or v.shape != values[0].shape or
            not np.isfinite(v).all() or (v < 0).any() or (v > 1).any() for v in values):
        raise ValueError('Corresponding finite per-GT probabilities and fixed threshold required')
    r, c, p = [v >= threshold for v in values]
    return dict(reference=int(r.sum()), current=int(c.sum()), past=int(p.sum()),
        current_gain_vs_reference=int((c & ~r).sum()), current_loss_vs_reference=int((r & ~c).sum()),
        current_gain_vs_past=int((c & ~p).sum()), current_loss_vs_past=int((p & ~c).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2); torch.manual_seed(0)
    started = time.monotonic(); spec = read(args.specification); spec_sha = sha(args.specification)
    sources = spec['source_hashes']
    assert all(sha(ROOT/p) == s for p, s in sources.items())
    original_path = ROOT/spec['original_diagnosis']; assert sha(original_path) == spec['original_diagnosis_sha256']
    original = read(original_path)
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    assert sha(plan_path) == original['main_plan_sha256']
    plan = read(plan_path); work = Path(plan['work_root'])
    require_host_memory(4*2**30, maximum_workers=1); require_storage(work)
    output = Path(spec['output_directory']); output.mkdir(exist_ok=False)
    state = dict(status='RUNNING_TRAIN_CURRENT_PAST_CUE_SPECIFICITY', source_hashes=sources,
        specification_sha256=spec_sha, pid=os.getpid(), completed_frames=0, completed_sequences=[],
        model_fit=False, physical_GPU_work=False, Dev_read=False, Test_read=False)
    def save_state():
        temporary = output/'state.tmp.json'; temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(output/'state.json')
    save_state()
    full_path = ROOT/'configs/perceptual_structural_fulltrain_20261123.json'
    assert sha(full_path) == original['structural_fulltrain_specification_sha256']
    architecture = read(full_path)['architecture']
    checkpoint = Path(original['RGB_source_checkpoint']); assert sha(checkpoint) == original['RGB_source_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    head = YOLOXHead(1, width=1.25).float().eval().requires_grad_(False)
    head.load_state_dict({k.removeprefix('head.'):v for k,v in saved['model'].items() if k.startswith('head.')}, strict=True)
    del saved
    models = {}; head_before = state_sha256(head.state_dict())
    for case in spec['cases']:
        assert sha(case['checkpoint']) == case['checkpoint_sha256'] and sha(case['complete_manifest']) == case['complete_manifest_sha256']
        saved = torch.load(case['checkpoint'], map_location='cpu', weights_only=False)
        assert saved['completed_epochs'] == 72 and saved['config_sha256'] == sha(full_path)
        module = StructuralFactorial(latent=architecture['latent_channels'], radius=architecture['local_radius'],
            temperature=architecture['structural_temperature'], bound=architecture['feature_residual_bound_RMS_fraction'],
            risk_threshold=architecture['teacher_risk_threshold'], tolerance=architecture['forward_backward_tolerance_cells'],
            components='111').float().eval().requires_grad_(False)
        module.load_state_dict(saved['adapter'], strict=True); del saved
        assert sum(p.numel() for p in module.parameters()) == 145600
        models[case['id']] = module
    before = {k:state_sha256(v.state_dict()) for k,v in models.items()}
    ir_case = next(c for c in spec['cases'] if c['cue'] == 'ir')
    content_info = read(ir_case['complete_manifest'])['source_image_content_manifest']
    assert sha(content_info['path']) == content_info['sha256']
    content = {(r['sequence'],r['frame'],r['modality']):r for r in read(content_info['path'])}
    verified_images = {}; verified_arrays = {}; moments = defaultdict(Moments)
    contrasts = defaultdict(list); counts = defaultdict(lambda:defaultdict(int)); frame_rows = []
    with torch.no_grad():
        for sample in spec['samples']:
            sequence = sample['sequence']
            assert sequence in read(ROOT/plan['protocol'])['train']
            assert sha(sample['feature_manifest']) == sample['feature_manifest_sha256']
            assert sha(sample['structure_manifest']) == sample['structure_manifest_sha256']
            meta, structure_meta = read(sample['feature_manifest']), read(sample['structure_manifest'])
            assert not meta['Dev_rows'] and not meta['Test_rows'] and not structure_meta['GT_read']
            assert meta['source_RGB_sha256'] == original['RGB_source_sha256']
            assert structure_meta['parent_cache_manifest_sha256'] == sample['feature_manifest_sha256']
            maps = {}
            for name, info in {**meta['outputs'], 'targets':meta['targets'], **structure_meta['outputs']}.items():
                assert sha(info['path']) == info['sha256']
                maps[name] = np.load(info['path'], mmap_mode='r'); verified_arrays[info['path']] = info['sha256']
            def tensor(name, index):
                return torch.from_numpy(np.array(maps[name][index], copy=True)).float()
            for index, frame in zip(sample['indices'], sample['frame_ids']):
                past_index = 0; past_frame = meta['frame_ids'][past_index]
                assert index > 0 and meta['frame_ids'][index] == frame and past_frame < frame
                for number in [frame, past_frame]:
                    for modality, image in [('rgb',Path(plan['data_root'])/'train'/sequence/'visible'/f'{number:06d}.jpg'),
                            ('ir',Path(plan['generated_ir_root'])/'train'/sequence/'infrared_diffv2ir'/f'{number:06d}.png')]:
                        key = (sequence,number,modality)
                        if key not in verified_images:
                            recorded = content[key]
                            assert image.stat().st_size == recorded['bytes'] and sha(image) == recorded['sha256']
                            verified_images[key] = dict(sequence=sequence,frame=number,modality=modality,sha256=recorded['sha256'])
                rgb = tuple(tensor(f'rgb_P{level}',index)[None] for level in [3,4,5])
                query, qv = tensor('rgb_descriptor',index)[None],tensor('rgb_valid',index)[None]
                reference, metadata = head_predictions(head,rgb)
                predictions = {}
                # Produce all three matched cues before using this frame's Train labels.
                for case in spec['cases']:
                    cue = case['cue']; module = models[case['id']]
                    for condition, source_index in [('current',index),('past',past_index)]:
                        auxiliary = tuple(tensor(f'{cue}_P{level}',source_index)[None] for level in [3,4,5])
                        descriptor_cue = 'ir' if cue == 'ir' else 'rgb'
                        key = tensor(descriptor_cue+'_descriptor',source_index)[None]
                        kv = tensor(descriptor_cue+'_valid',source_index)[None]
                        normal = module(rgb,auxiliary,metadata['scores'],query,key,qv,kv)
                        null = module(rgb,auxiliary,metadata['scores'],query,key,qv,kv,null=True)
                        prediction = relative_semantics(head_predictions(head,normal)[0],head_predictions(head,null)[0],
                            reference,centered=True,bound=architecture['semantic_logit_bound'])
                        assert torch.equal(prediction[:,:,:4],reference[:,:,:4])
                        predictions[case['id'],condition] = prediction[0,:,4:].sigmoid().prod(-1)
                        del normal,null,prediction
                target = tensor('targets',index); target = target[(target[:,3] > 0) & (target[:,4] > 0)]
                assert len(target)
                masks, eligible = anchor_regions(reference,target)
                reference_probability = reference[0,:,4:].sigmoid().prod(-1)
                masks['weak_foreground'] = masks['foreground'] & (reference_probability < .6)
                masks['strong_foreground'] = masks['foreground'] & (reference_probability >= .6)
                reference_support = torch.where(eligible,reference_probability[:,None],0.).amax(0)
                for case in spec['cases']:
                    identifier = case['id']; current,past = (predictions[identifier,c] for c in ['current','past'])
                    contrast = current-past
                    current_support = torch.where(eligible,current[:,None],0.).amax(0)
                    past_support = torch.where(eligible,past[:,None],0.).amax(0)
                    per_frame = dict(case=identifier,sequence=sequence,frame=frame,past_frame=past_frame,
                        GT_boxes=len(target),GT_without_eligible_RGB_anchor=int((~eligible.any(0)).sum()),thresholds={})
                    for threshold in spec['fixed_support_thresholds']:
                        item = support_counts(reference_support.numpy(),current_support.numpy(),past_support.numpy(),threshold)
                        per_frame['thresholds'][str(threshold)] = item
                        for scope in [sequence,'ALL_TRAIN_SAMPLE']:
                            for key,value in item.items():counts[identifier,scope,str(threshold)][key] += value
                    for region, mask in masks.items():
                        values = dict(reference_probability=reference_probability,current_probability=current,past_probability=past,
                            current_minus_past=contrast,current_minus_reference=current-reference_probability)
                        for scope in [sequence,'ALL_TRAIN_SAMPLE']:
                            for name,value in values.items():moments[identifier,scope,region,name].add(value[mask])
                            if region in ['foreground','far_background']:
                                contrasts[identifier,scope,region].append(contrast[mask].numpy().copy())
                    frame_rows.append(per_frame)
                state['completed_frames'] += 1; state['seconds'] = time.monotonic()-started
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 4*2**30
                save_state()
                print(json.dumps(dict(sequence=sequence,frames=state['completed_frames'],total=66,seconds=state['seconds'])),flush=True)
            state['completed_sequences'].append(sequence);save_state();del maps
    assert state['completed_frames'] == 66 and len(frame_rows) == 198 and not torch.cuda.is_initialized()
    assert state_sha256(head.state_dict()) == head_before
    assert all(state_sha256(m.state_dict()) == before[k] and all(p.grad is None for p in m.parameters()) for k,m in models.items())
    assert all(p.grad is None for p in head.parameters())
    assert sha(args.specification) == spec_sha and all(sha(ROOT/p) == s for p,s in sources.items())
    auc = []
    for case in spec['cases']:
        for scope in [s['sequence'] for s in spec['samples']]+['ALL_TRAIN_SAMPLE']:
            positive = np.concatenate(contrasts[case['id'],scope,'foreground'])
            negative = np.concatenate(contrasts[case['id'],scope,'far_background'])
            auc.append(dict(case=case['id'],scope=scope,foreground_anchors=len(positive),background_anchors=len(negative),
                current_minus_past_foreground_ranking_AUC=contrast_auc(positive,negative)))
    result = dict(status='COMPLETE_TRAIN_CURRENT_PAST_CUE_SPECIFICITY_NOT_TRACKING_EFFICACY',UTC=datetime.now(timezone.utc).isoformat(),
        specification_sha256=spec_sha,source_hashes=sources,cases=spec['cases'],unique_current_frames=66,Train_domains=6,
        verified_input_arrays=verified_arrays,raw_current_and_past_source_images=list(verified_images.values()),
        completed_IR_content_manifest_sha256=content_info['sha256'],frame_support=frame_rows,
        anchor_statistics=[dict(case=k[0],scope=k[1],region=k[2],variable=k[3],**v.summary()) for k,v in sorted(moments.items())],
        aggregated_support=[dict(case=k[0],scope=k[1],threshold=k[2],**v) for k,v in sorted(counts.items())],contrast_AUC=auc,
        source_head_and_all_adapters_unchanged=True,all_conditions_before_frame_GT_indexing=True,
        model_fit=False,new_GPU_work=False,Dev_read=False,Test_read=False,new_tracking_metrics=False,
        seconds=time.monotonic()-started,peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        scope=spec['scope'],supported_innovations=0,goal_met=False)
    require_storage(work)
    final=output/'complete.json';final.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    state.update(status=result['status'],result_sha256=sha(final),seconds=result['seconds']);save_state()
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],peak_process_RSS_GiB=result['peak_process_RSS_GiB'],output=str(final))),flush=True)


if __name__ == '__main__':main()
