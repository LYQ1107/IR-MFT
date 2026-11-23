#!/usr/bin/env python3
"""Train-only classical-structure probe; all predictions frozen before GT."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from tools.structural_correspondence import match, shifted
from tools.run_pseudo_ir_comparison import dump, sha256


def support(labels, yy, xx, height, width):
    mask = np.zeros_like(yy, dtype=bool)
    for x, y, w, h in labels:
        mask |= ((xx >= x*width/1920) & (xx < (x+w)*width/1920) &
                 (yy >= y*height/1080) & (yy < (y+h)*height/1080))
    return mask


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    spec_path = ROOT/'configs/perceptual_structural_correspondence_20261123.json'
    plan = json.loads(plan_path.read_text()); spec = json.loads(spec_path.read_text())
    assert spec['main_plan_sha256'] == sha256(plan_path)
    assert spec['parent_plan_sha256'] == sha256(ROOT/'configs/perceptual_correlation_repair_20261123.json')
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    work = Path(plan['work_root']); out = work/'structural_correspondence'
    out.mkdir(exist_ok=False); cv2.setNumThreads(1)
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in
                     [Path(__file__), ROOT/'tools/structural_correspondence.py']}
    state = dict(status='PREDICTING_TRAIN_NO_GT', source_hashes=source_hashes,
                 specification_sha256=sha256(spec_path), physical_gpu=None, Test_read=False)
    dump(out/'state.json', state)
    height, width = spec['input_hw']; stride = spec['uniform_sample_stride']
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    packets = {}; start = time.perf_counter()
    for sequence in protocol['train']:
        cache_path = work/'correlation_cache'/sequence/'features/complete.json'
        cache = json.loads(cache_path.read_text())
        frames = [cache['frame_ids'][i] for i in spec['Train_cached_indices']]
        gray = []; ir = []
        for frame in frames:
            rgb_path = Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'
            ir_path = Path(plan['generated_ir_root'])/'train'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
            a = cv2.imread(str(rgb_path)); b = cv2.imread(str(ir_path), -1)
            assert a.shape == (1080, 1920, 3) and b.shape == (1080, 1920) and b.dtype == np.uint8
            gray.append(cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)/255)
            ir.append(cv2.resize(b, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)/255)
        arrays = {}
        dy, dx = spec['known_gray_translation_yx']
        for index, frame in enumerate(frames):
            views = dict(gray_translated=shifted(gray[index], -dy, -dx), IR_current=ir[index],
                IR_wrong_frame=ir[spec['wrong_permutation'][index]],
                IR_spatial_roll=np.roll(ir[index], (height//4, width//4), (0, 1)))
            assert list(views) == spec['views']
            for name, view in views.items():
                prediction = match(gray[index], view, spec['local_search_radius'])
                for key, value in prediction.items():
                    arrays[f'{index}_{name}_{key}'] = value[::stride, ::stride]
        path = out/(sequence+'.npz'); np.savez_compressed(path, **arrays)
        packets[sequence] = dict(path=str(path), sha256=sha256(path), frames=frames,
                                 cache_manifest_sha256=sha256(cache_path), bytes=path.stat().st_size)
        print(json.dumps(dict(sequence=sequence, frames=4, predictions_frozen=True,
                              seconds=time.perf_counter()-start)), flush=True)
    frozen = dict(status='ALL24_TRAIN_FIELDS_FROZEN_BEFORE_GT', packets=packets,
                  uniform_sample_stride=stride, GT_read=False, source=state)
    dump(out/'predictions_frozen.json', frozen)
    rows = []
    for sequence, packet in packets.items():
        assert sha256(packet['path']) == packet['sha256']
        gt_path = Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        assert sha256(gt_path) == protocol['derived_gt_sha256']['train/'+sequence]
        gt = np.loadtxt(gt_path, delimiter=',', ndmin=2)
        gt = gt[(gt[:, 6] > 0) & (gt[:, 7] == 1)]
        labels = [gt[gt[:, 0] == frame, 2:6] for frame in packet['frames']]
        with np.load(packet['path']) as arrays:
            for name in spec['views']:
                count = consistent = correct = 0; cost = zero = 0.
                for index in range(4):
                    foreground = support(labels[index], yy, xx, height, width)
                    if name.startswith('IR_'):
                        foreground &= ~support(labels[spec['wrong_permutation'][index]], yy, xx, height, width)
                    prefix = f'{index}_{name}_'
                    if name.startswith('IR_'):
                        textured = np.logical_and.reduce([arrays[f'{index}_{view}_textured']
                            for view in ['IR_current', 'IR_wrong_frame', 'IR_spatial_roll']])
                    else:
                        textured = arrays[prefix+'textured']
                    selected = foreground & textured
                    n = int(selected.sum()); count += n
                    cost += float(arrays[prefix+'cost'][selected].sum(dtype=np.float64))
                    zero += float(arrays[prefix+'zero_offset_cost'][selected].sum(dtype=np.float64))
                    consistent += int(arrays[prefix+'inverse_consistent'][selected].sum())
                    if name == 'gray_translated':
                        field = arrays[prefix+'offset'][selected]
                        correct += int(np.all(field == spec['known_gray_translation_yx'], axis=-1).sum())
                assert count > 0
                rows.append(dict(sequence=sequence, view=name, textured_bbox_grid_cells=count,
                    mean_best_cost=cost/count, mean_zero_offset_cost=zero/count,
                    inverse_consistent_fraction=consistent/count,
                    known_translation_exact_fraction=correct/count if name == 'gray_translated' else None))
    by_seq = {seq: {row['view']: row for row in rows if row['sequence'] == seq} for seq in protocol['train']}
    good_translation = sum(v['gray_translated']['known_translation_exact_fraction'] >= .8 for v in by_seq.values())
    good_cost = sum(v['IR_current']['mean_best_cost'] < v['IR_wrong_frame']['mean_best_cost'] for v in by_seq.values())
    good_cycle = sum(v['IR_current']['inverse_consistent_fraction'] > v['IR_wrong_frame']['inverse_consistent_fraction'] for v in by_seq.values())
    passed = good_translation == 6 and good_cost >= 4 and good_cycle >= 4
    report = dict(status='COMPLETE_SMALL_TRAIN_STRUCTURE_DIAGNOSTIC', rows=rows,
        prerequisite_for_further_registration=passed, translation_video_passes=good_translation,
        current_IR_cost_video_wins=good_cost, current_IR_cycle_video_wins=good_cycle,
        predictions_frozen_sha256=sha256(out/'predictions_frozen.json'), source=state,
        scope='24 Train frames only, not heldout correspondence or tracking HOTA. '
              'Class1 box support is not a pixel landmark or segmentation label. '
              'Only synthetic grayscale translation has known displacement truth; actualIR '
              'current-vs-wrong structure discrimination does not prove correct fish geometry. '
              'All3 IR views use exactly the same intersection-textured source-grid pixels. '
              'No fitting, new full cache, Test evaluation or novelty claim is activated automatically.',
        Test_read=False, goal_met=False, supported_innovations=0)
    stem = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURE_TRAIN_AUDIT'
    dump(stem.with_suffix('.json'), report)
    lines = ['# Train-only local self-similarity correspondence diagnostic', '', report['scope'], '',
        '| Video | View | Textured bbox cells | Best cost | Zero-offset cost | Cycle-consistent fraction | Known-shift exact fraction |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        exact = row['known_translation_exact_fraction']
        lines.append(f"| {row['sequence']} | {row['view']} | {row['textured_bbox_grid_cells']} | "
            f"{row['mean_best_cost']:.5f} | {row['mean_zero_offset_cost']:.5f} | "
            f"{row['inverse_consistent_fraction']:.4f} | "+('—' if exact is None else f'{exact:.4f}')+' |')
    lines += ['', f'Prerequisite: {passed}; known-translation passes {good_translation}/6, '
              f'current-vs-wrong IR cost wins {good_cost}/6 and cycle wins {good_cycle}/6.', '',
              'Classical prior: [MIND2012](https://pubmed.ncbi.nlm.nih.gov/22722056/) and '
              '[self-similarity context2013](https://iacl.ece.jhu.edu/proceedings/miccai2013/papers/8149/81490187.pdf). '
              'This8-neighbor2D probe is an inspired implementation, not the original complete registration algorithm.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    state.update(status='COMPLETE', prerequisite=passed); dump(out/'state.json', state)
    print(json.dumps({key: report[key] for key in ['status', 'prerequisite_for_further_registration',
        'translation_video_passes', 'current_IR_cost_video_wins', 'current_IR_cycle_video_wins']}))


if __name__ == '__main__':
    main()
