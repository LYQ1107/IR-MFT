#!/usr/bin/env python3
"""Measure fixed-cache Train coverage before proposing a larger sampling study."""
import configparser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol_path = ROOT/plan['protocol']
    protocol = json.loads(protocol_path.read_text())
    work = Path(plan['work_root'])
    config = json.loads((ROOT/'configs/perceptual_structural_relative_20261123.json').read_text())
    rows = []
    for seq in protocol['train']:
        folder = Path(plan['evaluation']['GT_root'])/'train'/seq
        gt_path, info_path = folder/'gt/gt.txt', folder/'seqinfo.ini'
        assert sha256(gt_path) == protocol['derived_gt_sha256']['train/'+seq]
        info = configparser.ConfigParser()
        info.read(info_path)
        frames = int(info['Sequence']['seqLength'])
        ratio = min(config['training_input_hw'][0]/int(info['Sequence']['imHeight']),
                    config['training_input_hw'][1]/int(info['Sequence']['imWidth']))
        cache_path = work/'correlation_cache'/seq/'features/complete.json'
        meta = json.loads(cache_path.read_text())
        selected = np.array(meta['frame_ids'])
        assert len(selected) == len(np.unique(selected)) == 192
        assert selected.min() >= 1 and selected.max() <= frames
        gt = np.loadtxt(gt_path, delimiter=',', ndmin=2)
        gt = gt[(gt[:,6] > 0) & (gt[:,7] == 1)]
        cached_gt = gt[np.isin(gt[:,0], selected)]
        ids, cached_ids = set(map(int,gt[:,1])), set(map(int,cached_gt[:,1]))
        fractions = {str(i): float((cached_gt[:,1] == i).sum()/(gt[:,1] == i).sum()) for i in sorted(ids)}
        small = lambda values: {str(t): float((np.minimum(values[:,4],values[:,5])*ratio < t).mean())
                                for t in [8,16,32]}
        rows.append(dict(sequence=seq, full_frames=frames, cached_unique_frames=len(selected),
            unique_frame_fraction=len(selected)/frames, full_GT_boxes=len(gt), cached_GT_boxes=len(cached_gt),
            full_IDs=len(ids), cached_IDs=len(cached_ids), missing_IDs=sorted(ids-cached_ids),
            median_cache_frame_gap=float(np.median(np.diff(selected))),
            max_cache_frame_gap=int(np.diff(selected).max()),
            full_small_box_fractions_at_training_resolution=small(gt),
            cached_small_box_fractions_at_training_resolution=small(cached_gt),
            per_ID_cached_observation_fraction=fractions,
            GT_sha256=sha256(gt_path), seqinfo_sha256=sha256(info_path), cache_manifest_sha256=sha256(cache_path)))
    full, cached = sum(r['full_frames'] for r in rows), sum(r['cached_unique_frames'] for r in rows)
    assert full == protocol['frame_counts']['train'] == 27251 and cached == 1152
    assert sum(r['full_IDs'] for r in rows) == protocol['identity_counts']['train'] == 98
    for row in rows:
        row['full_frame_sampling_share'] = row['full_frames']/full
        row['current_cache_sampling_share'] = row['cached_unique_frames']/cached
    report = dict(status='COMPLETE_TRAIN_ONLY_CACHE_COVERAGE_AUDIT', rows=rows,
        full_Train_frames=full, unique_cached_Train_frames=cached, unique_frame_fraction=cached/full,
        cache_passes=24, training_sample_exposures=24*cached,
        exposure_count_divided_by_full_Train_frames=24*cached/full,
        sampled_IDs=sum(r['cached_IDs'] for r in rows), full_IDs=98,
        protocol_sha256=sha256(protocol_path), source_sha256=sha256(__file__),
        scope='Only existing Train GT, frame counts and cache manifests are read. No images/FPNs copied, '
              'no fitting, no Dev/Test labels or accuracy. The cache was chosen before this audit.',
        interpretation='Repeated cache passes do not create new temporal/appearance observations. '
            'Coverage differences motivate a controlled sampling study, not a claim that longer training '
            'or more unique frames improve HOTA. No new training or selection policy is activated.',
        Dev_read=False, Test_read=False, goal_met=False, supported_innovations=0)
    stem = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_TRAIN_COVERAGE'
    dump(stem.with_suffix('.json'), report)
    lines = ['# Existing Train-cache coverage', '', report['scope'], '',
             '| Video | Full frames | Cached frames | Cached IDs / all IDs | Median frame gap | Full-frame share | Cache share |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['sequence']} | {r['full_frames']} | {r['cached_unique_frames']} | "
                     f"{r['cached_IDs']} / {r['full_IDs']} | {r['median_cache_frame_gap']:.1f} | "
                     f"{r['full_frame_sampling_share']:.3%} | {r['current_cache_sampling_share']:.3%} |")
    lines += ['', f'{cached}/{full} unique frames ({cached/full:.3%});24cachepasses = {24*cached}sampleexposures, '
              f'{24*cached/full:.3f}times the fullTrainframe count, but still only{cached}unique frames.', '', report['interpretation']]
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: report[k] for k in ['status','unique_frame_fraction','sampled_IDs','full_IDs']}))


if __name__ == '__main__':
    main()
