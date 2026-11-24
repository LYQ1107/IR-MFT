#!/usr/bin/env python3
"""Full Dev posthoc identity concentration; no tracking or official metric rerun."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
from yolox.tracker.matching import ious, linear_assignment

SPEC = ROOT/'configs/perceptual_dev_track_concentration_20261125.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def summarize(truth, observations):
    gt_ids = sorted(set(map(int, truth[:, 1])))
    track_ids = sorted({track for rows in observations.values() for _, track, _ in rows})
    gi, pi = {v: i for i, v in enumerate(gt_ids)}, {v: i for i, v in enumerate(track_ids)}
    table = np.zeros((len(gt_ids), len(track_ids)), dtype=np.int64)
    for identity, rows in observations.items():
        for _, track, _ in rows:
            table[gi[identity], pi[track]] += 1
    matched = int(table.sum())
    gt_mass = int(table.max(1).sum()) if len(track_ids) else 0
    track_mass = int(table.max(0).sum()) if len(track_ids) else 0
    rr, cc = linear_sum_assignment(-table)
    bijective_mass = int(table[rr, cc].sum())
    assert 0 <= bijective_mass <= min(gt_mass, track_mass) <= matched
    per_gt = []
    for identity in gt_ids:
        rows = observations[identity]
        counts = Counter(track for _, track, _ in rows)
        dominant = min(counts, key=lambda t: (-counts[t], t)) if counts else None
        pieces = []
        for frame, track, _ in rows:
            if pieces and pieces[-1]['track'] == track and pieces[-1]['end'] == frame-1:
                pieces[-1]['end'] = frame
                pieces[-1]['frames'] += 1
            else:
                pieces.append(dict(track=track, start=frame, end=frame, frames=1))
        nondominant_long = [p for p in pieces if p['track'] != dominant and p['frames'] >= 25]
        bands = {name: dict(matched=0, nondominant=0) for name in ['lt0.1', '0.1_to_lt0.3', 'ge0.3']}
        for _, track, overlap in rows:
            band = 'lt0.1' if overlap < .1 else '0.1_to_lt0.3' if overlap < .3 else 'ge0.3'
            bands[band]['matched'] += 1
            bands[band]['nondominant'] += int(track != dominant)
        changes = {key: 0 for key in ['gap1', 'gap2to5', 'gap6to25', 'gap_over25']}
        for old, current in zip(rows, rows[1:]):
            if old[1] == current[1]:
                continue
            gap = current[0]-old[0]
            assert gap > 0
            changes['gap1' if gap == 1 else 'gap2to5' if gap <= 5 else 'gap6to25' if gap <= 25 else 'gap_over25'] += 1
        frequencies = [n/len(rows) for n in counts.values()]
        per_gt.append(dict(GT_identity=identity, GT_observations=int((truth[:, 1] == identity).sum()),
            matched_observations=len(rows), distinct_tracks=len(counts), dominant_track=dominant,
            largest_track_fraction=None if not rows else counts[dominant]/len(rows),
            track_assignment_entropy_bits=None if not rows else -sum(p*math.log2(p) for p in frequencies),
            contiguous_pieces=len(pieces), longest_contiguous_piece=max((p['frames'] for p in pieces), default=0),
            nondominant_contiguous_pieces_ge25=len(nondominant_long),
            nondominant_contiguous_observations_in_pieces_ge25=sum(p['frames'] for p in nondominant_long),
            first3_nondominant_long_pieces=nondominant_long[:3],
            diagnostic_identity_changes_by_gap=changes, GT_box_overlap_proxy_bands=bands))
    assert sum(p['matched_observations'] for p in per_gt) == matched
    return dict(matched_observations=matched, GT_observations=len(truth),
        GT_with_matches=sum(p['matched_observations'] > 0 for p in per_gt),
        matched_predicted_tracks=len(track_ids),
        GT_largest_track_mass=gt_mass, GT_largest_track_fraction=None if not matched else gt_mass/matched,
        track_largest_GT_mass=track_mass, track_largest_GT_fraction=None if not matched else track_mass/matched,
        bijective_contingency_mass=bijective_mass,
        bijective_contingency_fraction=None if not matched else bijective_mass/matched,
        diagnostic_identity_changes=sum(sum(p['diagnostic_identity_changes_by_gap'].values()) for p in per_gt),
        nondominant_contiguous_observations_in_pieces_ge25=sum(p['nondominant_contiguous_observations_in_pieces_ge25'] for p in per_gt),
        per_GT=per_gt)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.monotonic()
    specification = read(SPEC)
    assert specification['fixed_seed'] == 0 and specification['sequences'] == ['BT-005', 'SN-013']
    parent = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_NATIVE_DEV_DOMAINS.json'
    assert sha(parent) == specification['parent_Dev_diagnosis_sha256']
    plan = read(ROOT/'configs/perceptual_mft_20261122.json')
    protocol = read(ROOT/plan['protocol'])
    stem = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_DEV_TRACK_CONCENTRATION'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    # Known, permutation-invariant descriptive case; these are not official identity metrics.
    known_truth = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])
    known = summarize(known_truth, defaultdict(list, {1: [(1, 11, 0.), (2, 12, 0.)],
                                                     2: [(1, 12, 0.), (2, 11, 0.)]}))
    assert known['GT_largest_track_fraction'] == known['track_largest_GT_fraction'] == .5
    assert known['bijective_contingency_fraction'] == .5
    records = []
    for role, source in specification['roles'].items():
        path = Path(source['metrics_json'])
        assert sha(path) == source['metrics_json_sha256']
        metrics = read(path)
        manifest = metrics['manifest']
        assert metrics['status'] == 'COMPLETE' and manifest['split'] == 'dev'
        assert manifest['frames'] == 4834 and manifest['sequences'] == specification['sequences']
        assert manifest['config']['native_seed'] == 0
        for sequence, frames in zip(specification['sequences'], specification['frames']):
            gt_path = Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
            assert sha(gt_path) == protocol['derived_gt_sha256']['train/'+sequence]
            truth = np.loadtxt(gt_path, delimiter=',', ndmin=2)
            truth = truth[(truth[:, 6] > 0) & (truth[:, 7] == 1)]
            prediction_path = path.parent/'track_results'/(sequence+'.txt')
            assert sha(prediction_path) == manifest['coverage'][sequence]['output_sha256']
            prediction = np.loadtxt(prediction_path, delimiter=',', ndmin=2)
            assert bool(np.isfinite(prediction).all())
            assert bool(((prediction[:, 0] >= 1) & (prediction[:, 0] <= frames)).all())
            assert bool((prediction[:, 4:6] > 0).all())
            gt_groups = {frame: truth[truth[:, 0] == frame] for frame in range(1, frames+1)}
            pred_groups = {frame: prediction[prediction[:, 0] == frame] for frame in range(1, frames+1)}
            observations = defaultdict(list)
            for frame in range(1, frames+1):
                g, p = gt_groups[frame], pred_groups[frame]
                assert len(set(p[:, 1])) == len(p)
                gb, pb = g[:, 2:6].copy(), p[:, 2:6].copy()
                gb[:, 2:] += gb[:, :2]
                pb[:, 2:] += pb[:, :2]
                overlaps = ious(gb, gb)
                if len(overlaps):
                    np.fill_diagonal(overlaps, 0.)
                proxy = overlaps.max(1) if len(g) else np.zeros(0)
                pairs, _, _ = linear_assignment(1-ious(gb, pb), .5)
                for gi, pi in pairs:
                    observations[int(g[gi, 1])].append((frame, int(p[pi, 1]), float(proxy[gi])))
            stats = summarize(truth, observations)
            records.append(dict(role=role, source_group=source['group'], sequence=sequence,
                frames=frames, GT_sha256=sha(gt_path), prediction_sha256=sha(prediction_path),
                original_official_metrics=source['metrics'][sequence], **stats))
            print(json.dumps(dict(role=role, sequence=sequence, frames=frames,
                GT_largest_track_fraction=stats['GT_largest_track_fraction'],
                track_largest_GT_fraction=stats['track_largest_GT_fraction'],
                bijective_contingency_fraction=stats['bijective_contingency_fraction'])), flush=True)
    assert sum(r['frames'] for r in records) == 3*4834
    result = dict(status='COMPLETE_POSTHOC_FULL_DEV_TRACK_IDENTITY_CONCENTRATION',
        specification_sha256=sha(SPEC), parent_Dev_diagnosis_sha256=sha(parent),
        source_hashes={name: sha(ROOT/name) for name in ['tools/diagnose_dev_track_concentration.py',
                                                       'yolox/tracker/matching.py']},
        rows=records, fixed_seed=0, additional_seed_work=False, GPU_work=False,
        new_inference=False, new_fitting=False, Test_accuracy_read=False,
        actual_Dev_GT_used_for_posthoc_labels=True, GT_used_in_predictor=False,
        original_official_metrics_unchanged=True, new_official_metrics=False,
        seconds=time.monotonic()-started,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        scope=specification['diagnosis'],
        historical_Test_exposed=True, RGB_pretraining_saw_current_Dev=True,
        supported_innovations=0, goal_met=False)
    assert result['peak_process_RSS_GiB'] <= 2
    stem.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Dev身份集中度：次数相近不等于持续一致', '',
        '固定NR36 IR、匹配RGB和IR自身增量关闭三个既有Dev输出，各覆盖全部4834帧。'
        '逐帧IoU0.5 Hungarian匹配仅用于事后诊断，没有新模型推理或官方评估。', '',
        '| Dev序列 | 角色 | 官方HOTA | 官方AssA | 官方IDSW | GT最大单轨质量占比 | 轨迹最大GT质量占比 | 双射集中度 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        m = r['original_official_metrics']
        lines.append(f"| {r['sequence']} | {r['role']} | {m['HOTA']:.6f} | {m['AssA']:.6f} | {m['IDSW']} | "+
            ' | '.join(f"{100*r[k]:.3f}%" for k in ['GT_largest_track_fraction', 'track_largest_GT_fraction', 'bijective_contingency_fraction'])+' |')
    lines += ['',
        '三个集中度均以本诊断中已匹配观测为分母，GT逐帧匹配覆盖率和分母在JSON保留。'
        'GT最大单轨质量占比衡量一个GT的匹配观测有多少集中在同一预测ID；'
        '轨迹最大GT质量占比衡量预测轨迹的身份混合；双射集中度是整个Dev列联表的'
        '最大一一分配质量。三者是条件性事后统计，不是官方AssA、IDF1、Recall或新的HOTA。', '',
        '主身份与片段标签使用整个Dev序列，只允许诊断，不是因果在线输入。'
        '连续25帧以上的非主身份片段及GT框重叠区间只作描述；框重叠不是真实遮挡标签，'
        '非主片段也不能直接称为已证实的错误。诊断ID变化与官方IDSW分别保留，不替换。', '',
        '该分析不支持按视频名称设置门控，也不训练任何规则。'
        '它只为后续视觉质量、时序与记忆实验保留可检查的持续身份结构。'
        'RGB预训练见过当前Dev、历史Test暴露等限制继续存在；独立贡献仍为0。', '',
        '复现：CPU运行 `tools/diagnose_dev_track_concentration.py`，拒绝覆盖已有结果。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'],
        peak_process_RSS_GiB=result['peak_process_RSS_GiB'])), flush=True)


if __name__ == '__main__':
    main()
