#!/usr/bin/env python3
"""Describe hard-cycle suppression on fixed Train frames; no fitting or Test."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.models.structural_relative_repair import structural_weights


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    base_path = ROOT/'configs/perceptual_structural_relative_20261123.json'
    base = json.loads(base_path.read_text())
    a = base['architecture']
    work = Path(plan['work_root'])
    out = work/'cycle_support_train_audit'
    out.mkdir(exist_ok=False)
    state = dict(status='REGISTERED_BEFORE_FIELDS_AND_TRAIN_LABEL_SCORING', indices=[0, 48, 96, 144],
        source_sha256=sha256(__file__), operator_sha256=sha256(ROOT/'yolox/models/structural_relative_repair.py'),
        base_config_sha256=sha256(base_path), views=['RGB_self', 'actual_IR'],
        levels=dict(P3=[60,108], P4=[30,54], P5=[15,27]),
        scope='Descriptive24Trainframe support audit after complete59Dev. No fitting, threshold selection or '
              'new Test predictions/accuracy. A GT box is not a true fish-pixel or correspondence label.',
        Test_read=False, goal_met=False, supported_innovations=0)
    dump(out/'registration.json', state)
    packets = {}
    with torch.inference_mode():
        for sequence in protocol['train']:
            folder = work/'structural_cache'/sequence
            meta_path = folder/'complete.json'
            meta = json.loads(meta_path.read_text())
            assert not meta['GT_read'] and meta['specification_sha256'] == sha256(base_path)
            arrays = {}
            for name, row in meta['outputs'].items():
                assert sha256(row['path']) == row['sha256']
                arrays[name] = torch.from_numpy(np.stack(np.load(row['path'], mmap_mode='r')[state['indices']])).float()
            fields = {}
            for level, shape in state['levels'].items():
                q = F.adaptive_avg_pool2d(arrays['rgb_descriptor'], shape)
                qv = F.adaptive_avg_pool2d(arrays['rgb_valid'], shape) >= .5
                for view, prefix in [('RGB_self', 'rgb'), ('actual_IR', 'ir')]:
                    k = F.adaptive_avg_pool2d(arrays[prefix+'_descriptor'], shape)
                    kv = F.adaptive_avg_pool2d(arrays[prefix+'_valid'], shape) >= .5
                    for cycle in [False, True]:
                        _, confidence = structural_weights(q, k, qv, kv, radius=a['local_radius'],
                            temperature=a['structural_temperature'], cycle=cycle,
                            tolerance=a['forward_backward_tolerance_cells'])
                        fields[f'{level}_{view}_{"cycle" if cycle else "no_cycle"}'] = confidence[:,0].numpy()
            path = out/(sequence+'.npz')
            np.savez_compressed(path, **fields)
            packets[sequence] = dict(path=str(path), sha256=sha256(path),
                frames=[meta['frame_ids'][i] for i in state['indices']],
                cache_manifest_sha256=sha256(meta_path), bytes=path.stat().st_size)
    dump(out/'predictions_frozen.json', dict(status='ALL24_TRAIN_FIELDS_FROZEN_BEFORE_LABEL_SCORING',
        packets=packets, GT_read=False, source=state))
    rows = []
    input_h, input_w = base['training_input_hw']
    ratio = min(input_h/1080, input_w/1920)
    for sequence, packet in packets.items():
        gt_path = Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        assert sha256(gt_path) == protocol['derived_gt_sha256']['train/'+sequence]
        gt = np.loadtxt(gt_path, delimiter=',', ndmin=2)
        gt = gt[(gt[:,6] > 0) & (gt[:,7] == 1)]
        assert sha256(packet['path']) == packet['sha256']
        with np.load(packet['path']) as z:
            for level, (h,w) in state['levels'].items():
                yy, xx = np.mgrid[:h,:w]
                xx, yy = (xx+.5)*input_w/w/ratio, (yy+.5)*input_h/h/ratio
                foreground = []
                for frame in packet['frames']:
                    mask = np.zeros((h,w), bool)
                    for x,y,bw,bh in gt[gt[:,0] == frame, 2:6]:
                        mask |= (xx >= x) & (xx < x+bw) & (yy >= y) & (yy < y+bh)
                    foreground.append(mask)
                foreground = np.stack(foreground)
                for view in state['views']:
                    full = z[f'{level}_{view}_cycle']
                    no_cycle = z[f'{level}_{view}_no_cycle']
                    assert (full <= no_cycle+1e-7).all()
                    for name, region in [('GT_bbox', foreground), ('outside_GT_bbox', ~foreground)]:
                        count = int(region.sum())
                        supported = region & (no_cycle > 0)
                        valid_count = int(supported.sum())
                        total = float(no_cycle[region].sum(dtype=np.float64))
                        kept = float(full[region].sum(dtype=np.float64))
                        rows.append(dict(sequence=sequence, level=level, view=view, support=name,
                            grid_cells=count, no_cycle_supported_cells=valid_count,
                            no_cycle_mean_confidence=total/max(count,1), cycle_mean_confidence=kept/max(count,1),
                            cycle_rejected_fraction_of_supported=float((supported & (full == 0)).sum())/max(valid_count,1),
                            cycle_removed_confidence_fraction=(total-kept)/total if total else None))
    report = dict(status='COMPLETE_TRAIN_ONLY_CYCLE_SUPPORT_AUDIT', rows=rows,
        registration=state, packets=packets,
        predictions_frozen_sha256=sha256(out/'predictions_frozen.json'),
        interpretation='This measures how much of the actual structural confidence the hard inverse check suppresses. '
            'Higher confidence or bbox support is not correctness, recall, alignment truth or HOTA. '
            'No claim that rejected matches are true correspondences. Cycle harm itself is established only '
            'by the completed Dev removals, not by this descriptive grid audit.',
        Test_read=False, goal_met=False, supported_innovations=0)
    stem = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_CYCLE_SUPPORT_TRAIN_AUDIT'
    dump(stem.with_suffix('.json'), report)
    lines = ['# Train-only support suppressed by hard inverse consistency', '', state['scope'], '',
        report['interpretation'], '', '| Video | Level | View | Region | Supported cells | Rejected fraction | Removed confidence |',
        '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for row in rows:
        v = row['cycle_removed_confidence_fraction']
        lines.append('| '+' | '.join(str(row[k]) for k in ['sequence','level','view','support','no_cycle_supported_cells'])+
            f" | {row['cycle_rejected_fraction_of_supported']:.4f} | "+('—' if v is None else f'{v:.4f}')+' |')
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'], frames=24, videos=6, rows=len(rows), CUDA_used=False)))


if __name__ == '__main__':
    main()
