#!/usr/bin/env python3
"""Check actual numerical auxiliary influence, without annotations or metrics."""
import json
from pathlib import Path

import numpy as np

from tools.run_pseudo_ir_comparison import dump,sha256

ROOT=Path(__file__).resolve().parents[1]


def packet(path):
    with np.load(path)as z:
        return {key:z[key]for key in ['frame_ids','offsets','boxes','scores']}


def exact(left,right):
    return all(np.array_equal(left[k],right[k])for k in left)


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    spec_path=ROOT/'configs/perceptual_structural_relative_eval_20261123.json'
    spec=json.loads(spec_path.read_text());state=json.loads((work/'structural_relative_dev_state.json').read_text())
    assert len(state['both32_prefix_proofs'])==2
    rows=[];image_roll=[]
    for sequence in ['BT-005','SN-013']:
        base=work/'structural_relative_predictions/dev'/sequence/'prefix_check_32'
        meta=json.loads((base/'complete.json').read_text())
        assert meta['frames']==32 and meta['specification_sha256']==sha256(spec_path)
        reference=packet(base/'CG3_source_RGB.npz')
        for model in spec['models']:
            if model.get('baseline'):
                continue
            key=model['id']+'_raw';path=base/(key+'.npz');current=packet(path)
            with np.load(path)as z:
                rgb_reference=z['rgb_reference_scores']
            difference=current['scores']-rgb_reference
            assert len(difference)==len(current['boxes']) and np.isfinite(difference).all()
            strong=rgb_reference>=.6;weak=~strong
            rows.append(dict(sequence=sequence,id=key,cue=model['cue'],variant=model['variant'],
                epoch=model['epoch'],intervention=model.get('intervention','none'),
                frames=32,selected_detections=len(difference),
                selected_nonzero_score_deltas=int(np.count_nonzero(difference)),
                mean_absolute_score_delta=float(np.abs(difference).mean())if len(difference)else 0.,
                maximum_absolute_score_delta=float(np.max(np.abs(difference),initial=0)),
                mean_absolute_delta_at_strong_RGB=float(np.abs(difference[strong]).mean())if strong.any()else 0.,
                mean_absolute_delta_at_weak_RGB=float(np.abs(difference[weak]).mean())if weak.any()else 0.,
                selected_weak_RGB_crossing_above_point6=int(np.count_nonzero(weak&(current['scores']>=.6))),
                same_complete_packet_as_fresh_RGB=exact(current,reference),source_sha256=sha256(path)))
        for epoch in [6,12,18,24]:
            correct=base/f'SR_ir_full_s0_e{epoch:02d}_raw.npz'
            roll=base/f'SR_IR_image_roll_s0_e{epoch:02d}_raw.npz'
            image_roll.append(dict(sequence=sequence,epoch=epoch,
                correct_and_rolled_packets_exact=exact(packet(correct),packet(roll)),
                correct_sha256=sha256(correct),roll_sha256=sha256(roll)))
    assert len(rows)==58 and len(image_roll)==8
    report=dict(status='COMPLETE_UNLABELED32_PREFIX_RESPONSE_DIAGNOSIS',rows=rows,
        actual_IR_vs_image_roll=image_roll,source_specification_sha256=sha256(spec_path),
        source_sha256=sha256(Path(__file__)),GT_read=False,accuracy_read=False,
        scope='32prefixframes/video. Same original dense-anchor RGB reference score at each '
              'selected proposal, not geometry matching. Scores may change candidate membership '
              'and NMS. Nonzero response proves numerical influence, not correct fish recovery, '
              'identity information, calibration quality, HOTA improvement or novelty. '
              'A feature risk gate does not guarantee unchanged strong output logits because '
              'the frozen head reads neighboring features. No parameter or model selection here.')
    target=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_UNLABELED_RESPONSE_AUDIT.json'
    assert not target.exists();dump(target,report)
    for row in rows:
        if row['epoch']==24 and row['variant']=='full'and row['intervention']=='none':
            print(json.dumps({k:row[k]for k in ['sequence','cue','selected_detections','selected_nonzero_score_deltas',
                'mean_absolute_score_delta','mean_absolute_delta_at_strong_RGB','mean_absolute_delta_at_weak_RGB',
                'selected_weak_RGB_crossing_above_point6','same_complete_packet_as_fresh_RGB']}))
    print(json.dumps(dict(IR_roll_changes_packets=sum(not r['correct_and_rolled_packets_exact']for r in image_roll),
        comparisons=8,GT_read=False,accuracy_read=False)))


if __name__=='__main__':
    main()
