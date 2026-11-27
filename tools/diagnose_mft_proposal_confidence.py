#!/usr/bin/env python3
"""All-Train actual-proposal confidence/coverage counts; no score changes."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.fit_mft_candidate_risk import locations
from tools.mft_candidate_risk_data import qualify_Train,load_arrays,load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,parent=locations();base=work/'proposal_confidence_diagnosis_20261127';base.mkdir(exist_ok=True)
    q=qualify_Train(work,protocol);old=read(work/'causal_gap_completion_20261127/registered.json')
    sources={**old['source_hashes'],'tools/diagnose_mft_proposal_confidence.py':sha256(Path(__file__)),
        'configs/perceptual_proposal_quality_followup_20261127.json':sha256(ROOT/'configs/perceptual_proposal_quality_followup_20261127.json')}
    for n,h in sources.items():assert sha256(ROOT/n)==h,n
    assert not(base/'registered.json').exists();frozen_json(base/'registered.json',dict(
        status='REGISTERED_FULL_TRAIN_PROPOSAL_CONFIDENCE_COUNTS_BEFORE_LABELS',source_hashes=sources,qualified_Train=q,
        score_bands=['score_le_0_3','score_gt_0_3_le_0_6','score_gt_0_6'],Train_frames=27251,Dev_rows=0,Test_rows=0,
        scores_or_boxes_changed=False,new_models=False,new_GPU_work=False,seed=0,goal_met=False,supported_innovations=0))
    reference=read(work/'causal_gap_completion_20261127/complete_Train.json');started=time.monotonic();coverage={}
    for sequence,row in q.items():
        a=load_arrays(row['bank_path'],row['frames'],row['bank_sha256']);gt,bounds=load_GT(row['GT_path'],row['frames'],row['GT_sha256'])
        counts={band:{k:0 for k in ['proposals','known_positive','known_background','unknown']}for band in ['score_le_0_3','score_gt_0_3_le_0_6','score_gt_0_6']}
        signature=hashlib.sha256();npositive=0;validGT=0
        for i,f in enumerate(a['frame_ids']):
            frame=int(f);lo,hi=map(int,a['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2]);d=a['detections'][lo:hi];g=gt[left:right]
            y=current_detection_labels(d,g,frame=frame,split='train');validGT+=int(((g[:,6]>0)&(g[:,7]==1)).sum());npositive+=int((y>0).sum())
            signature.update(np.asarray([frame,len(y)],np.int64).tobytes()+d.tobytes()+y.tobytes())
            masks=[d[:,4]<=.3,(d[:,4]>.3)&(d[:,4]<=.6),d[:,4]>.6]
            for name,mask in zip(counts,masks):
                c=counts[name];c['proposals']+=int(mask.sum());c['known_positive']+=int((mask&(y>0)).sum())
                c['known_background']+=int((mask&(y==0)).sum());c['unknown']+=int((mask&(y<0)).sum())
        assert sum(c['proposals']for c in counts.values())==len(a['detections'])
        ordinary=reference['coverage'][sequence]['statistics']['original_output_rows']
        coverage[sequence]=dict(frames=row['frames'],proposals=len(a['detections']),valid_GT_rows=validGT,
            positive_proposal_rows=npositive,ordinary_output_rows=ordinary,
            at_least_proposal_GT_appearances_not_covered_by_ordinary_output_row_budget=max(0,npositive-ordinary),
            by_original_confidence=counts,current_Train_label_packet_sha256=signature.hexdigest(),
            bank_sha256=row['bank_sha256'],GT_sha256=row['GT_sha256'])
        guard_storage(work)
    for n,h in sources.items():assert sha256(ROOT/n)==h,n
    frozen_json(base/'complete_Train.json',dict(status='COMPLETE_ALL6_TRAIN_ACTUAL_PROPOSAL_CONFIDENCE_COUNTS',coverage=coverage,
        Train_frames=27251,proposals=sum(r['proposals']for r in coverage.values()),all_current_proposals_labeled=True,
        no_scores_boxes_or_tracker_outputs_changed=True,unknown_not_background=True,
        local_IoU_05_matching_not_HOTA_or_its_global_bound=True,new_models=False,new_GPU_work=False,new_accuracy=False,
        Dev_rows=0,Test_rows=0,seed=0,seconds=time.monotonic()-started,
        registration_sha256=sha256(base/'registered.json'),goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_ALL6_TRAIN_ACTUAL_PROPOSAL_CONFIDENCE_COUNTS',coverage=coverage)),flush=True)


if __name__=='__main__':run()
