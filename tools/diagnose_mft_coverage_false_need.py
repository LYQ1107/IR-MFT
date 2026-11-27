#!/usr/bin/env python3
"""Read-only Train failure probe; no candidate fitting or Test inspection."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.mft_birth_only import BirthOnlyTeacher
from tools.mft_proposal_quality import ProposalConfirmation
from tools.mft_candidate_risk_data import load_arrays,load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.fit_mft_candidate_risk import threshold_values
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.fit_mft_proposal_quality import utc
from yolox.tracker import matching

W=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
B=W/'coverage_recall_20261127'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=read(B/'registered.json');v=read(B/'complete_forward_Train_validation.json');state=read(B/'state.json')
    assert all(j['status']=='COMPLETE'and j['exit_code']==0 for phase,jobs in state['phases'].items()if phase.startswith('COVERAGE_RECALL_TRAINVAL')for j in jobs.values())
    reference=next(row for row in v['rows']if row['case']['id']=='CR_reference')
    paired=next(row for row in v['rows']if row['case']['id']=='CR_paired_t025')
    worst=min((b['HOTA']-a['HOTA'],a['seq'])for a,b in zip(reference['per_sequence47'],paired['per_sequence47'])if a['seq']!='COMBINED')
    sequence=worst[1];assert sequence in r['qualified']['train'];q=r['qualified']['train'][sequence]
    diagnostic=read(W/'scene_coverage_diagnostic_20261127/complete_Train_Dev.json');table=diagnostic['private_scalar_table']
    assert sha256(table['path'])==table['sha256']
    with np.load(table['path'],allow_pickle=False)as z:x=z['train_'+sequence+'_features']
    selected=(np.flatnonzero(x[:,table['feature_columns'].index('paired')]>=.25)+1)[:3]
    assert len(selected)==3;frames=set(map(int,selected));a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256'])
    teacher=BirthOnlyTeacher(threshold_values(W)['Train_unknown_speed_upper']);shadow=ProposalConfirmation();saved={}
    for frame in range(1,int(selected[-1])+1):
        lo,hi=map(int,a['offsets'][frame-1:frame+1]);d=a['detections'][lo:hi];rgb=a['rgb'][lo:hi];ir=a['ir'][lo:hi]
        streak=shadow.update(sequence,frame,d);teacher.update(sequence,frame,d,rgb,ir)
        if frame not in frames:continue
        assigned=np.asarray(list(teacher.current_proposal_assignments(frame).values()),np.int64)
        candidates=np.asarray([e['proposal_index']for e in teacher.core.last_birth_events if not e['original_allowed']and streak[e['proposal_index']]>=3],np.int64)
        measured=d[assigned,:4];out=[]
        for j in candidates:
            box=d[j,:4];area=float(np.prod(box[2:]-box[:2]));intersection=np.maximum(0,np.minimum(box[None,2:],measured[:,2:])-np.maximum(box[None,:2],measured[:,:2])).prod(axis=1)
            out.append(dict(proposal_index=int(j),box_xyxy=box.tolist(),score=float(d[j,4]),streak=int(streak[j]),
                max_IoU_with_current_measured=float(matching.ious(box[None],measured).max())if len(measured)else 0.,
                max_fraction_candidate_area_covered=float(intersection.max()/area)if len(intersection)else 0.))
        saved[frame]=dict(proposals=d.copy(),assigned=assigned.copy(),measured=measured.copy(),candidates=out)
    # All selected GT-free geometry has finished. Parse current Train labels now.
    gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256']);rows=[]
    for frame,z in saved.items():
        lo,hi=map(int,bounds[frame-1:frame+1]);current=gt[lo:hi]
        labels=current_detection_labels(z['proposals'],current,frame=frame,split='train')
        for candidate in z['candidates']:candidate['Train_label']=int(labels[candidate['proposal_index']])
        rows.append(dict(frame=frame,current_original_measured_boxes=z['measured'].tolist(),confirmed_unused_high_candidates=z['candidates'],
            current_GT_rows=current.tolist(),paired_deficit=float(x[frame-1,table['feature_columns'].index('paired')])))
    result=dict(status='COMPLETE_READONLY_TRAIN_WORST_COVERAGE_FAILURE_GEOMETRY',UTC=utc(),
        selection='Worst standalone-validation per-sequence HOTA delta for paired0.25, then first3 original full-Train gated frames; never a Test sequence or inference rule',
        sequence=sequence,validation_HOTA_delta=worst[0],rows=rows,bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'],
        validation_report_sha256=sha256(B/'complete_forward_Train_validation.json'),
        producer_source_sha256=sha256(Path(__file__)),GT_labels_only_after_selected_geometry=True,
        unknown_labels_unchanged=True,Test_read=False,no_tracking_modified=True,new_accuracy=False,
        interpretation='Geometric overlap can test duplicate or merged-object explanations; it does not by itself prove the cause of identity degradation.',goal_met=False)
    frozen_json(B/'failure_analysis_Train_false_need.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
