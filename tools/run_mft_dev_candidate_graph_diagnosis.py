#!/usr/bin/env python3
"""Describe frozen risk-policy masks at every native-IR Dev identity event."""
import argparse
from collections import Counter
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from yolox.tracker import matching
from yolox.tracker.basetrack import TrackState
from tools.fit_mft_candidate_risk import locations,threshold_values
from tools.run_mft_dev_identity_timeline import registered as timeline_registered
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.audit_mft_tracklet_r2 import bank_audit
from tools.run_mft_tracklet_permanence_r2 import registered as peer_registered
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_candidate_risk_features import candidate_features,FEATURE_NAMES
from tools.native_pyramid_stream_tracks import mot_line
from tools.fusion_memory_observation import LENGTHS


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def graph_snapshot(tracks,detections,base,frame,speed):
    x,eligible=candidate_features(tracks,detections,base,frame,speed)
    pairs=matching.linear_assignment(base,thresh=.9)[0];protected=[]
    for i,j in pairs:
        row=np.delete(base[i],j);column=np.delete(base[:,j],i)
        second=min(float(row.min())if row.size else np.inf,float(column.min())if column.size else np.inf)
        if tracks[i].state==TrackState.Tracked and tracks[i].frame_id==frame-1 and base[i,j]<=.4 and second-base[i,j]>=.1:
            protected.append((int(i),int(j)))
    return dict(frame=frame,track_ids=[int(t.track_id)for t in tracks],
        states=[int(t.state)for t in tracks],gaps=[frame-t.frame_id for t in tracks],
        memory_ready=[t.accepted>=3 and t.rgb is not None and t.last_box is not None for t in tracks],
        accepted=[int(t.accepted)for t in tracks],
        proposal_indices=[int(d.observation['proposal_index'])for d in detections],
        base=base.copy(),features=x,eligible=eligible,
        protected_rows={i for i,j in protected},protected_columns={j for i,j in protected})


def classify_event(event,assignments,scores,graph):
    previous=int(event['previous_prediction_ID']);current=int(event['current_prediction_ID'])
    assert previous!=current and current in assignments
    proposal=assignments[current];score=float(scores[proposal])
    result=dict(source_event=event,current_output_proposal_index=proposal,current_proposal_score=score,
        previous_ID_in_primary_pool=previous in graph['track_ids'],
        current_proposal_in_primary_columns=proposal in graph['proposal_indices'],
        learned_score_or_model_used=False,passes_nonlearned_policy_masks=False)
    if not result['previous_ID_in_primary_pool']:
        result['reason']='previous_ID_not_in_primary_pool';return result
    i=graph['track_ids'].index(previous)
    result.update(previous_track_state=graph['states'][i],previous_track_gap=graph['gaps'][i],
        accepted_template_updates=graph['accepted'][i],memory_ready=graph['memory_ready'][i],
        protected_previous_track_row=i in graph['protected_rows'])
    if not result['current_proposal_in_primary_columns']:
        assert score<=.6
        result['reason']='current_output_uses_nonprimary_proposal';return result
    j=graph['proposal_indices'].index(proposal)
    result.update(original_geometric_cost=float(graph['base'][i,j]),
        feature_eligible=bool(graph['eligible'][i,j]),protected_current_detection_column=j in graph['protected_columns'])
    if result['feature_eligible']:
        result['current_past_features']={n:float(graph['features'][i,j,FEATURE_NAMES.index(n)])for n in
            ['IoU','center_error_body','RGB_cosine','IR_cosine','RGB_row_margin','RGB_column_margin',
             'IR_row_margin','IR_column_margin','past_template_quality','current_quality']}
    if not result['memory_ready']:reason='past_template_history_unavailable'
    elif not result['feature_eligible']:reason='outside_feature_reach_or_scale_support'
    elif result['protected_previous_track_row']or result['protected_current_detection_column']:reason='hard_geometric_protection_masks_edge'
    elif result['original_geometric_cost']<=.9:
        reason='ordinary_edge_passes_nonlearned_masks';result['passes_nonlearned_policy_masks']=True
    elif result['previous_track_state']==TrackState.Lost and 2<=result['previous_track_gap']<=25:
        reason='lost_edge_still_requires_learned_threshold_mutual_margin';result['passes_nonlearned_policy_masks']=True
    elif result['previous_track_state']==TrackState.Tracked and result['previous_track_gap']==1:
        reason='tracked_gap1_edge_excluded_by_lost_only_rescue'
    else:reason='other_state_or_gap_excluded_by_lost_only_rescue'
    result['reason']=reason
    return result


def paths():
    plan,protocol,work,risk=locations()
    return plan,protocol,work,work/'dev_candidate_graph_diagnosis_20261127',work/'dev_identity_timeline_20261127'


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    parent=timeline_registered();plan,protocol,work,base,timeline=paths()
    execution=read(timeline/'CPU_diagnostic_r1.execution.json')
    assert execution['actual_child_exit_code']==0 and execution['source_unchanged']and identity(execution['identity']['PID'])is None
    proof=read(base/'CPU_regressions_r1.json');assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==4
    sources=dict(parent['source_hashes'])
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n;sources[n]=h
    report=read(timeline/'complete.json');assert report['events']==291 and report['all19_alpha_numeric_fields_checked']==2772
    event_path=Path(report['events_CSV']['path']);assert sha256(event_path)==report['events_CSV']['sha256']
    with event_path.open()as stream:events=[r for r in csv.DictReader(stream)if r['case']=='IT_PC_native_zero']
    assert len(events)==73
    case=next(c for c in parent['fixed_cases']if c['id']=='IT_PC_native_zero')
    peer=work/'tracklet_permanence_20261127_r2';peer_reg=peer_registered()
    banks={s:bank_audit(peer,'dev',s,peer_reg)for s in LENGTHS['dev']}
    artifacts={str(p):sha256(p)for p in [base/'CPU_regressions_r1.json',base/'CPU_regressions_r1.stdout.txt',
        timeline/'complete.json',timeline/'CPU_diagnostic_r1.execution.json',event_path]}
    for s in LENGTHS['dev']:
        meta=peer/'cache/dev'/(s+'.json');p=Path(read(meta)['path']);artifacts[str(meta)]=sha256(meta);artifacts[str(p)]=sha256(p)
    value=dict(status='REGISTERED_ALL73_NATIVE_DEV_EVENT_GRAPH_DIAGNOSIS',UTC=utc(),
        source_hashes=sources,artifact_hashes=artifacts,events=events,case=case,Dev_banks=banks,
        frames=4834,sequences=LENGTHS['dev'],new_model=False,new_weights=False,new_Test=False,
        altered_tracker_policy=False,graph_masks='Exact frozen risk eligibility/protection/lost-only gates; learned score gates remain unknown',
        only_original_native_trajectory_path=True,no_global_recoverability_bound=True,
        historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0)
    assert not(base/'registered.json').exists();frozen_json(base/'registered.json',value)
    print(json.dumps(dict(status=value['status'],events=73,sources=len(sources))),flush=True)


def registered():
    timeline_registered();plan,protocol,work,base,timeline=paths();reg=read(base/'registered.json')
    for n,h in reg['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in reg['artifact_hashes'].items():assert sha256(n)==h,n
    return reg


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,timeline=paths();started=time.monotonic()
    assert not(base/'complete.json').exists()
    rows=[];coverage={};speed=threshold_values(work)['Train_unknown_speed_upper']
    peer=work/'tracklet_permanence_20261127_r2'
    for sequence,frames in LENGTHS['dev'].items():
        meta=read(peer/'cache/dev'/(sequence+'.json'))
        with np.load(meta['path'],allow_pickle=False)as data:arrays={k:data[k].copy()for k in data.files}
        events={}
        for e in reg['events']:
            if e['sequence']==sequence:events.setdefault(int(e['current_match_frame']),[]).append(e)
        snapshot=[None]
        def observe(tracks,detections,base_cost,frame,stage):
            if stage=='primary'and frame in events:
                snapshot[0]=graph_snapshot(tracks,detections,base_cost,frame,speed)
            return base_cost
        teacher=CandidateTeacher(speed);teacher.reset(sequence);teacher.core.association_hook=observe
        digest=hashlib.sha256();count=0;proposals=0
        for i,f in enumerate(arrays['frame_ids']):
            frame=int(f);lo,hi=map(int,arrays['offsets'][i:i+2]);d=arrays['detections'][lo:hi]
            output=teacher.update(sequence,frame,d,arrays['rgb'][lo:hi],arrays['ir'][lo:hi])
            for r in output:digest.update(mot_line(r).encode())
            count+=len(output);proposals+=len(d)
            if frame in events:
                graph=snapshot[0];assert graph is not None and graph['frame']==frame
                assignments=teacher.current_proposal_assignments(frame)
                scores={j:float(row[4])for j,row in enumerate(d)}
                for e in events[frame]:rows.append(classify_event(e,assignments,scores,graph))
        assert digest.hexdigest()==reg['case']['MOT_sha256'][sequence]
        assert frames==teacher.statistics['frames']and proposals==reg['Dev_banks'][sequence]['proposal_count']
        coverage[sequence]=dict(frames=frames,proposals=proposals,MOT_rows=count,MOT_sha256=digest.hexdigest(),
            original_MOT_exact=True,events=sum(len(v)for v in events.values()),snapshots=len(events),GT_supplied_to_teacher=False)
        print(json.dumps(dict(sequence=sequence,frames=frames,MOT_exact=True,events=coverage[sequence]['events'])),flush=True)
        guard_storage(work)
    assert len(rows)==73
    counts=dict(Counter(r['reason']for r in rows));by_sequence={s:dict(Counter(r['reason']for r in rows if r['source_event']['sequence']==s))for s in LENGTHS['dev']}
    registered()
    frozen_json(base/'complete.json',dict(status='COMPLETE_ALL73_NATIVE_DEV_EVENT_GRAPH_DIAGNOSIS',UTC=utc(),
        frames=4834,events=73,rows=rows,reason_counts=counts,by_sequence=by_sequence,coverage=coverage,
        passes_nonlearned_masks=sum(r['passes_nonlearned_policy_masks']for r in rows),
        candidate_replay_seconds=time.monotonic()-started,registration_sha256=sha256(base/'registered.json'),
        actual_tracker_costs_changed=False,original_MOT_exact_for_all4834_frames=True,
        learned_models_read=False,new_accuracy=False,new_GPU_work=False,Test_GT_read=False,
        conditional_on_original_teacher_trajectory=True,not_a_global_recoverability_bound=True,
        limitations=['Official HOTA event labels are offline whole-sequence diagnostics, not oracle runtime identities.',
            'An edge passing nonlearned masks still needs learned score/threshold, mutual uniqueness and a globally consistent assignment. No recovery or HOTA improvement is predicted.',
            'A learned tracker may change earlier decisions and therefore later matching pools; these counts describe only the unchanged original trajectory.',
            'No Dev label or event time enters a fitted model, tracker cost, routing rule or actual matching decision here.'],
        goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_ALL73_NATIVE_DEV_EVENT_GRAPH_DIAGNOSIS',reason_counts=counts,
        passes_nonlearned_masks=sum(r['passes_nonlearned_policy_masks']for r in rows),seconds=time.monotonic()-started)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run']);args=p.parse_args()
    register()if args.mode=='register'else run()


if __name__=='__main__':main()
