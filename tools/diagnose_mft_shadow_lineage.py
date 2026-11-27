#!/usr/bin/env python3
"""Train/Dev-only inventory of continuous shadows spanning new trusted IDs.

This diagnostic creates no tracker outputs and no accuracy/selection result.
GT labels are joined only after the separate GT-free producer exits0.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mft_protected_peripheral_experiment import registered as parent_registered,locations,cache,qualified_input
from tools.mft_peripheral_frame_bank import SequentialFrameBank
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.fit_mft_proposal_quality import utc
from tools.mft_candidate_risk_data import load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.mft_scene_coverage import Dev_analysis_labels
from tools.fusion_memory_observation import LENGTHS


def context():
    plan,protocol,work,parent,previous=locations()
    return work,parent,work/'shadow_lineage_diagnostic_20261127'


def registered():
    work,parent,base=context();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    guard_storage(work)
    assert sum(p.stat().st_size for p in base.rglob('*')if p.is_file())<16*2**20
    return r


def produce():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    old=parent_registered();work,parent,base=context();base.mkdir(parents=True,exist_ok=True)
    source=dict(old['source_hashes']);source[str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__))
    artifacts=[parent/n for n in ['registered.json','source_Train_Dev_ready.json','run_r1.execution.json','audit_r2.execution.json']]
    frozen_json(base/'registered.json',dict(status='REGISTERED_SHADOW_LINEAGE_READONLY_DIAGNOSTIC',UTC=utc(),
        source_hashes=source,artifact_hashes={str(p):sha256(p)for p in artifacts},
        required_previous_consecutive_agreement=3,required_continuous_shadow_streak='gap+3',
        gap_buckets=[[1,5],[6,25],[26,250],[251,None]],
        no_HOTA_or_tracker_output=True,no_fitting_or_parameter_selection=True,Test_read=False,
        historical_Test_exposed=True,new_GPU_work=False,seed=0,goal_met=False,supported_innovations=0))
    manifest={};frames=0
    for split in ['train','dev']:
        for sequence,n in LENGTHS[split].items():
            meta,a=cache(split,sequence);reader=SequentialFrameBank(sequence,a);seen=set();aligned={};events=[]
            for frame in range(1,n+1):
                value=reader.next(frame);by_proposal={v['proposal_index']:v for v in value['shadow_records']}
                active=set(value['main_proposals']);updates=[]
                for base_id,proposal in value['main_proposals'].items():
                    record=by_proposal.get(proposal)
                    if record is None:continue
                    sid=record['shadow_id'];prior=aligned.get(sid)
                    if base_id not in seen and prior is not None and prior['base']!=base_id and prior['base']not in active:
                        gap=frame-prior['frame']
                        if prior['count']>=3 and record['streak']>=gap+3:
                            events.append(dict(frame=frame,new_base=base_id,new_proposal=proposal,shadow_id=sid,
                                old_base=prior['base'],old_frame=prior['frame'],old_proposal=prior['proposal'],
                                prior_agreements=prior['count'],continuous_shadow_streak=record['streak'],gap=gap))
                    count=prior['count']+1 if prior is not None and prior['base']==base_id and prior['frame']==frame-1 else 1
                    updates.append((sid,dict(base=base_id,proposal=proposal,frame=frame,count=count)))
                seen.update(active)
                for sid,v in updates:aligned[sid]=v
            path=base/'private_events'/split/(sequence+'.json')
            frozen_json(path,dict(split=split,sequence=sequence,frames=n,source_bank_sha256=meta['sha256'],events=events,
                GT_labels_read=False,current_only_snapshot_delivery=True,past_outputs_rewritten=False))
            manifest[split+'_'+sequence]=dict(path=str(path),sha256=sha256(path),frames=n,events=len(events))
            frames+=n;registered();print(json.dumps(dict(split=split,sequence=sequence,frames=n,GT_free_events=len(events))),flush=True)
    assert frames==32085
    frozen_json(base/'source_ready.json',dict(status='COMPLETE_GT_FREE_LINEAGE_EVENTS_ALL8',UTC=utc(),frames=frames,rows=manifest,
        GT_labels_read=False,Test_read=False,new_tracker_outputs=False))


def counters():
    return dict(opportunities=0,same_known_GT=0,different_known_GT=0,known_background_endpoint=0,unknown_endpoint=0)


def analyse():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();work,parent,base=context();e=read(base/'produce_r1.execution.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    ready=read(base/'source_ready.json');assert ready['frames']==32085
    parent_reg=parent_registered();report={};totals={}
    for name,m in ready['rows'].items():
        assert sha256(m['path'])==m['sha256'];v=read(m['path']);split,sequence=v['split'],v['sequence']
        q,a=qualified_input(split,sequence,parent_reg,all_features=False)
        gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256']);label_cache={}
        def labels(frame):
            if frame not in label_cache:
                lo,hi=map(int,a['offsets'][frame-1:frame+1]);g=gt[bounds[frame-1]:bounds[frame]]
                if split=='train':out=current_detection_labels(a['detections'][lo:hi],g,frame=frame,split='train')
                else:out=Dev_analysis_labels(a['detections'][lo:hi],g,frame=frame,split='dev',purpose='read_only_coverage_diagnostic')
                label_cache[frame]=out
            return label_cache[frame]
        rows={}
        for event in v['events']:
            f,old=int(event['frame']),int(event['old_frame']);gap=event['gap']
            assert old<f and event['prior_agreements']>=3 and event['continuous_shadow_streak']>=gap+3
            phase='early75'if f<=3*q['frames']//4 else'last25'
            if split=='dev':phase='full'
            bucket='gap1to5'if gap<=5 else'gap6to25'if gap<=25 else'gap26to250'if gap<=250 else'gap251plus'
            key=phase+'_'+bucket;counts=rows.setdefault(key,counters());total=totals.setdefault(split+'_'+key,counters())
            a_id=int(labels(old)[event['old_proposal']]);b_id=int(labels(f)[event['new_proposal']])
            outcome='unknown_endpoint'if min(a_id,b_id)<0 else'known_background_endpoint'if min(a_id,b_id)==0 else'same_known_GT'if a_id==b_id else'different_known_GT'
            for c in [counts,total]:c['opportunities']+=1;c[outcome]+=1
        report[name]=dict(frames=q['frames'],events=m['events'],GT_sha256=q['GT_sha256'],event_source_sha256=m['sha256'],counts=rows)
        print(json.dumps(dict(sequence=sequence,split=split,known_label_endpoint_diagnostic_complete=True)),flush=True)
    registered()
    frozen_json(base/'complete_readonly_diagnostic.json',dict(status='COMPLETE_SHADOW_CONTINUOUS_LINEAGE_TRAIN_DEV_DIAGNOSTIC',UTC=utc(),
        rows=report,totals=totals,source_producer_actual_exit0=True,source_execution_sha256=sha256(base/'produce_r1.execution.json'),
        unknown_not_background=True,no_HOTA_or_new_tracker_outputs=True,no_fitting_or_threshold_selection=True,
        scope='IoU/Hungarian endpoint identity agreement for GT-free candidate links, not an oracle trajectory or predicted HOTA gain.',
        Test_read=False,new_GPU_work=False,budget=guard_storage(work),goal_met=False,supported_innovations=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['produce','analyse']);a=p.parse_args()
    {'produce':produce,'analyse':analyse}[a.mode]()
