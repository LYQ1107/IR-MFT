#!/usr/bin/env python3
"""GT-free action collection, then separately closed Train-only labels."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.mft_assignment_actions import ActionTeacher,FEATURE_NAMES,STAGES,KINDS
from tools.mft_protected_peripheral_experiment import locations,registered as parent_registered,qualified_input
from tools.mft_proposal_quality_evaluation import ShapeTeacher
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.diagnose_mft_birth_only import state_equal
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_stream_tracks import mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_candidate_train_bank import atomic_json
from tools.fit_mft_proposal_quality import utc
from tools.fusion_memory_observation import LENGTHS
from tools.task_utility_queue import Queue

SOURCES=['tools/mft_assignment_actions.py','tools/run_mft_assignment_action_census.py',
    'tools/analyse_mft_assignment_actions.py','tests/test_mft_assignment_actions.py',
    'configs/perceptual_assignment_action_census_20261127.json']


def context():
    plan,protocol,work,parent,previous=locations()
    return plan,protocol,work,parent,work/'assignment_action_census_20261127'


def register():
    old=parent_registered();plan,protocol,work,parent,base=context();base.mkdir(parents=True,exist_ok=True)
    spec=read(ROOT/SOURCES[-1]);source=dict(old['source_hashes']);source.update({n:sha256(ROOT/n)for n in SOURCES})
    paths=[parent/'registered.json',parent/'run_r1.execution.json',parent/'audit_r2.execution.json',parent/'source_Train_Dev_ready.json',
        work/'anchor_exchange_20261127/complete_Train.json',work/'deformation_association_20261127/complete_Train_Dev.json',
        work/'deformation_association_20261127/run_r1.execution.json']
    for p in [parent/'run_r1.execution.json',parent/'audit_r2.execution.json',work/'deformation_association_20261127/run_r1.execution.json']:
        e=read(p);assert e['actual_child_exit_code']==0 and e['source_unchanged']
    proof=read(base/'unit_regressions.json');assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==4
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h
    speed_path=work/'tracklet_permanence_20261127_r2/Train_thresholds.json';paths.extend([speed_path,base/'unit_regressions.json'])
    budget=guard_storage(work);assert budget['work_file_bytes']+spec['new_study_disk_cap_bytes']<budget['work_cap_bytes']
    frozen_json(base/'registered.json',dict(status='REGISTERED_ACTION_CENSUS_BEFORE_NEW_COLLECTION_OR_LABELS',UTC=utc(),
        source_hashes=source,artifact_hashes={str(p):sha256(p)for p in paths},specification=spec,qualified_Train=old['qualified']['train'],
        speed_upper=read(speed_path)['thresholds']['Train_unknown_speed_upper'],feature_names=list(FEATURE_NAMES),
        stage_names=list(STAGES),kind_names=list(KINDS),Train_frames=27251,Test_or_Dev_rows=0,new_fits=0,
        no_new_NN_images_descriptors_or_weights=True,goal_met=False,supported_innovations=0))


def registered():
    plan,protocol,work,parent,base=context();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    guard_storage(work)
    assert sum(p.stat().st_size for p in base.rglob('*')if p.is_file())<=r['specification']['new_study_disk_cap_bytes']
    return r


class Recorder:
    def __init__(self):
        self.frames=[];self.stages=[];self.kinds=[];self.counts=[];self.before=[];self.after=[];self.regrets=[];self.features=[]
        self.matches=[];self.assignments=[];self.assign_offsets=[0]

    def observe(self,frame,stage,matches,actions):
        code=STAGES.index(stage)
        for track,proposal,available in matches:self.matches.append((frame,code,track,proposal,available))
        for a in actions:
            assert len(a['before'])==len(a['after'])in[1,2]
            pad=lambda edges:list(edges)+[(-1,-1)]*(2-len(edges))
            self.frames.append(frame);self.stages.append(code);self.kinds.append(KINDS.index(a['kind']));self.counts.append(len(a['before']))
            self.before.append(pad(a['before']));self.after.append(pad(a['after']));self.regrets.append(a['regret']);self.features.append(a['features'].copy())

    def commit_assignments(self,frame,assignments):
        assert frame==len(self.assign_offsets)
        self.assignments.extend(sorted(assignments.items()));self.assign_offsets.append(len(self.assignments))

    def arrays(self,frames):
        a=dict(action_frame=np.asarray(self.frames,np.int64),action_stage=np.asarray(self.stages,np.uint8),
            action_kind=np.asarray(self.kinds,np.uint8),action_count=np.asarray(self.counts,np.uint8),
            before=np.asarray(self.before,np.int64).reshape(-1,2,2),after=np.asarray(self.after,np.int64).reshape(-1,2,2),
            regret=np.asarray(self.regrets,np.float64),features=np.asarray(self.features,np.float32).reshape(-1,len(FEATURE_NAMES)),
            matches=np.asarray(self.matches,np.int64).reshape(-1,5),assignments=np.asarray(self.assignments,np.int64).reshape(-1,2),
            assign_offsets=np.asarray(self.assign_offsets,np.int64),frame_ids=np.arange(1,frames+1,dtype=np.int64))
        assert a['assign_offsets'].shape==(frames+1,)and a['assign_offsets'][-1]==len(a['assignments'])
        assert np.isfinite(a['features']).all()and np.isfinite(a['regret']).all()
        assert not len(a['regret'])or (a['regret']<=.1+1e-12).all()
        return a


def engineering():
    r=registered();plan,protocol,work,parent,base=context();rows=[]
    for sequence in LENGTHS['train']:
        q,a=qualified_input('train',sequence,parent_registered(),all_features=True);n=min(64,q['frames'])
        def replay(future):
            record=Recorder();teacher=ActionTeacher(r['speed_upper'],record.observe);plain=CandidateTeacher(r['speed_upper']);prefix=hashlib.sha256()
            for frame in range(1,n+1):
                lo,hi=map(int,a['offsets'][frame-1:frame+1]);d,rgb,ir=[a[k][lo:hi].copy()for k in ['detections','rgb','ir']]
                if future and frame>32:d[:,:4]+=777;rgb*=-1;ir*=-1
                before=[v.copy()for v in[d,rgb,ir]]
                old=plain.update(sequence,frame,d,rgb,ir);new=teacher.update(sequence,frame,d,rgb,ir)
                assert ''.join(map(mot_line,old))==''.join(map(mot_line,new));state_equal(plain,teacher)
                record.commit_assignments(frame,teacher.current_proposal_assignments(frame))
                if frame<=32:prefix.update(''.join(map(mot_line,new)).encode())
                for b,c in zip(before,[d,rgb,ir]):np.testing.assert_array_equal(b,c)
            values=record.arrays(n);selected=values['action_frame']<=32
            prefix.update(values['features'][selected].tobytes());prefix.update(values['before'][selected].tobytes());prefix.update(values['after'][selected].tobytes())
            return prefix.hexdigest()
        assert replay(False)==replay(True)
        rows.append(dict(sequence=sequence,frames=n,original_MOT_and_hidden_state_exact=True,causal_prefix_frames=32,feature_prefix_exact=True))
    registered();frozen_json(base/'engineering.json',dict(status='COMPLETE_ACTION_CENSUS_REAL_TRAIN_ENGINEERING',UTC=utc(),rows=rows,
        GT_labels_read=False,new_fits=0,Test_read=False))


def produce(sequence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();plan,protocol,work,parent,base=context();q,a=qualified_input('train',sequence,parent_registered(),all_features=True)
    record=Recorder();teacher=ActionTeacher(r['speed_upper'],record.observe,r['specification']['maximum_local_cost_regret'])
    shaped=ShapeTeacher(r['speed_upper']);shaped.teacher=teacher;digest=hashlib.sha256();packet=hashlib.sha256()
    for frame in range(1,q['frames']+1):
        lo,hi=map(int,a['offsets'][frame-1:frame+1]);d,rgb,ir=[a[k][lo:hi]for k in ['detections','rgb','ir']]
        before=[v.copy()for v in[d,rgb,ir]];output=shaped.update(sequence,frame,d,rgb,ir)
        digest.update(''.join(map(mot_line,output)).encode());packet.update(d.tobytes())
        record.commit_assignments(frame,teacher.current_proposal_assignments(frame))
        for b,c in zip(before,[d,rgb,ir]):np.testing.assert_array_equal(b,c)
        if frame%1000==0 or frame==q['frames']:
            registered();atomic_json(base/'progress'/(sequence+'.json'),dict(sequence=sequence,frame=frame,frames=q['frames'],actions=len(record.frames),GT_labels_read=False))
    original=read(parent/'inputs/train'/(sequence+'.json'))
    assert q['bank_sha256']==original['source_NN_bank_sha256']and digest.hexdigest()==original['original_main_MOT_sha256']
    values=record.arrays(q['frames']);raw_bytes=sum(v.nbytes for v in values.values())
    assert raw_bytes<=r['specification']['per_sequence_raw_array_cap_bytes']
    path=base/'private_source'/(sequence+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb')as stream:np.savez_compressed(stream,**values)
    assert sha256(q['bank_path'])==q['bank_sha256'];registered()
    frozen_json(base/'source_metadata'/(sequence+'.json'),dict(status='COMPLETE_GT_FREE_ACTION_SOURCE',UTC=utc(),sequence=sequence,
        frames=q['frames'],actions=len(record.frames),matches=len(record.matches),current_assignments=len(record.assignments),
        path=str(path),sha256=sha256(path),bytes=path.stat().st_size,raw_payload_bytes=raw_bytes,
        source_NN_bank_sha256=q['bank_sha256'],teacher_MOT_sha256=digest.hexdigest(),original_shaped_teacher_MOT_exact=True,
        concatenated_original_detection_values_sha256=packet.hexdigest(),source_versions=r['source_hashes'],
        GT_labels_parsed=False,future_frame_input=False,modified_tracker_outputs=False,new_weights_images_or_descriptors=False))


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();plan,protocol,work,parent,base=context()
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_ACTION_CENSUS',controller_pid=os.getpid(),phases={},seed=0,new_GPU_work=False,goal_met=False,supported_innovations=0)
        queue=Queue(work,base,state,plan['data_root'],registered);queue.save();cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
        try:
            queue.jobs('ACTION_CENSUS_ENGINEERING',[dict(id='all6',command=cmd+['engineering'])],False,
                lambda j:len(read(base/'engineering.json')['rows'])==6)
            queue.jobs('ACTION_CENSUS_SOURCE',[dict(id=s,command=cmd+['produce','--sequence',s])for s in LENGTHS['train']],False,
                lambda j:read(base/'source_metadata'/(j['id']+'.json'))['original_shaped_teacher_MOT_exact'])
            rows={s:dict(actual_exit_code=j['exit_code'],identity={k:j[k]for k in ['pid','start_ticks','command_sha256']},
                metadata_sha256=sha256(base/'source_metadata'/(s+'.json')))for s,j in state['phases']['ACTION_CENSUS_SOURCE'].items()}
            frozen_json(base/'source_ready.json',dict(status='ALL6_ACTION_SOURCE_WORKERS_ACTUAL_EXIT_ZERO',UTC=utc(),rows=rows,frames=27251,GT_labels_parsed=False))
            queue.jobs('ACTION_CENSUS_TRAIN_LABELS',[dict(id='all6',command=cmd+['analyse'])],False,
                lambda j:read(base/'complete_Train_census.json')['frames']==27251)
            state['status']='COMPLETE_ACTION_CENSUS_REQUIRES_RESEARCH_ASSESSMENT';queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_ACTION_CENSUS_ATTEMPT',error=repr(error));queue.save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run','engineering','produce','analyse']);p.add_argument('--sequence');a=p.parse_args()
    if a.mode=='produce':produce(a.sequence)
    elif a.mode=='analyse':
        from tools.analyse_mft_assignment_actions import analyse
        analyse()
    else:{'register':register,'run':run,'engineering':engineering}[a.mode]()
