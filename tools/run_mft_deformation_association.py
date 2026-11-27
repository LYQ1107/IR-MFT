#!/usr/bin/env python3
"""Fixed score/shape association factorial on qualified native cached inputs."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.mft_deformation_association import AssociationStreams,policies
from tools.mft_protected_peripheral import ProtectedStreams,PeripheralPublisher
from tools.mft_protected_peripheral_experiment import (locations,registered as parent_registered,
    qualified_input,models)
from tools.diagnose_mft_birth_only import state_equal
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_stream_tracks import mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256
from tools.fit_mft_proposal_quality import utc
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_evaluation import validate_mot,run_official_tracks,official_csv47
from tools.run_mft_coverage_recall import window_csv47
from tools.export_perceptual_metrics47 import official_schema
from tools.task_utility_queue import Queue

SOURCES=['tools/mft_deformation_association.py','tools/run_mft_deformation_association.py',
    'tests/test_mft_deformation_association.py','configs/perceptual_deformation_association_20261127.json']


def context():
    plan,protocol,work,parent,previous=locations()
    return plan,protocol,work,parent,work/'deformation_association_20261127'


def register():
    old=parent_registered();plan,protocol,work,parent,base=context();base.mkdir(parents=True,exist_ok=True)
    for name in ['run_r1.execution.json','audit_r2.execution.json']:
        e=read(parent/name);assert e['actual_child_exit_code']==0 and e['source_unchanged']
    selected=read(parent/'Test_selection_frozen.json')['chosen_case']
    assert selected['id']=='PP_original_paired_t025'
    val=read(parent/'complete_forward_Train_validation.json');dev=read(parent/'complete_Dev.json')
    source=dict(old['source_hashes']);source.update({n:sha256(ROOT/n)for n in SOURCES})
    paths=[parent/n for n in ['registered.json','source_Train_Dev_ready.json','run_r1.execution.json',
        'audit_r2.execution.json','Test_selection_frozen.json','complete_forward_Train_validation.json','complete_Dev.json']]
    spec=read(ROOT/SOURCES[-1]);budget=guard_storage(work)
    assert budget['work_file_bytes']+spec['new_study_private_file_cap_bytes']<budget['work_cap_bytes']
    frozen_json(base/'registered.json',dict(status='REGISTERED_DEFORMATION_SCORE_FACTORIAL_BEFORE_NEW_METRICS',UTC=utc(),
        source_hashes=source,artifact_hashes={str(p):sha256(p)for p in paths},specification=spec,
        normal_cases=policies(),selected_parent=selected,qualified=old['qualified'],validation_GT_views=old['validation_GT_views'],
        validation_lengths=old['specification']['validation_lengths'],
        parent_validation47=next(v['combined47']for v in val['rows']if v['case']==selected),
        parent_Dev47=next(v['combined47']for v in dev['rows']if v['case']==selected),
        new_Test_metrics_read_for_registration=False,historical_Test_exposed=True,budget=budget,
        goal_met=False,supported_innovations=0))


def registered():
    plan,protocol,work,parent,base=context();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    guard_storage(work)
    assert sum(p.stat().st_size for p in base.rglob('*')if p.is_file())<=r['specification']['new_study_private_file_cap_bytes']
    return r


def engineering():
    r=registered();plan,protocol,work,parent,base=context();m=models();records=[]
    for sequence in LENGTHS['train']:
        q,a=qualified_input('train',sequence,r,all_features=True);n=min(64,q['frames'])
        def replay(future):
            streams={c['id']:AssociationStreams(c,*m[:6],standalone=True)for c in policies()}
            pubs={c['id']:PeripheralPublisher(r['selected_parent'])for c in policies()}
            original=ProtectedStreams(*m[:6],standalone=True);original_pub=PeripheralPublisher(r['selected_parent'])
            prefix={key:hashlib.sha256()for key in pubs}
            for frame in range(1,n+1):
                lo,hi=map(int,a['offsets'][frame-1:frame+1])
                d,rgb,ir=[a[k][lo:hi].copy()for k in ['detections','rgb','ir']]
                if future and frame>32:d[:,:4]+=777;rgb*=-1;ir*=-1
                before=[v.copy()for v in [d,rgb,ir]]
                expected=''.join(map(mot_line,original_pub.update(original.update(sequence,frame,d,rgb,ir))))
                for key,source in streams.items():
                    value=source.update(sequence,frame,d,rgb,ir);output=pubs[key].update(value)
                    text=''.join(map(mot_line,output))
                    if key=='DA_reference':
                        assert text==expected;state_equal(source.main.teacher,original.main.teacher)
                    if frame<=32:prefix[key].update(text.encode())
                    assert source.main.teacher.core.args.mot20==next(c['without_score_fusion']for c in policies()if c['id']==key)
                for old,new in zip(before,[d,rgb,ir]):np.testing.assert_array_equal(old,new)
            return {k:v.hexdigest()for k,v in prefix.items()}
        assert replay(False)==replay(True)
        records.append(dict(sequence=sequence,frames=n,causal_prefix_frames=32,
            zero_MOT_and_original_hidden_state_exact=True,input_arrays_unchanged=True))
    registered();frozen_json(base/'engineering.json',dict(status='COMPLETE_DEFORMATION_REAL_TRAIN_ENGINEERING',
        UTC=utc(),rows=records,GT_labels_read=False,Test_read=False,new_GPU_work=False))


def cases_for(split):
    if split!='test':return policies()
    plan,protocol,work,parent,base=context();f=read(base/'Test_selection_frozen.json')
    assert f['Test_cases']==policies()and not f['Test_read_for_selection']
    return f['Test_cases']


def publish(split,sequence):
    r=registered();plan,protocol,work,parent,base=context();logical='train'if split=='trainval'else split
    cases=cases_for(split);q,a=qualified_input(logical,sequence,r,all_features=True);m=models()
    sources={c['id']:AssociationStreams(c,*m[:6],standalone=(logical=='train'))for c in cases}
    pubs={key:PeripheralPublisher(r['selected_parent'])for key in sources}
    digests={key:hashlib.sha256()for key in sources};projected={key:hashlib.sha256()for key in sources}
    counts={key:0 for key in sources};files={};cut=r['validation_GT_views'][sequence]['window_cut']if logical=='train'else 0
    started=time.monotonic();input_digest=hashlib.sha256()
    for key in sources:
        p=base/'streams'/split/key/'track_results'/(sequence+'.txt');p.parent.mkdir(parents=True,exist_ok=True);files[key]=(p,p.open('x'))
    try:
        for frame in range(1,q['frames']+1):
            lo,hi=map(int,a['offsets'][frame-1:frame+1]);d,rgb,ir=[a[k][lo:hi]for k in ['detections','rgb','ir']]
            before=[v.copy()for v in [d,rgb,ir]];input_digest.update(d.tobytes())
            for key,source in sources.items():
                value=source.update(sequence,frame,d,rgb,ir);output=pubs[key].update(value)
                projected[key].update(''.join(map(mot_line,value['main_rows'])).encode())
                assert len(output)>=len(value['main_rows'])
                for (f,box,base_id,score),observed in zip(value['main_rows'],output):
                    assert observed[0]==f and observed[2]==pubs[key].main_alias[base_id]and observed[3]==score
                    np.testing.assert_array_equal(observed[1],box)
                if frame>cut:
                    text=''.join(mot_line((f-cut,box,identity,score))for f,box,identity,score in output)
                    files[key][1].write(text);digests[key].update(text.encode());counts[key]+=len(output)
            for old,new in zip(before,[d,rgb,ir]):np.testing.assert_array_equal(old,new)
            if frame%2000==0:registered()
    finally:
        for p,stream in files.values():stream.close()
    manifest={}
    for key,source in sources.items():
        p=files[key][0];assert sha256(p)==digests[key].hexdigest()
        assert projected[key].hexdigest()==pubs[key].main_projection.hexdigest()
        validate_mot(p,q['frames']-cut,counts[key])
        manifest[key]=dict(track_path=str(p),track_sha256=sha256(p),track_rows=counts[key],frames=q['frames']-cut,
            publisher_statistics=pubs[key].statistics,source_statistics=dict(source.main.statistics),
            main_aliases=pubs[key].main_alias,source_main_MOT_sha256=projected[key].hexdigest(),
            original_observed_scores_boxes_and_order_preserved=True)
    old=read(parent/'streams'/split/'sequence_manifests'/(sequence+'.json'))
    assert manifest['DA_reference']['track_sha256']==old['streams'][r['selected_parent']['id']]['track_sha256']
    frozen_json(base/'streams'/split/'sequence_manifests'/(sequence+'.json'),dict(status='COMPLETE_GT_FREE_DEFORMATION_SEQUENCE',
        UTC=utc(),split=split,sequence=sequence,frames=q['frames']-cut,all_past_frames_consumed=q['frames'],validation_window_cut=cut,
        cases=cases,streams=manifest,source_NN_bank_sha256=q['bank_sha256'],source_versions=r['source_hashes'],
        original_detector_packet_sha256=input_digest.hexdigest(),reference_PP_MOT_exact=True,
        publisher_retains_own_main_rows=True,alternative_associations_may_change_subsequent_state=True,
        GT_labels_read=False,future_frame_input=False,past_outputs_rewritten=False,
        seconds=time.monotonic()-started,cost_excludes_neural_image_and_generation=True))
    registered()


def evaluate(split,case_id):
    r=registered();plan,protocol,work,parent,base=context();case=next(c for c in cases_for(split)if c['id']==case_id)
    lengths=r['validation_lengths']if split=='trainval'else LENGTHS[split]
    for sequence,n in lengths.items():
        m=read(base/'streams'/split/'sequence_manifests'/(sequence+'.json'));v=m['streams'][case_id]
        assert m['frames']==n and m['reference_PP_MOT_exact']and m['publisher_retains_own_main_rows']
        assert sha256(v['track_path'])==v['track_sha256']
    directory=base/'streams'/split/case_id/'track_results';output=base/'evaluation'/split/case_id
    assert not output.exists();output.mkdir(parents=True)
    if split=='trainval':
        from tools.eval_mft26ir import run_trackeval
        gt=Path(next(iter(r['validation_GT_views'].values()))['validation_GT_path']).parents[3]
        headline=run_trackeval(str(directory),str(gt),str(output),list(lengths),plan['evaluation']['TrackEval_root'],data_split='train')
        schema,families=official_schema(plan['evaluation']['TrackEval_root'])
        values=dict(metrics39=headline,official_schema=schema,official_family_field_counts=families,
            **window_csv47(output/'trackeval_output/pedestrian_detailed.csv',schema,lengths,headline))
    else:values=run_official_tracks(directory,output,plan,split)
    frozen_json(output/'metrics.json',dict(status='COMPLETE_DEFORMATION_OFFICIAL47',case=case,frames=sum(lengths.values()),
        source_versions=r['source_hashes'],**values))
    rows=[]
    for sequence in lengths:
        p=output/'trackeval_format'/(sequence+'.txt');canonical=directory/(sequence+'.txt')
        assert sha256(p)==sha256(canonical);rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p),retained_canonical=str(canonical)))
    for v in rows:Path(v['path']).unlink()
    (output/'trackeval_format').rmdir();frozen_json(output/'derived_duplicate_cleanup.json',dict(rows=rows,canonical_retained=True));registered()


def phase(queue,split):
    r=registered();plan,protocol,work,parent,base=context();cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
    lengths=r['validation_lengths']if split=='trainval'else LENGTHS[split]
    queue.jobs('DEFORMATION_'+split.upper()+'_PUBLISH',[dict(id=s,command=cmd+['publish','--split',split,'--sequence',s])for s in lengths],False,
        lambda j:read(base/'streams'/split/'sequence_manifests'/(j['id']+'.json'))['reference_PP_MOT_exact'])
    queue.jobs('DEFORMATION_'+split.upper()+'_OFFICIAL',[dict(id=c['id'],command=cmd+['evaluate','--split',split,'--case',c['id']])for c in cases_for(split)],False,
        lambda j:read(base/'evaluation'/split/j['id']/'metrics.json')['frames']==sum(lengths.values()))
    return [read(base/'evaluation'/split/c['id']/'metrics.json')for c in cases_for(split)]


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();plan,protocol,work,parent,base=context()
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_DEFORMATION_FACTORIAL',controller_pid=os.getpid(),phases={},seed=0,new_GPU_work=False,goal_met=False,supported_innovations=0)
        queue=Queue(work,base,state,plan['data_root'],registered);queue.save()
        try:
            cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
            queue.jobs('DEFORMATION_ENGINEERING',[dict(id='all6',command=cmd+['engineering'])],False,
                lambda j:len(read(base/'engineering.json')['rows'])==6)
            val=phase(queue,'trainval');dev=phase(queue,'dev')
            ref=next(v for v in val if v['case']['id']=='DA_reference');dref=next(v for v in dev if v['case']['id']=='DA_reference')
            assert ref['combined47']==r['parent_validation47']and dref['combined47']==r['parent_Dev47']
            gains={v['case']['id']:v['combined47']['HOTA']-ref['combined47']['HOTA']for v in val}
            eligible=[v for v in dev if v['case']['id']!='DA_reference'and
                gains[v['case']['id']]>=r['specification']['minimum_validation_HOTA_gain_over_PP_pp']and
                v['combined47']['HOTA']>=dref['combined47']['HOTA']-r['specification']['Dev_noninferiority_tolerance_pp']]
            chosen=min(eligible,key=lambda v:(-gains[v['case']['id']],*(-v['combined47'][k]for k in ['HOTA','AssA','IDF1']),v['case']['id']))if eligible else None
            frozen_json(base/'complete_Train_Dev.json',dict(status='COMPLETE_DEFORMATION_TRAIN_DEV_FACTORIAL',UTC=utc(),
                validation_rows=val,Dev_rows=dev,validation_HOTA_gains_over_PP_pp=gains,eligible_ids=[v['case']['id']for v in eligible],
                selected_case=None if chosen is None else chosen['case'],Test_read_for_selection=False,historical_Test_exposed=True,
                goal_met=False,supported_innovations=0))
            if chosen is None:
                state['status']='COMPLETE_DEFORMATION_NO_ELIGIBLE_TRAIN_DEV_CASE_NO_TEST';queue.save();return
            frozen_json(base/'Test_selection_frozen.json',dict(status='FROZEN_DEFORMATION_FROM_TRAIN_DEV',UTC=utc(),
                chosen_case=chosen['case'],Test_cases=policies(),complete_Train_Dev_sha256=sha256(base/'complete_Train_Dev.json'),Test_read_for_selection=False))
            rows=phase(queue,'test');selected=next(v for v in rows if v['case']==chosen['case'])
            csv=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            schema,_=official_schema(plan['evaluation']['TrackEval_root']);baseline=official_csv47(csv,schema,'test',read(csv.parent.parent/'metrics.json')['trackeval'])
            gain=selected['combined47']['HOTA']-baseline['combined47']['HOTA']
            frozen_json(base/'full_Test_results.json',dict(status='COMPLETE_FROZEN_DEFORMATION_TEST7',UTC=utc(),rows=rows,selected_case=chosen['case'],
                original_RGB_baseline=baseline,HOTA_gain_over_original_RGB_pp=gain,target_plus5_met=gain>=5,target_plus10_met=gain>=10,
                Test_used_for_selection=False,controls_not_promoted=True,historical_Test_exposed=True,goal_met=False,supported_innovations=0))
            state.update(status='COMPLETE_DEFORMATION_TEST_REQUIRES_ASSESSMENT',HOTA_gain_over_original_RGB_pp=gain);queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_DEFORMATION_ATTEMPT',error=repr(error));queue.save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run','engineering','publish','evaluate'])
    p.add_argument('--split',choices=['trainval','dev','test']);p.add_argument('--sequence');p.add_argument('--case');a=p.parse_args()
    if a.mode=='publish':publish(a.split,a.sequence)
    elif a.mode=='evaluate':evaluate(a.split,a.case)
    else:{'register':register,'run':run,'engineering':engineering}[a.mode]()
