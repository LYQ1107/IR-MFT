#!/usr/bin/env python3
"""Full-Dev quality/confirmation comparison and frozen conditional Test7."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.fit_mft_proposal_quality import registered as Train_registered,locations as Train_locations,utc
from tools.mft_proposal_quality import ForegroundView,QualityHead
from tools.mft_proposal_quality_evaluation import policies,matched,quality_anchor,interventions,Test_cases,SharedConfirmation,make_tracker
from tools.fit_mft_candidate_risk import threshold_values
from tools.run_mft_archive_reasoning import fitted as archive_fitted
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.task_utility_queue import Queue
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.fusion_memory_evaluation import run_official_tracks,validate_mot,official_csv47
from tools.export_perceptual_metrics47 import official_schema
from tools.audit_mft_tracklet_r2 import bank_audit
from tools.run_mft_tracklet_permanence_r2 import registered as peer_registered
from tools.mft_candidate_risk_data import load_arrays
from tools.native_pyramid_stream_tracks import mot_line

SPEC=ROOT/'configs/perceptual_proposal_quality_evaluation_20261127.json'
PREFIX='PROPOSAL_QUALITY'


def locations():
    plan,protocol,work,train=Train_locations();return plan,protocol,work,work/'proposal_quality_evaluation_20261127',train


def fitted():
    r=Train_registered();plan,protocol,work,base,train=locations();e=read(train/'fit_r1.execution.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    f=read(train/'fit_complete.json');assert f['Train_frames']==27251 and f['proposals']==249639 and f['gradient_converged']
    assert f['Dev_rows']==f['Test_rows']==0 and f['registration_sha256']==sha256(train/'fit_registered.json')
    assert sha256(f['model_file'])==f['model_sha256'];p=read(f['model_file'])
    views={k:ForegroundView.from_dict(v)for k,v in p['views'].items()};heads={k:QualityHead.from_dict(v)for k,v in p['heads'].items()}
    assert p['thresholds']==f['thresholds'];return r,f,views,heads,f['thresholds']


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r,f,views,heads,thresholds=fitted();plan,protocol,work,base,train=locations();spec=read(SPEC)
    assert spec['normal_cases']==policies()and len(policies())==16
    prior=work/'archive_reasoning_evaluation_20261127/complete_Dev.json';d=read(prior)
    chosen=min(d['rows'],key=lambda x:(*(-x['combined47'][k]for k in ['HOTA','AssA','IDF1']),x['case']['id']))
    assert chosen['case']['id']=='AR_RGB_q999_shape'and not d['Test_read_for_selection']
    sources=dict(r['source_hashes'])
    for n in ['tools/mft_proposal_quality_evaluation.py','tools/run_mft_proposal_quality.py',str(SPEC.relative_to(ROOT))]:sources[n]=sha256(ROOT/n)
    paths=[SPEC,train/'fit_complete.json',train/'fit_r1.execution.json',prior]
    banks={s:bank_audit(work/'tracklet_permanence_20261127_r2','dev',s,peer_registered())for s in LENGTHS['dev']}
    ae=read(train/'audit_r1.execution.json')
    v=dict(status='REGISTERED_ALL16_QUALITY_NORMAL_DEV_CASES_BEFORE_NEW_DEV',UTC=utc(),source_hashes=sources,
        artifact_hashes={str(p):sha256(p)for p in paths},model_sha256=f['model_sha256'],normal_cases=policies(),Dev_frames=4834,
        Dev_banks=banks,parent_best_Dev_HOTA=chosen['combined47']['HOTA'],parent_best_Dev_case=chosen['case'],
        independent_Train_audit_identity=ae['identity'],independent_audit_complete_at_registration=(train/'independent_fit_audit.json').exists(),
        current_Test_used_for_selection=False,historical_Test_previously_exposed=True,
        quality_ROI_modality_controls_do_not_remove_IR_visual_detector=True,goal_met=False,supported_innovations=0)
    assert not(base/'registered.json').exists();frozen_json(base/'registered.json',v)
    print(json.dumps(dict(status=v['status'],sources=len(sources),parent_Dev_HOTA=v['parent_best_Dev_HOTA'])),flush=True)


def registered():
    Train_registered();plan,protocol,work,base,train=locations();r=read(base/'registered.json');assert r['normal_cases']==policies()
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    return r


def prefix(arrays,cases,models,old_models,sequence,count,shared):
    views,heads,thresholds,speed=models;archive_models,archive_thresholds=old_models
    cache=SharedConfirmation()if shared else None
    trackers={c['id']:make_tracker(c,views,heads,thresholds,speed,archive_models,archive_thresholds,cache)for c in cases}
    rows={c['id']:[]for c in cases}
    for i in range(count):
        lo,hi=map(int,arrays['offsets'][i:i+2])
        for c in cases:rows[c['id']].extend(trackers[c['id']].update(sequence,i+1,arrays['detections'][lo:hi],arrays['rgb'][lo:hi],arrays['ir'][lo:hi]))
    return {k:''.join(map(mot_line,v))for k,v in rows.items()}


def engineer():
    reg=registered();_,f,views,heads,thresholds=fitted();plan,protocol,work,base,train=locations()
    _,_,old_models,old_thresholds=archive_fitted();models=(views,heads,thresholds,threshold_values(work)['Train_unknown_speed_upper'])
    old=(old_models,old_thresholds);cases=policies()
    controls=[dict(c,id='MISSING_'+c['id'],intervention='missing')for c in cases if c['kind']=='paired']
    rows={}
    for sequence,q in Train_registered()['qualified_Train'].items():
        a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);allcases=cases+controls
        direct=prefix(a,allcases,models,old,sequence,64,False);shared=prefix(a,allcases,models,old,sequence,64,True)
        assert direct==shared
        past=prefix(a,allcases,models,old,sequence,32,True);assert all(shared[k].startswith(past[k])for k in shared)
        for c in cases:
            if c['kind']=='paired':assert shared['MISSING_'+c['id']]==shared[matched(c,'RGB')['id']]
        at=int(a['offsets'][32]);a['detections'][at:,:4]+=777
        for cue in ['rgb','ir']:a[cue][at:]*=-1
        future=prefix(a,allcases,models,old,sequence,64,True);assert all(future[k].startswith(past[k])for k in future)
        rows[sequence]=dict(frames=64,prefix_frames=32,all20_cases_direct_and_shared_confirmation_MOT_exact=True,
            all16_cases_and4missing_controls_causal_prefix_exact=True,all4_missing_qualityIR_MOT_exact_RGB=True,
            bank_sha256=q['bank_sha256'])
        guard_storage(work)
    assert sha256(f['model_file'])==f['model_sha256'];registered()
    frozen_json(base/'real_Train_engineering.json',dict(status='COMPLETE_FITTED_QUALITY_ALL6_REAL_TRAIN_ENGINEERING',UTC=utc(),
        rows=rows,sequences=6,source_versions=reg['source_hashes'],model_sha256=f['model_sha256'],new_accuracy=False))


def split_cases(split,stage):
    plan,protocol,work,base,train=locations()
    if split=='test':
        f=read(base/'Test_selection_frozen.json');assert not f['Test_read_for_selection']
        assert f['complete_Dev_sha256']==sha256(base/'complete_Dev.json')
        return f['Test_cases']if stage=='normal'else f['Dev_interventions']
    assert split=='dev'
    return policies()if stage=='normal'else read(base/'Test_selection_frozen.json')['Dev_interventions']


def replay(split,stage,sequence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and sequence in LENGTHS[split]
    reg=registered();_,f,views,heads,thresholds=fitted();plan,protocol,work,base,train=locations()
    _,_,old_models,old_thresholds=archive_fitted();cases=split_cases(split,stage)
    label=split if stage=='normal'else split+'_intervention';peer=work/'tracklet_permanence_20261127_r2'
    bank=bank_audit(peer,split,sequence,peer_registered());meta=read(peer/'cache'/split/(sequence+'.json'))
    with np.load(meta['path'],allow_pickle=False)as data:a={k:data[k].copy()for k in data.files}
    speed=threshold_values(work)['Train_unknown_speed_upper'];cache=SharedConfirmation()
    trackers={c['id']:make_tracker(c,views,heads,thresholds,speed,old_models,old_thresholds,cache)for c in cases}
    writer=FusionMemoryStreamWriter(base/'streams'/label,sequence,[c['id']for c in cases]);started=time.monotonic()
    try:
        for i,fid in enumerate(a['frame_ids']):
            frame=int(fid);lo,hi=map(int,a['offsets'][i:i+2]);d=a['detections'][lo:hi]
            rows={c['id']:trackers[c['id']].update(sequence,frame,d,a['rgb'][lo:hi],a['ir'][lo:hi])for c in cases}
            # Packet receipts track ORIGINAL detector evidence. Calibrated
            # tracker-input score hashes are recorded separately below.
            writer.append(frame,rows,{c['id']:d for c in cases})
            if frame%250==0 or frame==LENGTHS[split][sequence]:
                guard_storage(work);atomic_json(base/'progress'/label/(sequence+'.json'),dict(sequence=sequence,frame=frame,
                    frames=LENGTHS[split][sequence],seconds=time.monotonic()-started))
        streams=writer.close(LENGTHS[split][sequence])
    finally:writer.abort()
    if stage=='normal':
        old=read(work/'archive_reasoning_evaluation_20261127/streams'/split/'sequence_manifests'/(sequence+'.json'))
        for own,ref in [('PQ_native_control','AR_native_zero'),('PQ_archive_control','AR_RGB_q999_shape')]:
            if own in streams:assert streams[own]['track_sha256']==old['streams'][ref]['track_sha256']
    else:
        freeze=read(base/'Test_selection_frozen.json');source=quality_anchor(freeze['chosen_case']);rgb=matched(source,'RGB')['id']
        normal=read(base/'streams'/split/'sequence_manifests'/(sequence+'.json'))
        assert streams['PQ_control_missing']['track_sha256']==normal['streams'][rgb]['track_sha256']
    assert all(s['detection_packet_sha256']==bank['packet_sha256']for s in streams.values())
    assert sha256(f['model_file'])==f['model_sha256'];registered()
    frozen_json(base/'streams'/label/'sequence_manifests'/(sequence+'.json'),dict(status='COMPLETE_CAUSAL_QUALITY_FULL_SEQUENCE',UTC=utc(),
        split=split,stage=stage,sequence=sequence,frames=LENGTHS[split][sequence],cases=cases,streams=streams,bank=bank,
        source_versions=reg['source_hashes'],model_sha256=f['model_sha256'],statistics={k:t.statistics for k,t in trackers.items()},
        calibrated_tracker_input_packet_sha256={k:t.calibrated_packet_hash.hexdigest()for k,t in trackers.items()if hasattr(t,'calibrated_packet_hash')},
        original_detector_packet_unchanged_but_selected_tracker_input_scores_promoted=True,
        original_references_or_missing_RGB_MOT_exact=True,GT_labels_read=False,future_frame_input=False,
        past_outputs_rewritten=False,model_bytes_unchanged=True,shared_original_confirmation=True,
        cached_CPU_replay_seconds=time.monotonic()-started,neural_encoders_image_IO_and_IR_generation_excluded=True))


def evaluate(split,stage,case_id):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();_,f,views,heads,thresholds=fitted();plan,protocol,work,base,train=locations()
    case=next(c for c in split_cases(split,stage)if c['id']==case_id);label=split if stage=='normal'else split+'_intervention';coverage={}
    for sequence,frames in LENGTHS[split].items():
        m=read(base/'streams'/label/'sequence_manifests'/(sequence+'.json'));assert m['frames']==frames and m['cases']==split_cases(split,stage)
        assert m['source_versions']==reg['source_hashes']and m['model_sha256']==f['model_sha256']and not m['GT_labels_read']
        s=m['streams'][case_id];assert sha256(s['track_path'])==s['track_sha256'];validate_mot(s['track_path'],frames,s['track_rows']);coverage[sequence]=s
    output=base/'evaluation'/label/case_id;assert not output.exists();output.mkdir(parents=True)
    values=run_official_tracks(base/'streams'/label/case_id/'track_results',output,plan,split);registered()
    frozen_json(output/'metrics.json',dict(status='COMPLETE_OFFICIAL_QUALITY_ALL47',case=case,frames=sum(LENGTHS[split].values()),
        coverage=coverage,source_versions=reg['source_hashes'],model_sha256=f['model_sha256'],**values,goal_met=False,supported_innovations=0))


def phase(queue,split,stage='normal'):
    plan,protocol,work,base,train=locations();cases=split_cases(split,stage);label=split if stage=='normal'else split+'_intervention'
    cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))];extra=['--split',split,'--stage',stage]
    queue.jobs(PREFIX+'_'+label.upper()+'_REPLAY',[dict(id=s,command=cmd+['replay',*extra,'--sequence',s])for s in LENGTHS[split]],False,
        lambda j:read(base/'streams'/label/'sequence_manifests'/(j['id']+'.json'))['original_references_or_missing_RGB_MOT_exact'])
    queue.jobs(PREFIX+'_'+label.upper()+'_OFFICIAL',[dict(id=c['id'],command=cmd+['evaluate',*extra,'--case',c['id']])for c in cases],False,
        lambda j:read(base/'evaluation'/label/j['id']/'metrics.json')['frames']==sum(LENGTHS[split].values()))
    return [read(base/'evaluation'/label/c['id']/'metrics.json')for c in cases]


def wait_audit(queue):
    reg=registered();plan,protocol,work,base,train=locations()
    while True:
        e=read(train/'audit_r1.execution.json');assert e['identity']==reg['independent_Train_audit_identity']
        if e['actual_child_exit_code']is not None:
            assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
            a=read(train/'independent_fit_audit.json');assert a['all6_Train_replayed']and a['model_sha256']==reg['model_sha256'];return
        live=identity(e['identity']['PID']);assert live is not None,'Retain unknown audit exit rather than assuming success'
        assert all(live[k]==e['identity'][k]for k in ['PID','start_ticks','raw_argv_sha256'])
        queue.state.update(status='WAITING_ACTUAL_FULL_TRAIN_QUALITY_AUDIT',last_wait_UTC=utc());queue.save()
        registered();guard_storage(work);time.sleep(5)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,train=locations()
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_QUALITY_FULL_DEV',controller_pid=os.getpid(),phases={},seed=0,new_GPU_work=False,goal_met=False,supported_innovations=0)
        queue=Queue(work,base,state,plan['data_root'],registered);queue.save();cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
        try:
            queue.jobs(PREFIX+'_ENGINEERING',[dict(id='all6',command=cmd+['engineer'])],False,
                lambda j:read(base/'real_Train_engineering.json')['sequences']==6)
            rows=phase(queue,'dev');prior=read(work/'archive_reasoning_evaluation_20261127/complete_Dev.json')
            for own,previous in [('PQ_native_control','AR_native_zero'),('PQ_archive_control','AR_RGB_q999_shape')]:
                a=next(r for r in rows if r['case']['id']==own);b=next(r for r in prior['rows']if r['case']['id']==previous)
                assert a['combined47']==b['combined47']and a['per_sequence47']==b['per_sequence47']
            chosen=min(rows,key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
            gain=chosen['combined47']['HOTA']-reg['parent_best_Dev_HOTA']
            frozen_json(base/'complete_Dev.json',dict(status='COMPLETE_ALL16_NORMAL_QUALITY_FULL_DEV',UTC=utc(),rows=rows,chosen_case=chosen['case'],
                gain_over_parent_best_Dev_pp=gain,parent_best_Dev_HOTA=reg['parent_best_Dev_HOTA'],Test_read_for_selection=False,
                both_references_all47_and_MOT_exact=True,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0))
            if gain<=1e-8:
                state.update(status='COMPLETE_QUALITY_NO_DEV_GAIN_NO_NEW_TEST',Dev_gain_pp=gain);queue.save();return
            case=chosen['case'];assert case['kind']in ['unconditional','RGB','paired','duplicatedRGB']
            frozen_json(base/'Test_selection_frozen.json',dict(status='FROZEN_COMPLETE_DEV_QUALITY_SELECTION_AND_CONTROLS',UTC=utc(),
                chosen_case=case,Test_cases=Test_cases(case),Dev_interventions=interventions(case),
                complete_Dev_sha256=sha256(base/'complete_Dev.json'),Test_read_for_selection=False,
                no_IR_quality_innovation_claim_if_unconditional_wins=True,historical_Test_previously_exposed=True))
            controls=phase(queue,'dev','intervention');rgb=matched(quality_anchor(case),'RGB')
            source=next(r for r in rows if r['case']==rgb)
            assert controls[0]['combined47']==source['combined47']and controls[0]['per_sequence47']==source['per_sequence47']
            frozen_json(base/'complete_Dev_interventions.json',dict(status='COMPLETE_FROZEN_QUALITY_DEV_INTERVENTIONS_AND_ABLATIONS',UTC=utc(),rows=controls,
                missing_qualityIR_fullMOT_and47_exact_RGB=True,no_reselection=True,Test_read_for_selection=False,
                control_scope='ROIqualityIR only;fixedIRvisualdetector isnotintervened',goal_met=False,supported_innovations=0))
            wait_audit(queue);test=phase(queue,'test');selected=next(r for r in test if r['case']==case)
            csv=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            schema,_=official_schema(plan['evaluation']['TrackEval_root']);baseline=official_csv47(csv,schema,'test',read(csv.parent.parent/'metrics.json')['trackeval'])
            delta=selected['combined47']['HOTA']-baseline['combined47']['HOTA']
            frozen_json(base/'full_Test_results.json',dict(status='COMPLETE_DEV_FROZEN_QUALITY_FULL7_TEST',UTC=utc(),rows=test,selected_case=case,
                original_RGB_baseline=baseline,HOTA_gain_over_original_RGB_pp=delta,target_plus5_met=delta>=5,target_plus10_met=delta>=10,
                Test_used_for_selection=False,controls_not_promoted_from_Test=True,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0))
            if delta>=5:
                ablation=phase(queue,'test','intervention')
                frozen_json(base/'full_Test_interventions.json',dict(status='COMPLETE_FROZEN_QUALITY_FULL7_TEST_ABLATIONS',UTC=utc(),rows=ablation,
                    no_reselection=True,quality_IR_control_scope_not_whole_visualIR=True,goal_met=False,supported_innovations=0))
            state.update(status='COMPLETE_QUALITY_FULL_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT',HOTA_gain_over_original_RGB_pp=delta);queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_QUALITY_EVALUATION_ATTEMPT',error=repr(error));queue.save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run','engineer','replay','evaluate'])
    p.add_argument('--split',choices=['dev','test']);p.add_argument('--stage',choices=['normal','intervention'],default='normal');p.add_argument('--sequence');p.add_argument('--case');a=p.parse_args()
    if a.mode=='register':register()
    elif a.mode=='run':run()
    elif a.mode=='engineer':engineer()
    elif a.mode=='replay':replay(a.split,a.stage,a.sequence)
    else:evaluate(a.split,a.stage,a.case)
