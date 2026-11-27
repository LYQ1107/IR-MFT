#!/usr/bin/env python3
"""Prospective forward-Train validation, full-Dev gate, then frozen Test7."""
import argparse
import configparser
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.mft_coverage_recall import CoverageRecallTracker,SharedCoverage,policies
from tools.mft_birth_only import BirthOnlyQuality
from tools.mft_proposal_quality_evaluation import ShapeTeacher
from tools.run_mft_proposal_quality import fitted,locations as parent_locations
from tools.run_mft_birth_only import registered as birth_registered
from tools.run_mft_archive_reasoning import fitted as archive_fitted
from tools.diagnose_mft_scene_coverage import load_bank
from tools.fit_mft_candidate_risk import threshold_values
from tools.fit_mft_proposal_quality import utc
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.native_pyramid_stream_tracks import mot_line
from tools.task_utility_queue import Queue
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.fusion_memory_evaluation import run_official_tracks,official_csv47,validate_mot
from tools.export_perceptual_metrics47 import official_schema
from tools.audit_mft_tracklet_r2 import bank_audit
from tools.run_mft_tracklet_permanence_r2 import registered as peer_registered

SPEC=ROOT/'configs/perceptual_coverage_recall_20261127.json'
NAMES=['tools/mft_coverage_recall.py','tests/test_mft_coverage_recall.py',
    'tools/run_mft_coverage_recall.py',str(SPEC.relative_to(ROOT))]
PREFIX='COVERAGE_RECALL'


def locations():
    plan,protocol,work,peer,train=parent_locations()
    return plan,protocol,work,work/'coverage_recall_20261127'


def models():
    _,fit,views,heads,thresholds=fitted();_,afit,am,at=archive_fitted()
    plan,protocol,work,base=locations();speed=threshold_values(work)['Train_unknown_speed_upper']
    return views,heads,thresholds,speed,am,at,fit,afit


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,base=locations();spec=read(SPEC);assert spec['normal_cases']==policies()
    views,heads,thresholds,speed,am,at,fit,afit=models();parent=birth_registered()
    diag=work/'scene_coverage_diagnostic_20261127';dr=read(diag/'registered.json')
    for name in ['run_r1','audit_r1']:
        e=read(diag/(name+'.execution.json'));assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    proof=read(base/'unit_regressions.json');assert proof['tests_passed']==3 and proof['actual_tool_exit_code']==0
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n
    sources={**parent['source_hashes'],**dr['source_hashes'],**{n:sha256(ROOT/n)for n in NAMES}}
    sources['tools/audit_mft_scene_coverage.py']=sha256(ROOT/'tools/audit_mft_scene_coverage.py')
    for n,h in sources.items():assert sha256(ROOT/n)==h,n
    prior=read(work/'birth_only_evaluation_20261127/complete_Dev.json')
    selected=next(v for v in prior['rows']if v['case']['id']=='BO_paired_q999_c3')
    assert selected['combined47']['HOTA']==spec['full_Dev_noninferiority_HOTA']
    gt_views={}
    for sequence,q in dr['qualified']['train'].items():
        assert sha256(q['GT_path'])==q['GT_sha256'];cut=int(.75*q['frames']);n=q['frames']-cut
        assert n==spec['validation_lengths'][sequence]
        directory=base/'validation_gt/train'/sequence/'gt';directory.mkdir(parents=True,exist_ok=True)
        destination=directory/'gt.txt';written=0
        with destination.open('x')as target,Path(q['GT_path']).open()as source:
            for line in source:
                cells=line.rstrip('\r\n').split(',');frame=int(cells[0])
                if frame>cut:cells[0]=str(frame-cut);target.write(','.join(cells)+'\n');written+=1
        ini=configparser.ConfigParser();original=Path(q['GT_path']).parent.parent/'seqinfo.ini';ini.read(original)
        ini['Sequence']['seqLength']=str(n)
        with (directory.parent/'seqinfo.ini').open('x')as target:ini.write(target)
        gt_views[sequence]=dict(original_GT_sha256=q['GT_sha256'],original_seqinfo_sha256=sha256(original),
            validation_GT_path=str(destination),validation_GT_sha256=sha256(destination),
            validation_seqinfo_sha256=sha256(directory.parent/'seqinfo.ini'),window_cut=cut,frames=n,GT_rows=written)
    paths=[SPEC,base/'unit_regressions.json',diag/'registered.json',diag/'complete_Train_Dev.json',
        diag/'independent_scalar_audit.json',diag/'run_r1.execution.json',diag/'audit_r1.execution.json',
        work/'birth_only_evaluation_20261127/complete_Dev.json',work/'proposal_quality_20261127/fit_complete.json']
    guard_storage(work)
    frozen_json(base/'registered.json',dict(status='REGISTERED_COVERAGE_RECALL_BEFORE_NEW_CLOSED_LOOP_HOTA',UTC=utc(),
        source_hashes=sources,artifact_hashes={str(p):sha256(p)for p in paths},specification=spec,
        qualified=dr['qualified'],validation_GT_views=gt_views,quality_model_sha256=fit['model_sha256'],
        archive_model_sha256=afit['model_sha256'],full_parent_Dev47=selected['combined47'],
        normal_cases=policies(),Train_warmup_and_validation_frames=27251,Train_validation_scored_frames=6814,
        Dev_frames=4834,Test_read_for_selection=False,historical_Test_exposed=True,
        standalone_validation_excludes_late_calibrated_quality_and_archive_thresholds=True,
        full_Dev_parent_uses_prior_Train_calibrations=True,encoder_Train_identity_exposure=True,
        exact_Dev_ties_prospectively_eligible=True,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='REGISTERED_COVERAGE_RECALL',sources=len(sources),normal_cases=len(policies()),validation_frames=6814)),flush=True)


def registered():
    plan,protocol,work,base=locations();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    for sequence,v in r['validation_GT_views'].items():
        p=Path(v['validation_GT_path']);assert sha256(p)==v['validation_GT_sha256']
        assert sha256(p.parent.parent/'seqinfo.ini')==v['validation_seqinfo_sha256']
    return r


def prefix(sequence,a,count,m,cases,standalone):
    views,heads,thresholds,speed,am,at,_,_=m;shared=SharedCoverage(speed,views,heads)
    trackers={c['id']:CoverageRecallTracker(c,views,heads,thresholds,speed,am,at,standalone=standalone,shared=shared)for c in cases}
    if standalone:parent=ShapeTeacher(speed)
    else:parent=BirthOnlyQuality(dict(id='BO_paired_q999_c3',kind='paired',quantile='q999',confirmation=3),views,heads,thresholds,speed,am,at)
    outputs={c['id']:[]for c in cases};outputs['frozen_parent']=[]
    for i in range(count):
        lo,hi=map(int,a['offsets'][i:i+2]);d=a['detections'][lo:hi];rgb=a['rgb'][lo:hi];ir=a['ir'][lo:hi]
        for k,t in trackers.items():outputs[k].extend(t.update(sequence,i+1,d,rgb,ir))
        outputs['frozen_parent'].extend(parent.update(sequence,i+1,d,rgb,ir))
    return {k:''.join(map(mot_line,v))for k,v in outputs.items()}


def engineer():
    r=registered();m=models();plan,protocol,work,base=locations();result={}
    cases=policies();controls=[dict(c,id='MISSING_'+c['id'],intervention='missing')for c in cases if c['kind']=='paired']
    for sequence,q in r['qualified']['train'].items():
        a=load_bank(q,'train');summary={}
        for standalone in [True,False]:
            original=prefix(sequence,a,64,m,cases+controls,standalone)
            assert original['CR_reference']==original['frozen_parent']
            short=prefix(sequence,a,32,m,cases+controls,standalone)
            assert all(original[k].startswith(v)for k,v in short.items())
            for c in cases:
                if c['kind']=='paired':
                    rgb=next(v for v in cases if v['kind']=='RGB'and v['threshold']==c['threshold'])
                    assert original['MISSING_'+c['id']]==original[rgb['id']]
            summary['standalone'if standalone else'full_parent']=dict(frames=64,
                zero_recall_parent_MOT_exact=True,causal_prefix32_exact=True,missing_gate_IR_MOT_exact_RGB=True)
        result[sequence]=summary;guard_storage(work)
        print(json.dumps(dict(sequence=sequence,status='ENGINEERING_COMPLETE')),flush=True)
    registered()
    frozen_json(base/'real_Train_engineering.json',dict(status='COMPLETE_ALL6_COVERAGE_RECALL_ENGINEERING',UTC=utc(),
        sequences=6,rows=result,GT_labels_read=False,new_accuracy=False))


def cases_for(split,stage):
    plan,protocol,work,base=locations()
    assert split in ['trainval','dev','test']and stage in ['normal','intervention']
    if split=='test' or stage=='intervention':
        freeze=read(base/'Test_selection_frozen.json');assert not freeze['Test_read_for_selection']
        assert freeze['complete_Dev_sha256']==sha256(base/'complete_Dev.json')
        return freeze['interventions']if stage=='intervention'else freeze['Test_cases']
    return policies()


def replay(split,stage,sequence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();m=models();views,heads,thresholds,speed,am,at,fit,afit=m;plan,protocol,work,base=locations()
    cases=cases_for(split,stage);label=split if stage=='normal'else split+'_intervention'
    if split=='trainval':
        q=r['qualified']['train'][sequence];a=load_bank(q,'train');frames=q['frames'];cut=r['validation_GT_views'][sequence]['window_cut']
        bank=dict(bank_sha256=q['bank_sha256'],frames=frames,GT_not_read_by_replay=True)
    else:
        assert sequence in LENGTHS[split]
        peer=work/'tracklet_permanence_20261127_r2';bank=bank_audit(peer,split,sequence,peer_registered())
        meta=read(peer/'cache'/split/(sequence+'.json'))
        with np.load(meta['path'],allow_pickle=False)as z:a={k:z[k]for k in z.files}
        frames=LENGTHS[split][sequence];cut=0
    shared=SharedCoverage(speed,views,heads)
    trackers={c['id']:CoverageRecallTracker(c,views,heads,thresholds,speed,am,at,standalone=split=='trainval',shared=shared)for c in cases}
    writer=FusionMemoryStreamWriter(base/'streams'/label,sequence,[c['id']for c in cases]);started=time.monotonic()
    try:
        for i in range(frames):
            frame=i+1;lo,hi=map(int,a['offsets'][i:i+2]);d=a['detections'][lo:hi]
            rows={k:t.update(sequence,frame,d,a['rgb'][lo:hi],a['ir'][lo:hi])for k,t in trackers.items()}
            if frame>cut:
                shifted={k:[(f-cut,box,identity,score)for f,box,identity,score in v]for k,v in rows.items()}
                writer.append(frame-cut,shifted,{k:d for k in trackers})
            if frame%500==0 or frame==frames:
                guard_storage(work);atomic_json(base/'progress'/label/(sequence+'.json'),dict(sequence=sequence,frame=frame,frames=frames,
                    scored_window_frames=max(0,frame-cut),seconds=time.monotonic()-started))
        streams=writer.close(frames-cut)
    finally:writer.abort()
    if split in ['dev','test']and stage=='normal':
        prior=read(work/'birth_only_evaluation_20261127/streams'/split/'sequence_manifests'/(sequence+'.json'))
        assert streams['CR_reference']['track_sha256']==prior['streams']['BO_paired_q999_c3']['track_sha256']
        assert all(v['detection_packet_sha256']==bank['packet_sha256']for v in streams.values())
    if stage=='intervention':
        normal=read(base/'streams'/split/'sequence_manifests'/(sequence+'.json'));chosen=read(base/'Test_selection_frozen.json')['chosen_case']
        rgb=next(c['id']for c in policies()if c['kind']=='RGB'and c['threshold']==chosen['threshold'])
        assert streams['CR_gate_missing']['track_sha256']==normal['streams'][rgb]['track_sha256']
    assert sha256(fit['model_file'])==fit['model_sha256']and sha256(afit['model_file'])==afit['model_sha256'];registered()
    frozen_json(base/'streams'/label/'sequence_manifests'/(sequence+'.json'),dict(status='COMPLETE_CAUSAL_COVERAGE_RECALL_SEQUENCE',UTC=utc(),
        split=split,stage=stage,sequence=sequence,frames=frames-cut,all_past_frames_consumed=frames,
        validation_window_cut=cut,cases=cases,streams=streams,bank=bank,source_versions=r['source_hashes'],
        quality_model_sha256=fit['model_sha256'],archive_model_sha256=afit['model_sha256'],statistics={k:t.statistics for k,t in trackers.items()},
        original_detector_scores_boxes_order_preserved=True,GT_labels_read=False,future_frame_input=False,past_outputs_rewritten=False,
        replay_seconds=time.monotonic()-started,cost_excludes_neural_inference_images_and_IR_generation=True))


def window_csv47(path,schema,lengths,headline):
    with Path(path).open(newline='')as f:
        reader=csv.DictReader(f);columns=reader.fieldnames;raw=list(reader)
    assert len(columns)==len(set(columns))==276 and len(schema)==47
    aggregate={k for k in columns if k!='seq'and('___'not in k or k.endswith('___AUC'))}
    assert aggregate=={v['source_column']for v in schema}
    assert [v['seq']for v in raw]==[*lengths,'COMBINED']
    rows=[]
    for v in raw:
        assert all(math.isfinite(float(v[k]))for k in columns if k!='seq')
        out={}
        for item in schema:
            value=float(v[item['source_column']])
            if item['unit']=='count':assert value.is_integer();out[item['metric']]=int(value)
            else:out[item['metric']]=value*item['scale_from_raw_detailed']
        expected=sum(lengths.values())if v['seq']=='COMBINED'else lengths[v['seq']]
        assert out['CLR_Frames']==expected and out['CLR_TP']+out['CLR_FN']==out['GT_Dets']
        assert out['IDTP']+out['IDFN']==out['GT_Dets']and out['IDTP']+out['IDFP']==out['Dets']
        assert math.isclose(out['HOTA_TP']+out['HOTA_FN'],out['GT_Dets'],abs_tol=1e-8,rel_tol=1e-12)
        assert math.isclose(out['HOTA_TP']+out['HOTA_FP'],out['Dets'],abs_tol=1e-8,rel_tol=1e-12)
        rows.append(dict(seq=v['seq'],split='forward_Train_validation',**out))
    combined={v['metric']:rows[-1][v['metric']]for v in schema}
    assert len(headline)==39 and all(abs(combined[k]-float(v))<=.000501 for k,v in headline.items())
    return dict(combined47=combined,per_sequence47=rows,official_CSV=str(path),official_CSV_sha256=sha256(path),original_columns=276)


def evaluate(split,stage,case_id):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();plan,protocol,work,base=locations();cases=cases_for(split,stage);case=next(c for c in cases if c['id']==case_id)
    label=split if stage=='normal'else split+'_intervention';lengths=r['specification']['validation_lengths']if split=='trainval'else LENGTHS[split];coverage={}
    for sequence,n in lengths.items():
        receipt=read(base/'streams'/label/'sequence_manifests'/(sequence+'.json'))
        assert receipt['frames']==n and receipt['cases']==cases and not receipt['GT_labels_read']and receipt['source_versions']==r['source_hashes']
        v=receipt['streams'][case_id];assert sha256(v['track_path'])==v['track_sha256'];validate_mot(v['track_path'],n,v['track_rows']);coverage[sequence]=v
    directory=base/'streams'/label/case_id/'track_results';output=base/'evaluation'/label/case_id
    assert not output.exists();output.mkdir(parents=True)
    if split=='trainval':
        from tools.eval_mft26ir import run_trackeval
        headline=run_trackeval(str(directory),str(base/'validation_gt'),str(output),list(lengths),plan['evaluation']['TrackEval_root'],data_split='train')
        schema,counts=official_schema(plan['evaluation']['TrackEval_root'])
        result=dict(metrics39=headline,official_schema=schema,official_family_field_counts=counts,
            **window_csv47(output/'trackeval_output/pedestrian_detailed.csv',schema,lengths,headline))
    else:result=run_official_tracks(directory,output,plan,split)
    registered()
    frozen_json(output/'metrics.json',dict(status='COMPLETE_COVERAGE_RECALL_OFFICIAL47',case=case,frames=sum(lengths.values()),
        coverage=coverage,source_versions=r['source_hashes'],**result,goal_met=False,supported_innovations=0))


def phase(queue,split,stage='normal'):
    plan,protocol,work,base=locations();r=registered();cases=cases_for(split,stage);label=split if stage=='normal'else split+'_intervention'
    lengths=r['specification']['validation_lengths']if split=='trainval'else LENGTHS[split]
    cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))];extra=['--split',split,'--stage',stage]
    queue.jobs(PREFIX+'_'+label.upper()+'_REPLAY',[dict(id=s,command=cmd+['replay',*extra,'--sequence',s])for s in lengths],False,
        lambda j:read(base/'streams'/label/'sequence_manifests'/(j['id']+'.json'))['GT_labels_read']is False)
    queue.jobs(PREFIX+'_'+label.upper()+'_OFFICIAL',[dict(id=c['id'],command=cmd+['evaluate',*extra,'--case',c['id']])for c in cases],False,
        lambda j:read(base/'evaluation'/label/j['id']/'metrics.json')['frames']==sum(lengths.values()))
    return [read(base/'evaluation'/label/c['id']/'metrics.json')for c in cases]


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();plan,protocol,work,base=locations()
    with (base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_PROSPECTIVE_COVERAGE_RECALL',controller_pid=os.getpid(),phases={},seed=0,goal_met=False,supported_innovations=0)
        queue=Queue(work,base,state,plan['data_root'],registered);queue.save();cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
        try:
            queue.jobs(PREFIX+'_ENGINEERING',[dict(id='all6',command=cmd+['engineer'])],False,
                lambda j:read(base/'real_Train_engineering.json')['sequences']==6)
            validation=phase(queue,'trainval');reference=next(v for v in validation if v['case']['kind']=='reference')
            gains={v['case']['id']:v['combined47']['HOTA']-reference['combined47']['HOTA']for v in validation}
            frozen_json(base/'complete_forward_Train_validation.json',dict(status='COMPLETE_ALL10_FORWARD_TRAIN_VALIDATION',UTC=utc(),
                rows=validation,reference_case=reference['case'],HOTA_gains_pp=gains,scored_frames=6814,all_past_frames=27251,
                late_Train_calibrated_parent_thresholds_not_used=True,legacy_encoders_Train_identity_exposure=True,
                Test_read_for_selection=False,goal_met=False,supported_innovations=0))
            dev=phase(queue,'dev');ref=next(v for v in dev if v['case']['kind']=='reference')
            assert ref['combined47']==r['full_parent_Dev47']
            spec=r['specification'];eligible=[v for v in dev if v['case']['kind']in spec['candidate_selection_kinds']and
                gains[v['case']['id']]>=spec['standalone_min_validation_HOTA_gain_pp']and
                v['combined47']['HOTA']>=spec['full_Dev_noninferiority_HOTA']-spec['full_Dev_noninferiority_tolerance_pp']]
            chosen=min(eligible,key=lambda v:(-gains[v['case']['id']],*(-v['combined47'][k]for k in ['HOTA','AssA','IDF1']),v['case']['id']))if eligible else None
            frozen_json(base/'complete_Dev.json',dict(status='COMPLETE_ALL10_FULL_DEV_COVERAGE_RECALL',UTC=utc(),rows=dev,
                selected_case=None if chosen is None else chosen['case'],eligible_ids=[v['case']['id']for v in eligible],
                validation_HOTA_gains_pp=gains,selection_rule=spec['selection'],Test_read_for_selection=False,
                original_frozen_parent_Dev47_exact=True,goal_met=False,supported_innovations=0))
            if chosen is None:
                state.update(status='COMPLETE_COVERAGE_RECALL_NO_ELIGIBLE_VALIDATION_DEV_CASE_NO_NEW_TEST');queue.save();return
            case=chosen['case'];t=case['threshold']
            test=[c for c in policies()if c['kind']in ['reference','unconditional']or c['threshold']==t]
            controls=[dict(case,id='CR_gate_'+mode,intervention=mode)for mode in ['roll','missing']]
            frozen_json(base/'Test_selection_frozen.json',dict(status='FROZEN_BEFORE_NEW_COVERAGE_RECALL_TEST',UTC=utc(),
                chosen_case=case,Test_cases=test,interventions=controls,complete_Dev_sha256=sha256(base/'complete_Dev.json'),
                validation_sha256=sha256(base/'complete_forward_Train_validation.json'),Test_read_for_selection=False,
                exact_Dev_ties_prospectively_allowed=True,controls_never_promoted_from_Test=True))
            intervention=phase(queue,'dev','intervention')
            frozen_json(base/'complete_Dev_interventions.json',dict(status='COMPLETE_COVERAGE_GATE_DEV_INTERVENTIONS',UTC=utc(),rows=intervention,
                scope='New coverage gate only; frozen IR detector and prior quality/archive remain fixed',no_reselection=True))
            testrows=phase(queue,'test');selected=next(v for v in testrows if v['case']==case)
            csvpath=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
            assert sha256(csvpath)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            schema,_=official_schema(plan['evaluation']['TrackEval_root']);baseline=official_csv47(csvpath,schema,'test',read(csvpath.parent.parent/'metrics.json')['trackeval'])
            gain=selected['combined47']['HOTA']-baseline['combined47']['HOTA']
            frozen_json(base/'full_Test_results.json',dict(status='COMPLETE_FROZEN_COVERAGE_RECALL_TEST7',UTC=utc(),rows=testrows,selected_case=case,
                original_RGB_baseline=baseline,HOTA_gain_over_original_RGB_pp=gain,target_plus5_met=gain>=5,target_plus10_met=gain>=10,
                Test_used_for_selection=False,controls_not_promoted_from_Test=True,historical_Test_exposed=True,goal_met=False,supported_innovations=0))
            if gain>=5:
                interventions=phase(queue,'test','intervention')
                frozen_json(base/'full_Test_interventions.json',dict(status='COMPLETE_COVERAGE_GATE_TEST7_INTERVENTIONS',UTC=utc(),rows=interventions,no_reselection=True))
            state.update(status='COMPLETE_COVERAGE_RECALL_FULL_TEST_REQUIRES_ASSESSMENT',HOTA_gain_over_original_RGB_pp=gain);queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_COVERAGE_RECALL_ATTEMPT',error=repr(error));queue.save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run','engineer','replay','evaluate'])
    p.add_argument('--split',choices=['trainval','dev','test']);p.add_argument('--stage',choices=['normal','intervention'],default='normal')
    p.add_argument('--sequence');p.add_argument('--case');a=p.parse_args()
    if a.mode=='register':register()
    elif a.mode=='run':run()
    elif a.mode=='engineer':engineer()
    elif a.mode=='replay':replay(a.split,a.stage,a.sequence)
    else:evaluate(a.split,a.stage,a.case)
