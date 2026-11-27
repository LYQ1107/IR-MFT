#!/usr/bin/env python3
"""Full Train/Dev causal coverage diagnostic; never fit or evaluate Test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.mft_scene_coverage import (KINDS,coverage_features,Dev_analysis_labels,
    uncovered_target,threshold_summary)
from tools.mft_birth_only import BirthOnlyTeacher
from tools.mft_proposal_quality import ProposalConfirmation
from tools.mft_candidate_risk_data import load_arrays,load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.run_mft_proposal_quality import fitted,locations as parent_locations
from tools.fit_mft_candidate_risk import threshold_values
from tools.fit_mft_proposal_quality import utc
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256
from tools.native_pyramid_stream_tracks import mot_line
from tools.audit_mft_tracklet_r2 import bank_audit
from tools.run_mft_tracklet_permanence_r2 import registered as peer_registered
from tools.fusion_memory_observation import LENGTHS

SPEC=ROOT/'configs/perceptual_scene_coverage_diagnostic_20261127.json'
NAMES=['tools/mft_scene_coverage.py','tools/diagnose_mft_scene_coverage.py',
    'tests/test_mft_scene_coverage.py',str(SPEC.relative_to(ROOT))]
TARGETS=('valid_GT','confirmed_opportunities','known_positive','known_background','unknown','uncovered_known_positive','need_ratio')


def locations():
    plan,protocol,work,peer,train=parent_locations()
    return plan,protocol,work,work/'scene_coverage_diagnostic_20261127'


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r,f,views,heads,thresholds=fitted();plan,protocol,work,base=locations();spec=read(SPEC)
    assert f['model_sha256']==spec['foreground_model_sha256']
    assert spec['kinds']==list(KINDS)
    sources={**r['source_hashes'],**{n:sha256(ROOT/n)for n in NAMES}}
    q=dict(train=r['qualified_Train'],dev={})
    peer=work/'tracklet_permanence_20261127_r2';p=peer_registered()
    for sequence,frames in LENGTHS['dev'].items():
        audit=bank_audit(peer,'dev',sequence,p)
        meta=read(peer/'cache/dev'/(sequence+'.json'))
        gt=Path(protocol['derived_gt_root_hint'])/'train'/sequence/'gt/gt.txt'
        expected=protocol['derived_gt_sha256']['train/'+sequence];assert sha256(gt)==expected
        q['dev'][sequence]=dict(bank_path=meta['path'],bank_sha256=meta['sha256'],frames=frames,
            GT_path=str(gt),GT_sha256=expected,bank_audit=audit,logical_split='dev',GT_storage_partition='train')
    proof=read(base/'unit_regressions.json');assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==2
    assert proof['source_hashes']=={n:sources[n]for n in NAMES if not n.endswith('diagnose_mft_scene_coverage.py')}
    prior=work/'birth_only_20261127/complete_Train_diagnosis.json'
    paths=[SPEC,base/'unit_regressions.json',prior,work/'proposal_quality_20261127/fit_complete.json']
    guard_storage(work)
    frozen_json(base/'registered.json',dict(status='REGISTERED_BEFORE_FULL_TRAIN_DEV_COVERAGE_DIAGNOSTIC',UTC=utc(),
        source_hashes=sources,artifact_hashes={str(p):sha256(p)for p in paths},qualified=q,
        model_sha256=f['model_sha256'],specification=spec,Dev_model_fitting=False,Test_read=False,
        historical_Test_exposed=True,feature_producer_has_no_GT_argument=True,seed=0,
        private_scalar_table=True,no_descriptor_or_image_dump=True,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='REGISTERED',sources=len(sources),Train_frames=27251,Dev_frames=4834)),flush=True)


def registered():
    plan,protocol,work,base=locations();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    return r


def load_bank(q,split):
    if split=='train':return load_arrays(q['bank_path'],q['frames'],q['bank_sha256'])
    assert split=='dev' and sha256(q['bank_path'])==q['bank_sha256']
    with np.load(q['bank_path'],allow_pickle=False)as z:a={k:z[k]for k in z.files}
    assert set(a)=={'detections','rgb','ir','offsets','frame_ids'}
    n=len(a['detections']);assert a['detections'].dtype==np.float32 and a['detections'].shape==(n,5)
    assert a['offsets'].dtype==np.int64 and a['offsets'].shape==(q['frames']+1,)
    assert a['offsets'][0]==0 and a['offsets'][-1]==n and (np.diff(a['offsets'])>=0).all()
    np.testing.assert_array_equal(a['frame_ids'],np.arange(1,q['frames']+1))
    for k in ['rgb','ir']:
        assert a[k].dtype==np.float32 and a[k].shape==(n,512)and np.isfinite(a[k]).all()
    return a


def produce(sequence,a,frames,speed,views,heads,progress=None):
    """No GT interface. Produce all frames before the caller opens GT."""
    teacher=BirthOnlyTeacher(speed);shadow=ProposalConfirmation();records=[]
    mot=hashlib.sha256();packet=hashlib.sha256();feature_packet=hashlib.sha256()
    for i in range(frames):
        frame=i+1;lo,hi=map(int,a['offsets'][i:i+2]);d=a['detections'][lo:hi];rgb=a['rgb'][lo:hi];ir=a['ir'][lo:hi]
        before=d.copy();streak=shadow.update(sequence,frame,d);rows=teacher.update(sequence,frame,d,rgb,ir)
        assert teacher.statistics['extra_qualified_births']==0
        unused=[e['proposal_index']for e in teacher.core.last_birth_events if not e['original_allowed']]
        assigned=list(teacher.current_proposal_assignments(frame).values())
        confirmed,values=coverage_features(d,rgb,ir,views,heads,streak,unused,assigned)
        np.testing.assert_array_equal(d,before)
        out=np.asarray([np.r_[v[1][:2],v[1][:2]+v[1][2:]]for v in rows],np.float32).reshape(-1,4)
        deficits=np.asarray([values[k]['deficit']for k in KINDS],np.float64)
        masses=np.asarray([[values[k]['uncovered_mass'],values[k]['measured_mass']]for k in KINDS],np.float64)
        records.append(dict(confirmed=confirmed,outputs=out,deficits=deficits,masses=masses))
        mot.update(''.join(map(mot_line,rows)).encode())
        packet.update(np.asarray([frame,len(d)],dtype='<i8').tobytes()+np.asarray(d,dtype='<f4').tobytes())
        feature_packet.update(np.asarray([frame],dtype='<i8').tobytes()+confirmed.astype('<i8').tobytes()+deficits.astype('<f8').tobytes()+masses.astype('<f8').tobytes())
        if progress is not None and(frame%1000==0 or frame==frames):progress(frame)
    return records,dict(original_zero_gate_MOT_sha256=mot.hexdigest(),original_detector_packet_sha256=packet.hexdigest(),
        GT_free_feature_packet_sha256=feature_packet.hexdigest(),extra_qualified_births=teacher.statistics['extra_qualified_births'])


def engineer():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();_,fit,views,heads,thresholds=fitted();plan,protocol,work,base=locations()
    speed=threshold_values(work)['Train_unknown_speed_upper'];result={}
    for sequence,q in r['qualified']['train'].items():
        a=load_bank(q,'train');normal,receipt=produce(sequence,a,64,speed,views,heads)
        at=int(a['offsets'][32]);a['detections'][at:,:4]+=777
        for cue in ['rgb','ir']:a[cue][at:]*=-1
        future,_=produce(sequence,a,64,speed,views,heads)
        for x,y in zip(normal[:32],future[:32]):
            for k in ['confirmed','outputs','deficits','masses']:np.testing.assert_array_equal(x[k],y[k])
        result[sequence]=dict(frames=64,prefix32_future_suffix_exact=True,GT_not_opened=True,zero_gate_only=True)
    assert sha256(fit['model_file'])==fit['model_sha256'];registered()
    frozen_json(base/'real_Train_engineering.json',dict(status='COMPLETE_ALL6_REAL_TRAIN_CAUSAL_COVERAGE_ENGINEERING',
        UTC=utc(),rows=result,GT_opened=False,new_accuracy=False))
    print(json.dumps(dict(status='ENGINEERING_COMPLETE',sequences=6)),flush=True)


def summarize(x,y,thresholds):
    result={}
    for j,kind in enumerate(KINDS):
        a=x[:,j];b=y[:,-1]
        corr=float(np.corrcoef(a,b)[0,1])if len(a)>1 and a.std()>0 and b.std()>0 else None
        result[kind]=dict(mean_deficit=float(a.mean()),mean_need_ratio=float(b.mean()),Pearson_with_need=corr,
            thresholds={str(t):threshold_summary(a,b,t)for t in thresholds})
    return result


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();_,fit,views,heads,thresholds=fitted();plan,protocol,work,base=locations()
    assert read(base/'engineering_r1.execution.json')['actual_child_exit_code']==0
    speed=threshold_values(work)['Train_unknown_speed_upper'];started=time.monotonic();rows={};private={}
    prior=read(work/'birth_only_20261127/complete_Train_diagnosis.json')
    for split in ['train','dev']:
        rows[split]={}
        for sequence,q in r['qualified'][split].items():
            a=load_bank(q,split)
            def progress(frame):
                guard_storage(work);atomic_json(base/'progress.json',dict(status='GT_FREE_SEQUENCE_INFERENCE',
                    split=split,sequence=sequence,frame=frame,frames=q['frames'],seconds=time.monotonic()-started))
            records,receipt=produce(sequence,a,q['frames'],speed,views,heads,progress)
            # The entire feature stream has completed. Only now open GT.
            gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256']);ys=[];label_packet=hashlib.sha256()
            for i,v in enumerate(records):
                lo,hi=map(int,a['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2]);d=a['detections'][lo:hi];frame=i+1;current=gt[left:right]
                if split=='train':labels=current_detection_labels(d,current,frame=frame,split='train')
                else:labels=Dev_analysis_labels(d,current,frame=frame,split='dev',purpose='read_only_coverage_diagnostic')
                label_packet.update(np.asarray([frame,len(d)],np.int64).tobytes()+d.tobytes()+labels.tobytes())
                y=uncovered_target(d,labels,current,v['outputs'],v['confirmed']);ys.append([y[k]for k in TARGETS])
            x=np.asarray([v['deficits']for v in records]);y=np.asarray(ys,np.float64)
            np.testing.assert_array_equal(x[:,KINDS.index('RGB')],x[:,KINDS.index('IR_missing')])
            if split=='train':
                old=prior['coverage'][sequence]
                assert receipt['original_zero_gate_MOT_sha256']==old['zero_gate_MOT_sha256']
                assert receipt['original_detector_packet_sha256']==old['original_detector_packet_sha256']
                assert label_packet.hexdigest()==old['current_Train_label_packet_sha256']
                for phase,mask in [('early',np.arange(q['frames'])<int(.75*q['frames'])),('late',np.arange(q['frames'])>=int(.75*q['frames']))]:
                    expected=old['by_case'][phase]['BO_unconditional_c3'];s=y[mask].sum(axis=0)
                    for j,key in enumerate(['opportunities','known_positive','known_background','unknown','known_positive_uncovered_by_zero_output_local_IoU05'],1):
                        assert int(s[j])==expected[key],(sequence,phase,key,s[j],expected[key])
            cuts=[0,int(.5*q['frames']),int(.75*q['frames']),q['frames']]
            phase_rows={phase:summarize(x[left:right],y[left:right],r['specification']['thresholds'])for phase,left,right in zip(['first50','middle25','last25'],cuts[:-1],cuts[1:])}
            rows[split][sequence]=dict(frames=q['frames'],proposals=len(a['detections']),inference_receipt=receipt,
                labels_sha256=label_packet.hexdigest(),bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'],
                totals={key:float(y[:,j].sum())for j,key in enumerate(TARGETS[:-1])},
                full=summarize(x,y,r['specification']['thresholds']),phases=phase_rows,
                all_features_completed_before_GT_open=True,Dev_fitting=False,Train_prior_reconstruction_exact=split=='train')
            private[split+'_'+sequence+'_features']=x;private[split+'_'+sequence+'_targets']=y
            print(json.dumps(dict(split=split,sequence=sequence,totals=rows[split][sequence]['totals'],
                count=rows[split][sequence]['full']['count'],paired=rows[split][sequence]['full']['paired'])),flush=True)
            del records,a,gt,x,y
    assert sum(v.nbytes for v in private.values())<r['specification']['private_scalar_frame_table_max_bytes']
    path=base/'private_scalar_frames.npz';assert not path.exists();np.savez_compressed(path,**private)
    summaries={}
    for split in ['train','dev']:
        x=np.concatenate([private[split+'_'+s+'_features']for s in rows[split]])
        y=np.concatenate([private[split+'_'+s+'_targets']for s in rows[split]])
        summaries[split]=dict(pooled=summarize(x,y,r['specification']['thresholds']),equal_video={})
        for k in KINDS:
            metrics={}
            for threshold in r['specification']['thresholds']:
                values=[v['full'][k]['thresholds'][str(threshold)]for v in rows[split].values()]
                def avg(key):
                    valid=[v[key]for v in values if v[key]is not None]
                    return float(np.mean(valid))if valid else None
                metrics[str(threshold)]={key:avg(key)for key in ['precision','recall','selected_mean_target','unselected_mean_target']}
            summaries[split]['equal_video'][k]=metrics
    assert registered()==r and sha256(fit['model_file'])==fit['model_sha256']
    budget=guard_storage(work)
    value=dict(status='COMPLETE_FULL_TRAIN_DEV_SCENE_COVERAGE_DIAGNOSTIC_NO_NEW_TRACKER_OR_TEST',UTC=utc(),
        rows=rows,summaries=summaries,Train_frames=27251,Dev_frames=4834,Test_frames=0,
        model_sha256=fit['model_sha256'],registration_sha256=sha256(base/'registered.json'),
        private_scalar_table=dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size,
            payload_bytes=sum(v.nbytes for v in private.values()),feature_columns=list(KINDS),target_columns=list(TARGETS),not_for_Git=True),
        GT_opens_after_each_complete_sequence_feature_stream=True,original_outputs_unchanged=True,
        confidence_is_not_calibrated_count=True,proxy_is_not_HOTA=True,Dev_fitting=False,new_GPU_work=False,
        no_selection_or_Test_authorized_by_this_diagnostic=True,seconds=time.monotonic()-started,
        budget=budget,goal_met=False,supported_innovations=0)
    frozen_json(base/'complete_Train_Dev.json',value)
    print(json.dumps(dict(status=value['status'],seconds=value['seconds'],scalar_bytes=path.stat().st_size,budget=budget)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','engineer','run']);args=p.parse_args()
    {'register':register,'engineer':engineer,'run':run}[args.mode]()
