#!/usr/bin/env python3
"""Read-only full official47, calibrated input and frozen choice quality audit."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_mft_proposal_quality as driver
from tools.mft_proposal_quality import ProposalConfirmation,current_features
from tools.mft_proposal_quality_evaluation import policies,matched,quality_anchor,Test_cases,interventions
from tools.native_pyramid_experiment import read,frozen_json
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_evaluation import official_csv47,validate_mot
from tools.export_perceptual_metrics47 import official_schema
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.audit_mft_candidate_risk import require_closed_jobs
from tools.run_mft_candidate_risk import identity_score_hash
from tools.run_mft_candidate_train_bank import atomic_json
from tools.native_pyramid_training_loop import guard_storage


def audit_split(source,split,stage,reg,fit,models,state,base,work,schema,artifacts):
    cases=driver.split_cases(split,stage);assert [r['case']for r in source['rows']]==cases
    label=split if stage=='normal'else split+'_intervention'
    require_closed_jobs(state,[driver.PREFIX+'_'+label.upper()+'_REPLAY',driver.PREFIX+'_'+label.upper()+'_OFFICIAL'],len(LENGTHS[split])+len(cases))
    views,heads,thresholds=models;records={};reconstructed={};count=0
    for sequence,frames in LENGTHS[split].items():
        path=base/'streams'/label/'sequence_manifests'/(sequence+'.json');m=read(path);artifacts[str(path)]=sha256(path)
        assert m['frames']==frames and m['cases']==cases and m['source_versions']==reg['source_hashes']
        assert m['model_sha256']==fit['model_sha256']and not m['GT_labels_read']and not m['future_frame_input']and not m['past_outputs_rewritten']
        assert m['original_detector_packet_unchanged_but_selected_tracker_input_scores_promoted']and m['model_bytes_unchanged']
        for case in cases:
            s=m['streams'][case['id']];assert sha256(s['track_path'])==s['track_sha256']
            validate_mot(s['track_path'],frames,s['track_rows'])
            assert s['frames']==frames and s['detection_packet_sha256']==m['bank']['packet_sha256']
        peer=work/'tracklet_permanence_20261127_r2/cache'/split/(sequence+'.json');meta=read(peer);assert sha256(meta['path'])==meta['sha256']
        with np.load(meta['path'],allow_pickle=False)as data:a={k:data[k].copy()for k in data.files}
        quality=[c for c in cases if c['kind']not in ['reference_native','reference_archive']]
        hashes={c['id']:hashlib.sha256()for c in quality};promotion={c['id']:0 for c in quality};original_hash=hashlib.sha256()
        confirmation=ProposalConfirmation()
        for i,fid in enumerate(a['frame_ids']):
            frame=int(fid);lo,hi=map(int,a['offsets'][i:i+2]);d=a['detections'][lo:hi];rgb=a['rgb'][lo:hi];ir=a['ir'][lo:hi]
            header=np.asarray([frame,len(d)],dtype='<i8').tobytes();original_hash.update(header);original_hash.update(np.asarray(d,dtype='<f4').tobytes())
            streak=confirmation.update(sequence,frame,d);predictions={}
            for c in quality:
                kind=c['kind'];mode=c.get('intervention');required=c['confirmation']
                if kind=='unconditional':prob=np.ones(len(d));threshold=0.
                else:
                    effective='RGB'if kind=='paired'and mode=='missing'else kind
                    key=(effective,mode)
                    if key not in predictions:
                        value=None if mode=='missing'else np.roll(ir,1,axis=0).copy()if mode=='roll'else ir
                        predictions[key]=heads[effective].probabilities(current_features(d,rgb,value,views))
                    prob=predictions[key]
                    condition=c.get('calibration_condition','confirmed'if required else'all')
                    threshold=thresholds[effective][c['quantile']][condition]
                # Independent mask andscore construction;no qualitytracker or
                # promotion helper is used by this reconstruction.
                mask=(d[:,4]>.3)&(d[:,4]<=.7)&(prob>threshold)
                if required:mask&=streak>=3
                changed=d.copy();changed[mask,4]=np.float32(.75)
                assert np.array_equal(changed[:,:4],d[:,:4])and np.array_equal(changed[d[:,4]>.7],d[d[:,4]>.7])
                hashes[c['id']].update(header);hashes[c['id']].update(np.asarray(changed,dtype='<f4').tobytes())
                promotion[c['id']]+=int(mask.sum())
            if frame%1000==0 or frame==frames:
                guard_storage(work);atomic_json(base/'independent_results_audit_progress.json',dict(split=split,stage=stage,sequence=sequence,frame=frame,frames=frames))
        assert original_hash.hexdigest()==m['bank']['packet_sha256']
        for c in quality:
            name=c['id'];assert hashes[name].hexdigest()==m['calibrated_tracker_input_packet_sha256'][name]
            assert promotion[name]==m['statistics'][name]['promoted']
        reconstructed[sequence]=dict(frames=frames,calibrated_input_hashes_exact=True,promotion_counts=promotion)
        if stage=='normal':
            old=read(work/'archive_reasoning_evaluation_20261127/streams'/split/'sequence_manifests'/(sequence+'.json'))
            for own,previous in [('PQ_native_control','AR_native_zero'),('PQ_archive_control','AR_RGB_q999_shape')]:
                if own in m['streams']:assert m['streams'][own]['track_sha256']==old['streams'][previous]['track_sha256']
        else:
            f=read(base/'Test_selection_frozen.json');rgb=matched(quality_anchor(f['chosen_case']),'RGB')['id']
            normal=read(base/'streams'/split/'sequence_manifests'/(sequence+'.json'))
            assert m['streams']['PQ_control_missing']['track_sha256']==normal['streams'][rgb]['track_sha256']
            selected=normal['streams'][f['chosen_case']['id']]['track_path']
            assert identity_score_hash(selected)==identity_score_hash(m['streams']['PQ_control_no_shape']['track_path'])
        records[sequence]=m
    for r in source['rows']:
        assert sha256(r['official_CSV'])==r['official_CSV_sha256'];artifacts[r['official_CSV']]=r['official_CSV_sha256']
        values=official_csv47(r['official_CSV'],schema,split,r['metrics39'])
        assert values['combined47']==r['combined47']and values['per_sequence47']==r['per_sequence47']
        assert r['coverage']=={s:m['streams'][r['case']['id']]for s,m in records.items()}
        assert r['model_sha256']==fit['model_sha256']and r['source_versions']==reg['source_hashes']
        count+=47*(len(LENGTHS[split])+1)
    return dict(split=split,stage=stage,cases=cases,fields=count,reconstructed_tracker_input_packets=reconstructed)


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=driver.registered();_,fit,views,heads,thresholds=driver.fitted();plan,protocol,work,base,train=driver.locations();started=time.monotonic()
    e=read(base/'run_r1.execution.json');assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    te=read(train/'audit_r1.execution.json');ta=read(train/'independent_fit_audit.json')
    assert te['actual_child_exit_code']==0 and te['source_unchanged']and identity(te['identity']['PID'])is None
    assert ta['all6_Train_replayed']and ta['model_sha256']==fit['model_sha256']
    state=read(base/'state.json');require_closed_jobs(state,list(state['phases']),sum(len(j)for j in state['phases'].values()))
    schema,_=official_schema(plan['evaluation']['TrackEval_root']);artifacts={};rows=[]
    for p in [base/'registered.json',base/'state.json',base/'run_r1.execution.json',base/'real_Train_engineering.json',train/'fit_complete.json',train/'independent_fit_audit.json',train/'audit_r1.execution.json']:
        artifacts[str(p)]=sha256(p)
    dev=read(base/'complete_Dev.json');artifacts[str(base/'complete_Dev.json')]=sha256(base/'complete_Dev.json')
    assert dev['Test_read_for_selection']is False
    rows.append(audit_split(dev,'dev','normal',reg,fit,(views,heads,thresholds),state,base,work,schema,artifacts))
    chosen=min(dev['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
    gain=chosen['combined47']['HOTA']-reg['parent_best_Dev_HOTA'];assert dev['chosen_case']==chosen['case']and gain==dev['gain_over_parent_best_Dev_pp']
    fields=sum(r['fields']for r in rows);delta=None
    if gain<=1e-8:
        assert state['status']=='COMPLETE_QUALITY_NO_DEV_GAIN_NO_NEW_TEST'and not(base/'Test_selection_frozen.json').exists()and not(base/'full_Test_results.json').exists()
    else:
        assert state['status']=='COMPLETE_QUALITY_FULL_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT'
        f=read(base/'Test_selection_frozen.json');assert f['chosen_case']==chosen['case']and f['Test_cases']==Test_cases(chosen['case'])
        assert f['Dev_interventions']==interventions(chosen['case'])and not f['Test_read_for_selection']and f['complete_Dev_sha256']==sha256(base/'complete_Dev.json')
        artifacts[str(base/'Test_selection_frozen.json')]=sha256(base/'Test_selection_frozen.json')
        controls=read(base/'complete_Dev_interventions.json');test=read(base/'full_Test_results.json')
        assert controls['no_reselection']and test['Test_used_for_selection']is False
        for name in ['complete_Dev_interventions.json','full_Test_results.json']:artifacts[str(base/name)]=sha256(base/name)
        rows.append(audit_split(controls,'dev','intervention',reg,fit,(views,heads,thresholds),state,base,work,schema,artifacts))
        rows.append(audit_split(test,'test','normal',reg,fit,(views,heads,thresholds),state,base,work,schema,artifacts))
        canonical=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
        assert sha256(canonical)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
        baseline=official_csv47(canonical,schema,'test',read(canonical.parent.parent/'metrics.json')['trackeval']);fields+=376
        selected=next(r for r in test['rows']if r['case']==chosen['case']);delta=selected['combined47']['HOTA']-baseline['combined47']['HOTA']
        assert delta==test['HOTA_gain_over_original_RGB_pp']and abs(baseline['combined47']['HOTA']-29.528970687393972)<1e-10
        if delta>=5:
            ablation=read(base/'full_Test_interventions.json');assert ablation['no_reselection']
            artifacts[str(base/'full_Test_interventions.json')]=sha256(base/'full_Test_interventions.json')
            rows.append(audit_split(ablation,'test','intervention',reg,fit,(views,heads,thresholds),state,base,work,schema,artifacts))
        else:assert not(base/'full_Test_interventions.json').exists()
    fields+=sum(r['fields']for r in rows)-rows[0]['fields']
    assert sha256(fit['model_file'])==fit['model_sha256'];driver.registered()
    frozen_json(base/'independent_results_audit.json',dict(status='COMPLETE_READ_ONLY_ALL47_QUALITY_CALIBRATED_PACKET_AND_CHOICE_AUDIT',UTC=driver.utc(),
        rows=rows,chosen_case=chosen['case'],Dev_gain_pp=gain,HOTA_gain_over_original_RGB_pp=delta,official_fields_reparsed=fields,
        all_calibrated_tracker_input_hashes_and_promotion_counts_rebuilt=True,all_box_order_and_highscore_preservation_checked=True,
        all_required_jobs_and_controller_actual_exit0=True,independent_Train_fit_actual_exit0=True,model_sha256=fit['model_sha256'],
        model_bytes_unchanged=True,artifact_hashes=artifacts,source_hashes={**reg['source_hashes'],'tools/audit_mft_proposal_quality_results.py':sha256(Path(__file__))},
        quality_ROI_IR_controls_not_whole_visual_IR=True,Test_selection_changed=False,new_fits_or_tracking_inference=False,
        cached_quality_and_confirmation_rebuilt=True,historical_Test_previously_exposed=True,seconds=time.monotonic()-started,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_READ_ONLY_ALL47_QUALITY_CALIBRATED_PACKET_AND_CHOICE_AUDIT',fields=fields,HOTA_gain=delta)),flush=True)


if __name__=='__main__':audit()
