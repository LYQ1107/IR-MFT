#!/usr/bin/env python3
"""Fixed four-case full-Dev temporal diagnosis of official offline HOTA matches."""
import argparse
import csv
import datetime
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fit_mft_candidate_risk import locations,require_planning
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256
from tools.fusion_memory_observation import LENGTHS
from tools.audit_mft_native_dev_track_errors import check_official,check_detailed,identity_rows
from tools.mft_dev_identity_timeline import observe_preprocessing,observe_official_timeline,identity_events


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def paths():
    plan,protocol,work,risk=locations()
    return plan,protocol,work,work/'dev_identity_timeline_20261127'


def cases(work):
    source=read(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json')
    canonical=next(r for r in source['rows']if r['id']=='NP_source_canonical_RGB')
    result=[dict(id='IT_original_RGB',method='NP_source_canonical_RGB',
        directory=str(Path(canonical['config']['stream_directory'])/'NP_source_canonical_RGB/track_results'),
        official_CSV=str(ROOT/canonical['official_CSV']['file']),official_CSV_sha256=canonical['official_CSV']['sha256'],
        MOT_sha256={s:canonical['original_coverage'][s]['output_sha256']for s in LENGTHS['dev']},
        combined47=canonical['combined47'],per_sequence47=canonical['per_sequence47'])]
    source=read(work/'pose_continuity_20261127/complete_Dev.json')
    for name in ['PC_native_zero','PC_late_only','PC_shape_anchor_late']:
        r=next(r for r in source['rows']if r['case']['id']==name)
        directories={str(Path(v['track_path']).parent)for v in r['coverage'].values()};assert len(directories)==1
        result.append(dict(id='IT_'+name,method=name,directory=directories.pop(),
            official_CSV=r['official_CSV'],official_CSV_sha256=r['official_CSV_sha256'],
            MOT_sha256={s:r['coverage'][s]['track_sha256']for s in LENGTHS['dev']},
            combined47=r['combined47'],per_sequence47=r['per_sequence47']))
    assert len(result)==4
    return result


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    parent=require_planning();plan,protocol,work,base=paths()
    proof=read(base/'CPU_regressions_r1.json');assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==3
    sources=dict(parent['source_hashes'])
    own=read(work/'candidate_risk_20261127/independent_audit_registered.json')
    sources.update(own['source_hashes'])
    for name,digest in proof['source_hashes'].items():assert sha256(ROOT/name)==digest,name;sources[name]=digest
    normal=cases(work);artifacts={}
    for case in normal:
        csv_path=Path(case['official_CSV']);assert sha256(csv_path)==case['official_CSV_sha256']
        artifacts[str(csv_path)]=sha256(csv_path)
        for sequence in LENGTHS['dev']:
            path=Path(case['directory'])/(sequence+'.txt');assert sha256(path)==case['MOT_sha256'][sequence]
            artifacts[str(path)]=sha256(path)
    for sequence in LENGTHS['dev']:
        gt=Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        assert sha256(gt)==protocol['derived_gt_sha256']['train/'+sequence]
        for p in [gt,gt.parent.parent/'seqinfo.ini']:artifacts[str(p)]=sha256(p)
    vendor=Path(plan['evaluation']['TrackEval_root'])
    for p in [vendor/'trackeval/metrics/hota.py',vendor/'trackeval/datasets/mot_challenge_2d_box.py',
        vendor/'trackeval/datasets/_base_dataset.py',base/'CPU_regressions_r1.json',base/'CPU_regressions_r1.stdout.txt']:
        artifacts[str(p)]=sha256(p)
    guard_storage(work)
    value=dict(status='REGISTERED_POSTHOC_DEV_IDENTITY_TIMELINE_BEFORE_EXECUTION',UTC=utc(),
        source_hashes=sources,artifact_hashes=artifacts,fixed_cases=normal,frames_per_case=4834,
        sequences=LENGTHS['dev'],alpha=.5,all21_GT_IDs_retained=True,new_model_or_policy=False,
        GT_use='Offline descriptive diagnosis only; never fitting weights or routing a tracker',
        timeline_uses_official_global_alignment=True,not_an_online_or_causal_GT_matcher=True,
        previous_prediction_last_output_is_posthoc=True,new_Test_accuracy_parsed=False,Test_GT_read=False,
        historical_Test_previously_exposed=True,new_GPU_work=False,new_seed=False,
        goal_met=False,supported_innovations=0)
    assert not(base/'registered.json').exists();frozen_json(base/'registered.json',value)
    print(json.dumps(dict(status=value['status'],cases=4,sources=len(sources))),flush=True)


def registered():
    require_planning();plan,protocol,work,base=paths();reg=read(base/'registered.json')
    assert reg['fixed_cases']==cases(work)
    for n,h in reg['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in reg['artifact_hashes'].items():assert sha256(n)==h,n
    return reg


def write_csv(path,rows):
    assert rows
    with Path(path).open('x',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base=paths();out=base/'output'
    assert not out.exists();out.mkdir();started=time.monotonic()
    import numpy as np
    if 'float'not in np.__dict__:np.float=float
    if 'int'not in np.__dict__:np.int=int
    vendor=Path(plan['evaluation']['TrackEval_root']);sys.path.insert(0,str(vendor))
    from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    from trackeval.metrics.hota import HOTA
    hota_path=inspect.getsourcefile(HOTA);dataset_path=inspect.getsourcefile(MotChallenge2DBox)
    assert hota_path==str(vendor/'trackeval/metrics/hota.py')and dataset_path==str(vendor/'trackeval/datasets/mot_challenge_2d_box.py')
    all_summaries=[];all_events=[];evaluations=[];private_timelines=[];numeric_fields=0
    for case in reg['fixed_cases']:
        folder=Path(case['directory'])
        with Path(case['official_CSV']).open()as stream:detailed={r['seq']:r for r in csv.DictReader(stream)}
        assert set(detailed)==set(LENGTHS['dev'])|{'COMBINED'}
        dataset=MotChallenge2DBox(dict(TRACKERS_FOLDER=str(folder),TRACKERS_TO_EVAL=['QRTM_V1'],
            GT_FOLDER=str(Path(plan['evaluation']['GT_root'])/'train'),OUTPUT_FOLDER=str(out),BENCHMARK='MFT26IR',
            CLASSES_TO_EVAL=['pedestrian'],GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt',SEQ_INFO=protocol['dev'],
            PRINT_RESULTS=False,PRINT_CONFIG=False,TIME_PROGRESS=False,DO_PREPROC=False,
            use_super_categories=False,USE_PARALLEL=False))
        metric=HOTA();results={}
        for sequence,frames in LENGTHS['dev'].items():
            raw=dataset.get_raw_seq_data(sequence)
            data,original_GT,original_pred=observe_preprocessing(dataset,raw,dataset_path)
            assert data['num_timesteps']==frames
            result,captured,matches=observe_official_timeline(metric,data,hota_path)
            expected=next(r for r in case['per_sequence47']if r['seq']==sequence)
            mean_error=check_official(result,expected,metric,np)
            error,checks=check_detailed(result,detailed[sequence],metric,np);numeric_fields+=checks
            identities=identity_rows(case['id'],sequence,dataset,raw,data,metric,result,captured,np)
            GT_frames={int(g):set()for g in original_GT};prediction_frames={}
            for t in range(frames):
                for g in data['gt_ids'][t]:GT_frames[int(original_GT[g])].add(t+1)
                for p in raw['tracker_ids'][t]:prediction_frames.setdefault(int(p),set()).add(t+1)
            original_matches=[(f,int(original_GT[g]),int(original_pred[p]))for f,g,p in matches]
            summaries,events=identity_events(case['id'],sequence,original_matches,GT_frames,prediction_frames)
            assert len(summaries)==len(identities)==len(original_GT)
            for summary in summaries:
                identity=next(r for r in identities if r['original_GT_ID']==summary['GT_ID'])
                assert summary['GT_frames']==identity['GT_detections']
                assert summary['matched_frames']==identity['matched_GT_detections']
                assert summary['dominant_prediction_matches']==identity['dominant_tracker_matches']
                assert summary['distinct_matched_prediction_IDs']==identity['distinct_matched_tracker_IDs']
            assert sum(s['matched_frames']for s in summaries)==len(original_matches)
            path=out/(case['id']+'_'+sequence+'_private_timeline.npz')
            array=np.asarray(original_matches,np.int32).reshape(-1,3)
            assert array.tolist()==[list(r)for r in original_matches]
            np.savez_compressed(path,matches=array)
            private_timelines.append(dict(case=case['id'],sequence=sequence,path=str(path),
                sha256=sha256(path),bytes=path.stat().st_size,matched_rows=len(array)))
            all_summaries.extend(summaries);all_events.extend(events);results[sequence]=result
            evaluations.append(dict(case=case['id'],sequence=sequence,HOTA=expected['HOTA'],
                full19_alpha_CSV_max_error=error,aggregate_HOTA_family_max_error=mean_error,
                numeric_fields=checks,timeline_matrix_and_original_ID_mapping_exact=True,
                frames=frames,GT_IDs=len(original_GT),matched_rows=len(matches),transitions=len(events)))
            guard_storage(work)
        combined=metric.combine_sequences(results);mean_error=check_official(combined,case['combined47'],metric,np)
        error,checks=check_detailed(combined,detailed['COMBINED'],metric,np);numeric_fields+=checks
        evaluations.append(dict(case=case['id'],sequence='COMBINED',HOTA=case['combined47']['HOTA'],
            full19_alpha_CSV_max_error=error,aggregate_HOTA_family_max_error=mean_error,numeric_fields=checks,frames=4834))
        print(json.dumps(dict(case=case['id'],full_Dev_frames=4834,all19_alpha_official_parity=True,
            HOTA=case['combined47']['HOTA'])),flush=True)
    assert numeric_fields==2772 and len(all_summaries)==84
    write_csv(out/'all21_GT_per_case_summaries.csv',all_summaries)
    if all_events:write_csv(out/'all_matched_ID_transition_events.csv',all_events)
    counts=[]
    for case in reg['fixed_cases']:
        for sequence in LENGTHS['dev']:
            summaries=[r for r in all_summaries if r['case']==case['id']and r['sequence']==sequence]
            events=[r for r in all_events if r['case']==case['id']and r['sequence']==sequence]
            counts.append(dict(case=case['id'],sequence=sequence,GT_IDs=len(summaries),
                matched_ID_transitions=len(events),continuous=sum(r['unmatched_intermediate_frames']==0 for r in events),
                short_gap=sum(1<=r['unmatched_intermediate_frames']<=25 for r in events),
                long_gap=sum(r['unmatched_intermediate_frames']>=26 for r in events),
                previous_ID_still_output=sum(r['previous_prediction_still_output_at_current_frame']for r in events),
                current_ID_established_earlier=sum(not r['current_prediction_newly_output_within_one_frame']for r in events)))
    registered()
    frozen_json(base/'complete.json',dict(status='COMPLETE_FIXED4_FULL_DEV_OFFICIAL_MATCH_TIME_DIAGNOSIS',UTC=utc(),
        cases=4,frames_per_case=4834,GT_rows=84,original_GT_IDs_per_case=21,events=len(all_events),
        alpha=.5,all19_alpha_numeric_fields_checked=numeric_fields,evaluations=evaluations,
        event_counts=counts,identity_summaries=all_summaries,
        summary_CSV=dict(path=str(out/'all21_GT_per_case_summaries.csv'),sha256=sha256(out/'all21_GT_per_case_summaries.csv')),
        events_CSV=None if not all_events else dict(path=str(out/'all_matched_ID_transition_events.csv'),
            sha256=sha256(out/'all_matched_ID_transition_events.csv')),
        private_timeline_artifacts=private_timelines,registration_sha256=sha256(base/'registered.json'),
        seconds=time.monotonic()-started,new_official_accuracy=False,new_tracker_or_model=False,
        original_official47_retained_HOTA_family_recomputed=True,Test_GT_read=False,new_Test_accuracy_parsed=False,
        historical_Test_previously_exposed=True,images_or_models_loaded=False,new_GPU_work=False,
        future_evaluation_alignment_is_not_runtime_input=True,
        event_definition='A different original prediction ID between consecutive alpha0.5 HOTA-matched observations of one GT; not CLEAR IDSW/Frag or an online decision oracle',
        limitations=['Official HOTA uses a whole-sequence alignment matrix. These match times are offline descriptive evaluation, not causal inference or a true online oracle.',
            'A change can reflect geometry/GT-matching ambiguity even if internal tracker IDs do not change. Shape render preserves frame/ID/score/order internally.',
            'Prediction first/last output times and GT visibility are posthoc descriptors, not features available to a tracker.',
            'No per-GT/video policy routing, threshold fitting, checkpoint selection or new Test evaluation occurs here.'],
        goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_FIXED4_FULL_DEV_OFFICIAL_MATCH_TIME_DIAGNOSIS',cases=4,
        frames_per_case=4834,GT_rows=84,events=len(all_events),numeric_fields=numeric_fields,
        seconds=time.monotonic()-started)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run']);args=p.parse_args()
    register()if args.mode=='register'else run()


if __name__=='__main__':main()
