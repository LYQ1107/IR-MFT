#!/usr/bin/env python3
"""Retrospective motion error on actual native histories, after all6 byte parities."""
import argparse,gzip,json,os
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from cfp.provenance import sha256,snapshot_sources,write_json
from cfp.observable_occlusion import current_overlap
from cfp.joint_set_supervision import consistent_past_identity
from yolox.tracker import matching


def describe(rows):
    if not rows:return dict(supported=0)
    data=np.asarray(rows,np.float64)
    return dict(supported=len(data),KF_center_error_mean=float(data[:,0].mean()),KF_center_error_median=float(np.median(data[:,0])),KF_center_error_p95=float(np.quantile(data[:,0],.95)),last_two_measurement_CV_center_error_mean=float(data[:,1].mean()),KF_minus_CV_mean=float((data[:,0]-data[:,1]).mean()),CV_better_fraction=float((data[:,1]<data[:,0]).mean()),past_velocity_change_median=float(np.median(data[:,2])),mean_current_gap=float(data[:,3].mean()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--observation',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU-only aftercompletion diagnosis required')
    m=json.loads((a.observation/'manifest.json').read_text());c=m['config']
    if m['status']!='COMPLETE_GT_FREE_NATIVE_MOTION_OBSERVATION' or not m['all_six_native_prediction_bytes_equal'] or m['annotation_labels_loaded_online'] or set(m['videos'])!=set(c['sequences']):raise ValueError('All6original native predictionbyteparities beforeanyGT diagnosis required')
    for video,e in m['videos'].items():
        if not e['baseline_byte_equal'] or sha256(a.observation/(video+'.txt'))!=e['prediction_sha256'] or sha256(a.observation/(video+'.motion.jsonl.gz'))!=e['motion_trace_sha256']:raise ValueError('Actual read-only observation/predictions drift')
    # This import and every GT label load occur only after ALL six online
    # prediction traces are already finished and their byte equality checked.
    from cfp.recovery_supervision import physical_bank
    report=dict(status='COMPLETE_RETROSPECTIVE_NATIVE_MOTION_ERROR_DIAGNOSIS',observation=str(a.observation),observation_manifest_sha256=sha256(a.observation/'manifest.json'),videos={},sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,'cfp/recovery_supervision.py','cfp/observable_occlusion.py','cfp/fish_geometry.py','cfp/joint_set_supervision.py']),scope='RetrospectivecurrentGTtarget afterall6GT-free predictionparities; neveranonlineinput. RequireallrecentacceptedpastproposalIDsknown/samephysicalID/min3; rejectunknown/mixed/short/missingreferences andduplicatecurrentGT. NativeKFstatecanstillcontainolderpollution despitecleanrecenthistory, detectionmeasurementscanbenoisy. CompareactualpredictedKFcenter andlast2acceptedmeasurement CV, scaledbycurrentGTboxdiagonal. Pastvelocitychange ispastmeasurementproxy, nottruefishturn/occlusion label. No causalclaimthat learnednonlinearitywouldfixMOT, no teacherforcedKF/biologicalidentity/globalexternalgeneralization claim. Fit-only4videos namedseparately; inner/outer summaries diagnosticnotmodel/threshold selection.')
    fit_rows=[]
    for video in c['sequences']:
        physical=physical_bank(video,'train');rows=[];grouped=defaultdict(list);counts=Counter();sampled=[]
        with gzip.open(a.observation/(video+'.motion.jsonl.gz'),'rt') as stream:
            for entry in map(json.loads,stream):
                frame=entry['frame'];indices=np.asarray(entry['proposals'],np.int64);boxes=np.asarray(entry['boxes'],np.float64).reshape(-1,4)
                if len(indices) and (physical['frame_of_proposal'][indices]!=frame).any():raise ValueError('Currentnativeproposal trace timestamp drift')
                overlap=current_overlap(boxes);gt=defaultdict(list)
                for identity,box in physical['gt'][frame]:gt[int(identity)].append(np.asarray(box,np.float64))
                for track in entry['tracks']:
                    counts['native_track_rows']+=1;history=track['history']
                    if any(index is None for time,index in history):counts['unknown_past_reference']+=1;continue
                    old=np.asarray([index for time,index in history],np.int64);times=np.asarray([time for time,index in history],np.int64)
                    if len(old) and ((times>=frame).any() or not np.array_equal(physical['frame_of_proposal'][old],times)):raise ValueError('Actualpastnative trace mismatch')
                    identity=consistent_past_identity(old,physical['labels'],3)
                    if identity<0:counts['mixed_unknown_or_short_recent_history']+=1;continue
                    counts['known_consistent_recent_history']+=1
                    if len(gt[identity])!=1:counts['current_GT_absent_or_nonunique']+=1;continue
                    target=gt[identity][0];width=target[2:]-target[:2];scale=float(np.linalg.norm(width))
                    if not scale>0:counts['invalid_current_GT_shape']+=1;continue
                    if len(np.unique(times))!=len(times):raise ValueError('Repeatedacceptedmeasurement timestamp cannotdefinepastvelocity')
                    past=physical['boxes'][old];centers=(past[:,:2]+past[:,2:])/2;target_center=(target[:2]+target[2:])/2;velocity=(centers[-1]-centers[-2])/(times[-1]-times[-2]);old_velocity=(centers[-2]-centers[-3])/(times[-2]-times[-3]);prediction=centers[-1]+velocity*(frame-times[-1]);kf=np.asarray(track['predicted_xyah'])[:2]
                    values=[float(np.linalg.norm(kf-target_center)/scale),float(np.linalg.norm(prediction-target_center)/scale),float(np.linalg.norm(velocity-old_velocity)/scale),int(frame-times[-1])];rows.append(values)
                    correct=np.flatnonzero(physical['labels'][indices]==identity)
                    if len(correct):
                        quality_proxy=float(overlap[correct].max());group='no_overlap' if quality_proxy==0 else 'overlap_below_half' if quality_proxy<.5 else 'overlap_at_least_half';raw=matching.iou_distance([np.asarray(track['predicted_box'])],boxes[correct]);counts['true_candidate_present']+=1;counts['all_correct_candidates_outside_native_appearance_graph']+=int((raw[0]>.9).all())
                    else:group='no_current_known_target_proposal';quality_proxy=None;counts[group]+=1
                    grouped[group].append(values);grouped['recent_gap_1' if values[3]==1 else 'recent_gap_gt1'].append(values)
                    if len(sampled)<8:sampled.append(dict(frame=frame,track_id=track['track_id'],physical_ID_offline=identity,KF_error=values[0],CV_error=values[1],past_velocity_change=values[2],latest_actual_measurement_gap=values[3],current_overlap_proxy=quality_proxy))
        if not rows:raise ValueError('No valid actualnativehistories; cannotmanufacturemotionevidence')
        if video in c['fit_evidence_videos']:fit_rows.extend(rows)
        report['videos'][video]=dict(role='fit_evidence' if video in c['fit_evidence_videos'] else 'inner_diagnostic' if video==c['inner_evidence_video'] else 'outer_diagnostic',counts=dict(counts),all_supported=describe(rows),groups={k:describe(v) for k,v in grouped.items()},examples_offline=sampled,annotation_sha256=physical['annotation_sha256'],detection_sha256=physical['detection_sha256'],source_trace_sha256=m['videos'][video]['motion_trace_sha256'])
    report['fit_only_pooled_description']=describe(fit_rows);report['fit_evidence_videos']=c['fit_evidence_videos'];write_json(a.output,report);print(json.dumps({v:e['all_supported'] for v,e in report['videos'].items()}))

if __name__=='__main__':main()
