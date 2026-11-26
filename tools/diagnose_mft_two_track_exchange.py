#!/usr/bin/env python3
"""Train-only evidence for sparse two-track exchanges; never alter the teacher."""
import argparse
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
from tools.fit_mft_candidate_risk import locations,threshold_values,require_planning
from tools.run_mft_candidate_risk import require_fitted
from tools.mft_candidate_risk_data import qualify_Train,load_arrays,load_GT
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_candidate_risk_features import candidate_features,_margins
from tools.mft_candidate_risk_labels import current_detection_labels,PastLabelHistory
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_stream_tracks import mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256

SPEC=ROOT/'configs/perceptual_two_track_exchange_20261127.json'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cycle_scores(base,eligible,original_pairs,probabilities):
    """Closed2x2 exchanges only; all eligible row/column competitors count."""
    base=np.asarray(base,np.float64);eligible=np.asarray(eligible,bool);p=np.asarray(probabilities,np.float64)
    assert base.shape==eligible.shape==p.shape and np.isfinite(base).all()and np.isfinite(p).all()
    assert ((p>=0)&(p<=1)).all()
    row_margin,col_margin=_margins(p,eligible);result=[]
    for left,(i,a)in enumerate(original_pairs):
        i,a=int(i),int(a)
        for j,b in original_pairs[left+1:]:
            j,b=int(j),int(b)
            assert i!=j and a!=b
            if not(eligible[i,a]and eligible[i,b]and eligible[j,a]and eligible[j,b]):continue
            if base[i,b]>.9 or base[j,a]>.9:continue
            score=min(row_margin[i,b],col_margin[i,b],row_margin[j,a],col_margin[j,a])
            result.append(dict(rows=(i,j),columns=(a,b),score=float(score)))
    return result


class PastWindow:
    def __init__(self,length=3):
        assert type(length)is int and length>0;self.length=length;self.memory={}

    def add(self,key,frame,score):
        assert type(frame)is int and frame>0 and np.isfinite(score)
        old=self.memory.get(key,[])
        if old and old[-1][0]!=frame-1:old=[]
        values=(old+[(frame,float(score))])[-self.length:];self.memory[key]=values
        return min(v for f,v in values)if len(values)==self.length else None

    def prune(self,frame):self.memory={k:v for k,v in self.memory.items()if v[-1][0]>=frame}


def paths():
    plan,protocol,work,parent=locations();return plan,protocol,work,parent,work/'two_track_exchange_20261127'


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    main,fit,models,thresholds=require_fitted();plan,protocol,work,parent,base=paths();q=qualify_Train(work,protocol)
    e=read(parent/'controller.execution.json');ae=read(parent/'independent_audit_r1.execution.json')
    assert e['actual_child_exit_code']==ae['actual_child_exit_code']==0 and e['source_unchanged']and ae['source_unchanged']
    assert identity(e['identity']['PID'])is None and identity(ae['identity']['PID'])is None
    proof=read(base/'CPU_regressions_r1.json');assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==3
    sources=dict(main['source_hashes']);sources.update(read(parent/'independent_audit_registered.json')['source_hashes'])
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n;sources[n]=h
    sources[str(SPEC.relative_to(ROOT))]=sha256(SPEC)
    files=[SPEC,base/'CPU_regressions_r1.json',base/'CPU_regressions_r1.stdout.txt',parent/'complete_Dev.json',
        parent/'independent_Train_fit_audit.json',parent/'independent_audit_complete.json',parent/'fit_complete.json']
    v=dict(status='REGISTERED_ALL6_TRAIN_TWO_TRACK_EXCHANGE_DIAGNOSIS_BEFORE_COLLECTION',UTC=utc(),
        source_hashes=sources,artifact_hashes={str(p):sha256(p)for p in files},qualified_Train=q,
        specification_sha256=sha256(SPEC),parent_model_sha256=fit['model_sha256'],models=['RGB','paired'],
        Train_frames=27251,Dev_or_Test_labels_for_collection=0,new_optimizer_steps=0,new_model_fits=0,
        diagnostic_only_no_modified_tracker=True,teacher_association_cost_unchanged=True,
        new_accuracy=False,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0)
    assert not(base/'registered.json').exists();frozen_json(base/'registered.json',v)
    print(json.dumps(dict(status=v['status'],sources=len(sources),Train_frames=27251)),flush=True)


def registered():
    require_planning();plan,protocol,work,parent,base=paths();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    return r


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();main,fit,models,thresholds=require_fitted();plan,protocol,work,parent,base=paths();started=time.monotonic()
    assert sha256(fit['model_file'])==reg['parent_model_sha256']
    reference=read(parent/'Train_feature_statistics.json')['full_teacher_references']
    buckets={k:{phase:{label:[]for label in ['beneficial','harmful','neutral']}for phase in ['early','late']}for k in models}
    sizes={k:0 for k in models};coverage={};speed=threshold_values(work)['Train_unknown_speed_upper']
    for sequence,q in reg['qualified_Train'].items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        teacher=CandidateTeacher(speed);teacher.reset(sequence);history=PastLabelHistory();current=[None]
        windows={k:PastWindow(3)for k in models}
        stats=dict(frames=0,proposals=0,eligible_cycles=0,known_cycles=0,unknown_cycles=0,
            beneficial_cycles=0,harmful_cycles=0,neutral_cycles=0,three_frame_windows={k:dict(beneficial=0,harmful=0,neutral=0,unknown=0)for k in models})
        signature=hashlib.sha256()
        def observe(tracks,detections,cost,frame,stage):
            if stage!='primary'or not cost.size:return cost
            x,eligible=candidate_features(tracks,detections,cost,frame,speed)
            pairs=matching.linear_assignment(cost,thresh=.9)[0];by_kind={}
            for k,m in models.items():
                p=np.zeros(cost.shape,np.float64)
                if eligible.any():p[eligible]=m.probabilities(x[eligible])
                by_kind[k]=cycle_scores(cost,eligible,pairs,p)
            assert [v['rows']+v['columns']for v in by_kind['RGB']]==[v['rows']+v['columns']for v in by_kind['paired']]
            for index,c in enumerate(by_kind['RGB']):
                i,j=c['rows'];a,b=c['columns'];key=tuple(sorted((int(tracks[i].track_id),int(tracks[j].track_id))))
                prior=[history.prior(int(tracks[t].track_id),frame)for t in [i,j]]
                labels=[int(current[0][detections[t].observation['proposal_index']])for t in [a,b]]
                known=all(v is not None for v in prior)and all(v>=0 for v in labels)
                stats['eligible_cycles']+=1
                if known:
                    old=int(prior[0]==labels[0])+int(prior[1]==labels[1]);new=int(prior[0]==labels[1])+int(prior[1]==labels[0])
                    target='beneficial'if new>old else'harmful'if new<old else'neutral'
                    stats['known_cycles']+=1;stats[target+'_cycles']+=1
                else:target='unknown';stats['unknown_cycles']+=1
                scores=[]
                for k in models:
                    score=by_kind[k][index]['score'];past=windows[k].add(key,frame,score);scores.append(score)
                    if past is not None:
                        stats['three_frame_windows'][k][target]+=1
                        if known:
                            phase='early'if frame<=int(.75*q['frames'])else'late'
                            sizes[k]+=8;assert sizes[k]<=67108864,'Preserve failure; never truncate proposal windows'
                            buckets[k][phase][target].append(past)
                signature.update(np.asarray([frame,*key,int(known)],np.int64).tobytes()+np.asarray(scores,np.float64).tobytes())
            for window in windows.values():window.prune(frame)
            return cost
        teacher.core.association_hook=observe;digest=hashlib.sha256()
        for index,f in enumerate(arrays['frame_ids']):
            frame=int(f);lo,hi=map(int,arrays['offsets'][index:index+2]);left,right=map(int,bounds[index:index+2]);d=arrays['detections'][lo:hi]
            current[0]=current_detection_labels(d,gt[left:right],frame=frame,split='train')
            rows=teacher.update(sequence,frame,d,arrays['rgb'][lo:hi],arrays['ir'][lo:hi])
            assignments=teacher.current_proposal_assignments(frame)
            history.commit_after_update(frame,{i:int(current[0][j])for i,j in assignments.items()});current[0]=None
            for row in rows:digest.update(mot_line(row).encode())
            stats['frames']+=1;stats['proposals']+=len(d)
            if frame%500==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'progress.json',dict(status='ALL6_TRAIN_TWO_TRACK_EXCHANGE_DIAGNOSIS',UTC=utc(),
                    sequence=sequence,frame=frame,frames=q['frames'],seconds=time.monotonic()-started,
                    score_RAM_bytes=sizes,new_optimizer_steps=0,new_accuracy=False))
        assert digest.hexdigest()==reference[sequence]['original_teacher_MOT_sha256']
        assert stats['frames']==q['frames']and stats['proposals']==reference[sequence]['proposal_rows']
        coverage[sequence]=dict(**stats,original_teacher_MOT_sha256=digest.hexdigest(),teacher_MOT_exact=True,
            causal_graph_score_signature=signature.hexdigest(),bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'])
        print(json.dumps(dict(sequence=sequence,frames=q['frames'],statistics=stats)),flush=True)
    summaries={}
    for k,phases in buckets.items():
        summaries[k]={};late=phases['late'];wrong=np.asarray(late['harmful']+late['neutral'],np.float64)
        thresholds={name:max(.05,float(np.quantile(wrong,q)))for name,q in [('q99',.99),('q999',.999)]}if len(wrong)else None
        for phase,labels in phases.items():
            summaries[k][phase]={}
            for label,scores in labels.items():
                a=np.asarray(scores,np.float64)
                summaries[k][phase][label]=dict(windows=len(a),positive_margin_windows=int((a>=.05).sum()),
                    min=float(a.min())if len(a)else None,max=float(a.max())if len(a)else None,
                    median=float(np.median(a))if len(a)else None,
                    passing_late_wrong_thresholds=None if thresholds is None else{n:int((a>t).sum())for n,t in thresholds.items()})
        summaries[k]['late_nonbeneficial_three_frame_window_thresholds']=thresholds
        summaries[k]['late_nonbeneficial_window_count']=len(wrong)
    assert sha256(fit['model_file'])==reg['parent_model_sha256'];registered()
    frozen_json(base/'complete_Train.json',dict(status='COMPLETE_ALL6_TRAIN_TWO_TRACK_EXCHANGE_DIAGNOSIS_NO_NEW_TRACKER',UTC=utc(),
        coverage=coverage,models=summaries,frames=27251,sequences=6,source_registration_sha256=sha256(base/'registered.json'),
        parent_model_sha256=reg['parent_model_sha256'],score_RAM_bytes=sizes,all_teacher_MOT_exact=True,
        model_bytes_unchanged=True,new_optimizer_steps=0,new_model_fits=0,Dev_rows=0,Test_rows=0,
        new_GPU_work=False,new_tracking_accuracy=False,seconds=time.monotonic()-started,
        not_a_false_link_probability_bound=True,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_ALL6_TRAIN_TWO_TRACK_EXCHANGE_DIAGNOSIS_NO_NEW_TRACKER',models=summaries)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run']);a=p.parse_args()
    register()if a.mode=='register'else run()


if __name__=='__main__':main()
