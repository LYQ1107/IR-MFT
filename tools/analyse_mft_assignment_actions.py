"""Post-producer Train labels for GT-free assignment-action evidence."""
from pathlib import Path
from collections import defaultdict
import numpy as np

from tools.run_mft_assignment_action_census import registered,context
from tools.mft_assignment_actions import STAGES,KINDS
from tools.mft_protected_peripheral_experiment import qualified_input,registered as parent_registered
from tools.mft_candidate_risk_data import load_GT
from tools.mft_candidate_risk_labels import current_detection_labels,PastLabelHistory
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_pseudo_ir_comparison import sha256
from tools.fit_mft_proposal_quality import utc


def count():return dict(actions=0,known=0,unknown=0,beneficial=0,harmful=0,neutral=0,correct_pair_delta_sum=0,beneficial_runs=0)


def analyse():
    r=registered();plan,protocol,work,parent,base=context();ready=read(base/'source_ready.json');assert ready['frames']==27251
    state=read(base/'state.json');sources=state['phases']['ACTION_CENSUS_SOURCE']
    assert len(sources)==6 and all(j['status']=='COMPLETE'and j['exit_code']==0 and identity(j['pid'])is None for j in sources.values())
    total={};all_rows={};eligible_videos=0
    for sequence,q in r['qualified_Train'].items():
        meta_path=base/'source_metadata'/(sequence+'.json');meta=read(meta_path)
        assert sha256(meta_path)==ready['rows'][sequence]['metadata_sha256']and sha256(meta['path'])==meta['sha256']and not meta['GT_labels_parsed']
        with np.load(meta['path'],allow_pickle=False)as z:a={k:z[k]for k in z.files}
        qq,d=qualified_input('train',sequence,parent_registered(),all_features=False);assert qq==q
        gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256']);history=PastLabelHistory()
        action_offsets=np.searchsorted(a['action_frame'],np.arange(1,q['frames']+2));match_offsets=np.searchsorted(a['matches'][:,0],np.arange(1,q['frames']+2))
        labels_known=np.zeros(len(a['action_frame']),bool);deltas=np.full(len(labels_known),-9,np.int8);phases=np.empty(len(labels_known),np.uint8)
        rows={};matched={};last_beneficial={};video_beneficial=0
        for frame in range(1,q['frames']+1):
            lo,hi=map(int,d['offsets'][frame-1:frame+1]);g=gt[bounds[frame-1]:bounds[frame]]
            current=current_detection_labels(d['detections'][lo:hi],g,frame=frame,split='train')
            phase='fit50'if frame<=q['frames']//2 else'calibrate25'if frame<=3*q['frames']//4 else'validate25';phase_code=['fit50','calibrate25','validate25'].index(phase)
            ml,mh=map(int,match_offsets[frame-1:frame+1])
            for f,stage,track,proposal,template in a['matches'][ml:mh]:
                key=phase+'_'+STAGES[int(stage)];c=matched.setdefault(key,dict(matches=0,known=0,correct=0,wrong=0,known_template_wrong=0));c['matches']+=1
                prior=history.prior(int(track),frame);label=int(current[proposal])
                if prior is not None and label>=0:
                    c['known']+=1;c['correct']+=int(prior==label);c['wrong']+=int(prior!=label);c['known_template_wrong']+=int(prior!=label and template==1)
            al,ah=map(int,action_offsets[frame-1:frame+1]);seen_runs=set()
            for i in range(al,ah):
                kind,stage=KINDS[int(a['action_kind'][i])],STAGES[int(a['action_stage'][i])];k=int(a['action_count'][i]);old=a['before'][i,:k];new=a['after'][i,:k]
                tracks={int(t)for t,p in np.concatenate([old,new])};props={int(p)for t,p in np.concatenate([old,new])}
                priors={t:history.prior(t,frame)for t in tracks};known=all(x is not None for x in priors.values())and all(current[p]>=0 for p in props)
                key=phase+'_'+stage+'_'+kind;c=rows.setdefault(key,count());c['actions']+=1;phases[i]=phase_code
                if not known:c['unknown']+=1;continue
                old_correct=sum(priors[int(t)]==int(current[p])for t,p in old);new_correct=sum(priors[int(t)]==int(current[p])for t,p in new);delta=new_correct-old_correct
                assert -2<=delta<=2;labels_known[i]=True;deltas[i]=delta;c['known']+=1;c['correct_pair_delta_sum']+=delta
                outcome='beneficial'if delta>0 else'harmful'if delta<0 else'neutral';c[outcome]+=1
                if delta>0:
                    video_beneficial+=1
                    run=(phase,stage,kind,tuple(sorted(tracks)),tuple(sorted(int(current[p])for p in props)))
                    if run not in seen_runs:
                        if last_beneficial.get(run)!=frame-1:c['beneficial_runs']+=1
                        seen_runs.add(run)
            for run in seen_runs:last_beneficial[run]=frame
            ll,hh=map(int,a['assign_offsets'][frame-1:frame+1]);assigned=a['assignments'][ll:hh]
            history.commit_after_update(frame,{int(t):int(current[p])for t,p in assigned})
        label_path=base/'private_labels'/(sequence+'.npz');label_path.parent.mkdir(parents=True,exist_ok=True)
        with label_path.open('xb')as stream:np.savez_compressed(stream,known=labels_known,delta=deltas,phase=phases)
        assert sha256(meta['path'])==meta['sha256']
        for key,c in rows.items():
            assert c['actions']==c['known']+c['unknown']and c['known']==c['beneficial']+c['harmful']+c['neutral']
            agg=total.setdefault(key,count())
            for k,v in c.items():agg[k]+=v
        if video_beneficial:eligible_videos+=1
        all_rows[sequence]=dict(frames=q['frames'],actions=len(labels_known),source_sha256=meta['sha256'],GT_sha256=q['GT_sha256'],
            phase_stage_kind=rows,original_matched_pair_counts=matched,private_labels_path=str(label_path),private_labels_sha256=sha256(label_path),
            known=int(labels_known.sum()),unknown=int((~labels_known).sum()),beneficial_actions=video_beneficial)
        registered();print(sequence,'Train-only action labels complete',flush=True)
    gate=r['specification']['opportunity_gate_for_later_fit'];runs=sum(c['beneficial_runs']for c in total.values());late=sum(c['beneficial_runs']for k,c in total.items()if k.startswith('validate25_'))
    qualifies=runs>=gate['known_beneficial_runs_all_Train_minimum']and late>=gate['known_beneficial_runs_last25_minimum']and eligible_videos>=gate['Train_videos_with_known_beneficial_action_minimum']
    frozen_json(base/'complete_Train_census.json',dict(status='COMPLETE_ALL6_TRAIN_STAGE_ACTION_OPPORTUNITY_CENSUS',UTC=utc(),frames=27251,rows=all_rows,totals=total,
        known_beneficial_runs=runs,known_last25_beneficial_runs=late,videos_with_known_beneficial_action=eligible_videos,
        later_fit_opportunity_gate_passed=qualifies,source_producers_all_actual_exit0=True,source_ready_sha256=sha256(base/'source_ready.json'),
        first50_fit_middle25_calibration_last25_validation=True,no_models_fitted=True,no_tracker_outputs_modified=True,
        no_HOTA_prediction=True,unknown_not_background=True,Test_or_Dev_labels_read=False,
        scope='Past-purity/current-IoU local identity proxy and phase-specific beneficial runs, not independent episodes, HOTA or an oracle tracker.',
        goal_met=False,supported_innovations=0))
