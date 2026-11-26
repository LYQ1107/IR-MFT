"""Observe offline official HOTA match times; never supply labels to a tracker."""
import inspect
import sys
import numpy as np


def observe_preprocessing(dataset,raw,dataset_path):
    """Read real preprocessor ID maps, without reimplementing ignore matching."""
    maps=[]
    def profile(frame,event,arg):
        if(event=='return'and frame.f_code.co_filename==dataset_path
                and frame.f_code.co_name=='get_preprocessed_seq_data'):
            values=frame.f_locals
            maps.append((np.asarray(values['unique_gt_ids'],np.int64).copy(),
                np.asarray(values['unique_tracker_ids'],np.int64).copy()))
    previous=sys.getprofile()
    try:
        sys.setprofile(profile);data=dataset.get_preprocessed_seq_data(raw,'pedestrian')
    finally:sys.setprofile(previous)
    assert len(maps)==1
    gt_ids,pred_ids=maps[0]
    assert len(gt_ids)==data['num_gt_ids']and len(pred_ids)==data['num_tracker_ids']
    assert len(np.unique(gt_ids))==len(gt_ids)and len(np.unique(pred_ids))==len(pred_ids)
    for t in range(data['num_timesteps']):
        for p,box,score in zip(data['tracker_ids'][t],data['tracker_dets'][t],data['tracker_confidences'][t]):
            indices=np.flatnonzero(raw['tracker_ids'][t]==pred_ids[p]);assert len(indices)==1
            j=int(indices[0]);np.testing.assert_array_equal(box,raw['tracker_dets'][t][j])
            assert score==raw['tracker_confidences'][t][j]
    return data,gt_ids,pred_ids


def observe_official_timeline(metric,data,hota_path):
    lines,start=inspect.getsourcelines(type(metric))
    targets=[start+i for i,line in enumerate(lines)if "res['HOTA_TP'][a] += num_matches"in line]
    assert len(targets)==1,'Bind the observer to the inspected official accumulation line'
    target=targets[0];rows=[];terminal=[]
    index=int(np.flatnonzero(np.isclose(metric.array_labels,.5))[0])
    def trace(frame,event,arg):
        if frame.f_code.co_filename!=hota_path or frame.f_code.co_name!='eval_sequence':return None
        if event=='line'and frame.f_lineno==target:
            values=frame.f_locals
            if int(values['a'])==index:
                gt=values['gt_ids_t'][values['alpha_match_rows']]
                pred=values['tracker_ids_t'][values['alpha_match_cols']]
                rows.extend((int(values['t'])+1,int(g),int(p))for g,p in zip(gt,pred))
        elif event=='return':
            values=frame.f_locals
            if 'matches_counts'in values:
                terminal.append({k:values[k]for k in ['matches_counts','gt_id_count','tracker_id_count']})
        return trace
    previous=sys.gettrace()
    try:
        sys.settrace(trace);result=metric.eval_sequence(data)
    finally:sys.settrace(previous)
    assert len(terminal)==1
    captured=terminal[0];matches=captured['matches_counts'][index]
    rebuilt=np.zeros_like(matches)
    for frame,g,p in rows:rebuilt[g,p]+=1
    np.testing.assert_array_equal(rebuilt,matches)
    assert len(rows)==result['HOTA_TP'][index]
    per_frame={}
    for r in rows:per_frame.setdefault(r[0],[]).append(r)
    for own in per_frame.values():
        assert len(own)==len({r[1]for r in own})==len({r[2]for r in own})
    return result,captured,rows


def identity_events(case,sequence,rows,GT_frames,prediction_frames):
    """Transitions between consecutive matched observations, not CLEAR IDSW."""
    assert all(type(g)is int and g>0 for g in GT_frames)
    pred_first={p:min(frames)for p,frames in prediction_frames.items()}
    pred_last={p:max(frames)for p,frames in prediction_frames.items()}
    frame_pred={}
    for p,frames in prediction_frames.items():
        for f in frames:frame_pred.setdefault(f,set()).add(p)
    by_GT={g:[]for g in GT_frames}
    for f,g,p in rows:
        assert g in by_GT and f in GT_frames[g]and f in prediction_frames[p]
        by_GT[g].append((f,p))
    events=[];summaries=[]
    for g,frames in sorted(GT_frames.items()):
        values=by_GT[g];assert values==sorted(values)
        assert len(values)==len(set(f for f,p in values))
        counts={};own=[]
        for f,p in values:counts[p]=counts.get(p,0)+1
        for (previous_frame,previous),(frame,current)in zip(values,values[1:]):
            if previous==current:continue
            gap=frame-previous_frame-1
            GT_present=sum(f in frames for f in range(previous_frame+1,frame))
            old_visible=previous in frame_pred.get(frame,set())
            current_birth=pred_first[current]
            event=dict(case=case,sequence=sequence,GT_ID=g,previous_match_frame=previous_frame,
                current_match_frame=frame,previous_prediction_ID=previous,current_prediction_ID=current,
                unmatched_intermediate_frames=gap,GT_present_unmatched_intermediate_frames=GT_present,
                GT_absent_intermediate_frames=gap-GT_present,previous_prediction_still_output_at_current_frame=old_visible,
                current_prediction_first_output_frame=current_birth,
                previous_prediction_last_output_frame=pred_last[previous],
                current_prediction_newly_output_within_one_frame=current_birth>=frame-1,
                gap_category='continuous'if gap==0 else'short_1_to_25'if gap<=25 else'several_26_plus')
            events.append(event);own.append(event)
        summaries.append(dict(case=case,sequence=sequence,GT_ID=g,GT_frames=len(frames),
            matched_frames=len(values),unmatched_GT_frames=len(frames)-len(values),
            distinct_matched_prediction_IDs=len(counts),dominant_prediction_matches=max(counts.values())if counts else 0,
            dominant_prediction_GT_coverage=(max(counts.values())/len(frames))if counts else 0.,
            matched_ID_transitions=len(own),continuous_transitions=sum(x['unmatched_intermediate_frames']==0 for x in own),
            short_gap_transitions=sum(1<=x['unmatched_intermediate_frames']<=25 for x in own),
            long_gap_transitions=sum(x['unmatched_intermediate_frames']>=26 for x in own),
            previous_ID_still_output_transitions=sum(x['previous_prediction_still_output_at_current_frame']for x in own),
            transitions_into_earlier_established_ID=sum(not x['current_prediction_newly_output_within_one_frame']for x in own)))
    return summaries,events
