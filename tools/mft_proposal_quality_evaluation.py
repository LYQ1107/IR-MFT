"""Prospective quality/confirmation controls and immutable parent rendering."""
import numpy as np
from tools.mft_proposal_quality import ProposalConfirmation,ProposalQualityTracker
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_archive_evaluation import ArchiveTracker,policies as archive_policies


def policies():
    cases=[dict(id='PQ_native_control',kind='reference_native'),dict(id='PQ_archive_control',kind='reference_archive')]
    for confirmation in [0,3]:cases.append(dict(id='PQ_unconditional_c'+str(confirmation),kind='unconditional',confirmation=confirmation,quantile=None))
    for q in ['q99','q999']:
        for confirmation in [0,3]:
            for kind in ['RGB','paired','duplicatedRGB']:
                cases.append(dict(id='PQ_'+kind+'_'+q+'_c'+str(confirmation),kind=kind,quantile=q,confirmation=confirmation))
    return cases


def matched(case,kind):
    assert case in policies()and case['kind']in ['RGB','paired','duplicatedRGB']
    return next(c for c in policies()if c['kind']==kind and c.get('quantile')==case['quantile']and c.get('confirmation')==case['confirmation'])


def quality_anchor(case):
    if case['kind']in ['RGB','paired','duplicatedRGB']:return matched(case,'paired')
    assert case['kind']=='unconditional'
    return next(c for c in policies()if c['kind']=='paired'and c['quantile']=='q99'and c['confirmation']==case['confirmation'])


def interventions(case):
    paired=quality_anchor(case)
    controls=[dict(paired,id='PQ_control_'+mode,intervention=mode,source_case=paired['id'])for mode in ['missing','roll']]
    condition='confirmed'if case['confirmation']else'all'
    controls.append(dict(case,id='PQ_control_confirmation',confirmation=0 if case['confirmation']else 3,
        calibration_condition=condition,source_case=case['id']))
    controls.append(dict(case,id='PQ_control_no_archive',parent='native_shape',source_case=case['id']))
    controls.append(dict(case,id='PQ_control_no_shape',parent='archive_raw',source_case=case['id']))
    return controls


def Test_cases(case):
    anchor=quality_anchor(case)
    rows=policies()[:2]+[case]+[matched(anchor,k)for k in ['RGB','paired','duplicatedRGB']]
    rows.append(next(c for c in policies()if c['kind']=='unconditional'and c['confirmation']==case['confirmation']))
    return list({c['id']:c for c in rows}.values())


class SharedConfirmation:
    """One read-only current-frame original-proposal observation for all cases."""
    def __init__(self):self.observer=ProposalConfirmation();self.sequence=None;self.frame=0

    def update(self,sequence,frame,d):
        if sequence!=self.sequence or frame!=self.frame:
            self.values=self.observer.update(sequence,frame,d);self.sequence=sequence;self.frame=frame;self.packet=np.asarray(d).copy()
        else:np.testing.assert_array_equal(d,self.packet)
        return self.values.copy()


class ShapeTeacher:
    def __init__(self,speed):self.teacher=CandidateTeacher(speed)
    @property
    def statistics(self):return dict(self.teacher.statistics)
    def update(self,sequence,frame,d,rgb,ir=None):
        rows=self.teacher.update(sequence,frame,d,rgb,ir)
        tracks={int(t.track_id):t for t in self.teacher.core.tracked_stracks};result=[]
        for f,box,identity,score in rows:
            measured=tracks[identity].current_measurement
            if measured[2]*measured[3]>100:
                new=measured.copy();new[:2]=box[:2]+box[2:]/2-new[2:]/2;box=new
            result.append((f,box,identity,score))
        return result


def make_tracker(case,views,heads,thresholds,speed,archive_models,archive_thresholds,shared=None):
    if case['kind']=='reference_native':return CandidateTeacher(speed)
    if case['kind']=='reference_archive':
        c=next(c for c in archive_policies()if c['id']=='AR_RGB_q999_shape')
        return ArchiveTracker(c,archive_models,archive_thresholds,speed)
    parent=case.get('parent','archive_shape')
    if parent=='native_shape':inner=ShapeTeacher(speed)
    else:
        assert parent in ['archive_shape','archive_raw']
        name='AR_RGB_q999_shape'if parent=='archive_shape'else'AR_RGB_q999'
        c=next(c for c in archive_policies()if c['id']==name)
        inner=ArchiveTracker(c,archive_models,archive_thresholds,speed)
    result=ProposalQualityTracker(case,views,heads,thresholds,inner)
    if shared is not None:result.confirmation=shared
    return result
