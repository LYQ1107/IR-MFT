"""Fixed archive ablations and current-box rendering for full-Dev comparison."""
import numpy as np
from tools.mft_archive_reasoning import ArchiveLinker
from tools.mft_candidate_risk_teacher import CandidateTeacher


def policies():
    refs=[dict(id='AR_native_zero',kind='reference_native',quantile=None,shape=False),
        dict(id='AR_fixed_RGB_late',kind='reference_late',quantile=None,shape=False),
        dict(id='AR_fixed_RGB_late_shape',kind='reference_pose',quantile=None,shape=True)]
    fitted=[]
    for q in ['q99','q999']:
        for shape in [False,True]:
            for kind in ['RGB','paired']:
                fitted.append(dict(id='AR_'+kind+'_'+q+('_shape'if shape else''),kind=kind,quantile=q,shape=shape))
    return refs+fitted


def matched_case(case,kind):
    assert case in policies()and case['kind']in ['RGB','paired']and kind in ['RGB','paired']
    return next(c for c in policies()if c['kind']==kind and c['shape']==case['shape']and c['quantile']==case['quantile'])


def interventions(case):
    assert case in policies()and case['kind']in ['RGB','paired']
    source=matched_case(case,'paired')
    return [dict(source,id='AR_intervention_'+mode,intervention=mode,source_case=source['id'])for mode in ['missing','roll']]


def intervene(ir,case):
    mode=case.get('intervention')
    if mode=='missing':return None
    if mode=='roll':return np.roll(np.asarray(ir),1,axis=0).copy()
    assert mode is None;return ir


class ArchiveTracker:
    def __init__(self,case,models,thresholds,speed):
        assert case in policies()or case in interventions(next(c for c in policies()if c['id']==case.get('source_case')))
        assert case['kind']in ['RGB','paired']
        self.case=dict(case);self.linker=ArchiveLinker(CandidateTeacher(speed),speed,kind=case['kind'],models=models,
            thresholds={k:v[case['quantile']]for k,v in thresholds.items()})
        self.sequence=None;self.shape_rows=0

    @property
    def statistics(self):
        return dict(self.linker.inner.statistics,
            **{'archive_'+k:v for k,v in self.linker.statistics.items()},shape_rows=self.shape_rows)

    def update(self,sequence,frame,detections,rgb,ir=None):
        if self.sequence!=sequence:self.sequence=sequence;self.shape_rows=0
        rows=self.linker.update(sequence,frame,detections,rgb,intervene(ir,self.case))
        if not self.case['shape']:return rows
        live={self.linker.published[int(t.track_id)]:t for t in self.linker.inner.core.tracked_stracks
            if t.is_activated and t.frame_id==frame and int(t.track_id)in self.linker.published}
        result=[]
        for f,filtered,identity,score in rows:
            t=live[identity];assert t.measurement_frame==frame
            measured=t.current_measurement
            if measured[2]*measured[3]>100:
                box=measured.copy();box[:2]=filtered[:2]+filtered[2:]/2-box[2:]/2;self.shape_rows+=1
            else:box=filtered
            result.append((f,box,identity,score))
        return result
