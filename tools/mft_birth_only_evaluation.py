"""Frozen birth-only matched controls and current qualification receipts."""
import hashlib
import numpy as np
from tools.mft_birth_only import BirthOnlyQuality,birth_mask,policies
from tools.mft_proposal_quality_evaluation import SharedConfirmation


def matched(case,kind):
    assert case['kind']in ['RGB','paired','duplicatedRGB']
    return next(c for c in policies()if c['kind']==kind and c.get('quantile')==case['quantile']and c.get('confirmation')==case['confirmation'])


def anchor(case):
    if case['kind']in ['RGB','paired','duplicatedRGB']:return matched(case,'paired')
    assert case['kind']=='unconditional'
    return next(c for c in policies()if c['kind']=='paired'and c['quantile']=='q99'and c['confirmation']==case['confirmation'])


def interventions(case):
    paired=anchor(case)
    rows=[dict(paired,id='BO_control_'+mode,intervention=mode,source_case=paired['id'])for mode in ['missing','roll']]
    rows.append(dict(case,id='BO_control_confirmation',confirmation=0 if case['confirmation']else 3,
        calibration_condition='confirmed'if case['confirmation']else'all',source_case=case['id']))
    rows.append(dict(case,id='BO_control_no_archive',parent='native_shape',source_case=case['id']))
    rows.append(dict(case,id='BO_control_no_shape',parent='archive_raw',source_case=case['id']))
    return rows


def Test_cases(case):
    paired=anchor(case);rows=policies()[:2]+[case]+[matched(paired,k)for k in ['RGB','paired','duplicatedRGB']]
    rows.append(next(c for c in policies()if c['kind']=='unconditional'and c['confirmation']==case['confirmation']))
    return list({c['id']:c for c in rows}.values())


class ReceiptBirthOnlyQuality(BirthOnlyQuality):
    def update(self,sequence,frame,detections,rgb,ir=None):
        if self.sequence!=sequence:
            assert frame==1;self.qualification_packet_hash=hashlib.sha256()
        d=np.asarray(detections,np.float32)
        if self.case['kind'].startswith('reference_'):allowed=np.zeros(len(d),bool)
        else:
            streak=self.confirmation.update(sequence,frame,d)
            allowed=birth_mask(self.case,d,rgb,ir,self.views,self.heads,self.thresholds,streak)
        self.qualification_packet_hash.update(np.asarray([frame,len(d)],dtype='<i8').tobytes()+allowed.tobytes())
        before=d.copy();result=super().update(sequence,frame,d,rgb,ir)
        np.testing.assert_array_equal(d,before)
        return result


def make_tracker(case,views,heads,thresholds,speed,models,old_thresholds,shared):
    return ReceiptBirthOnlyQuality(case,views,heads,thresholds,speed,models,old_thresholds,shared)
