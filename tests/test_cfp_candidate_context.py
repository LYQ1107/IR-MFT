import numpy as np

from cfp.association import IdentityTrack
from cfp.candidate_context import complete_candidate_context


def test_true_geometry_fallback_candidate_survives_counterfactual_context():
    one=np.array([1.,0.],np.float32);other=np.array([0.,1.],np.float32)
    track=IdentityTrack([0,0,30,20],.9,one,one)
    track.memory_rgb=track.memory_ir=one;track.last_identity_frame=1;track.is_activated=True;track.start_frame=1
    track.identity_history.append((1,one,one))
    detections=[IdentityTrack([0,0,30,20],.9,other,other,10),IdentityTrack([0,0,30,20],.9,one,other,11)]
    context=complete_candidate_context(track,detections,np.array([.1,.2]),np.array([.2,.3]),2)
    assert context['pool'].tolist()==[0,1] and context['modifiable'].tolist()==[1]
    assert context['rgb'][0]==context['ir'][0]==.2  # True candidate remains a geometry score, not falsely NULL.
    assert context['rgb'][1]!=context['ir'][1]
    assert context['null_distance']==.97
    track.is_activated=False
    assert complete_candidate_context(track,detections,np.array([.1,.2]),np.array([.2,.3]),2)['null_distance']==.7
