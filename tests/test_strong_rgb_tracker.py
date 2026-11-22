import numpy as np
import torch

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker
from yolox.tracker.strong_rgb_tracker import StrongRGBForegroundTracker
from test_calibrated_tracker import data_with_descriptors
from test_foreground_tracker import assert_same


def fitted_model(dimensions,seed):
    torch.manual_seed(seed);model=AssociationCalibrator(base_dimensions=dimensions,logit_bound=8 if dimensions==26 else 4)
    with torch.no_grad():model.net[-1].weight.normal_(std=.2)
    return model


def run(data,tracker,frames):
    BaseTrack._count=0
    return [tracker.update(frame,data) for frame in range(1,frames+1)]


def test_zero_new_residual_preserves_fitted_RGB_probabilities_and_emitted_identities():
    data=data_with_descriptors();RGB=fitted_model(23,2);residual=AssociationCalibrator(base_dimensions=26)
    config=dict(preserve_distinct_objects=True)
    old=CalibratedForegroundTracker(config,model=RGB,cue='rgb')
    expected=run(data,old,3)
    for cue in ['rgb','ir']:
        new=StrongRGBForegroundTracker(config,model=residual,frozen_RGB=RGB,cue=cue)
        actual=run(data,new,3);assert_same(expected,actual)
        for lag in [1,2]:
            old.data=data;old.pair_cache={};new.data=data;new.pair_cache={}
            np.testing.assert_array_equal(old.predict_pairs(3,lag)[1],new.predict_pairs(3,lag)[1])
        assert all(not p.requires_grad for p in new.frozen_RGB.parameters())


def test_missing_auxiliary_or_disabled_new_module_retains_entire_learned_RGB_scorer():
    data=data_with_descriptors();RGB=fitted_model(23,3);residual=fitted_model(26,8)
    old=CalibratedForegroundTracker({},model=RGB,cue='rgb');old.data=data
    expected=old.predict_pairs(2,1)[1]
    for disabled,missing in [(True,False),(False,True)]:
        data['foreground_transport']['quality'][:,1]=0 if missing else 1
        new=StrongRGBForegroundTracker(dict(disable_learned_increment=disabled),model=residual,frozen_RGB=RGB,cue='ir');new.data=data
        value=new.predict_pairs(2,1)
        np.testing.assert_array_equal(value[1],expected)
        assert not np.any(value[2])
    # An ordinary RGB capacity control must run without any IR arrays.
    for key in ['ir_hog','ir_mask_embedding']:del data[key]
    for key in list(data['foreground_transport']):
        if key.startswith('vote_') and ('joint' in key or '_ir' in key):del data['foreground_transport'][key]
    new=StrongRGBForegroundTracker({},model=residual,frozen_RGB=RGB,cue='rgb')
    run(data,new,3)


def test_new_residual_has_no_access_to_future_candidates_or_images():
    first=data_with_descriptors()
    second={k:(v.copy() if isinstance(v,np.ndarray) else {name:arr.copy() for name,arr in v.items()}) for k,v in first.items()}
    for key in ['boxes','embedding','rgb_mask_embedding','ir_mask_embedding','rgb_hog','ir_hog','color']:
        second[key][4:]=-10000
    for lag in [1,2,4,8]:
        point=second['foreground_transport']
        for key in point:
            if key.startswith(f'vote_{lag}_'):
                point[key][point[f'vote_offsets_{lag}'][2]:]=100
    RGB=fitted_model(23,3);residual=fitted_model(26,9)
    a=StrongRGBForegroundTracker({},model=residual,frozen_RGB=RGB,cue='ir')
    b=StrongRGBForegroundTracker({},model=residual,frozen_RGB=RGB,cue='ir')
    assert_same(run(first,a,2),run(second,b,2));assert a.statistics==b.statistics
