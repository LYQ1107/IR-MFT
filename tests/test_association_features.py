import numpy as np

from yolox.tracker.association_features import pair_features,vote_features


def frames():
    rng=np.random.default_rng(6)
    data=dict(boxes=np.array([[10,10,30,30],[50,10,70,30]],np.float32),
              quality=np.ones((2,2),np.float32),point_quality=np.ones((2,2),np.float32))
    for key in ['embedding','rgb_mask_embedding','ir_mask_embedding','rgb_hog','ir_hog','color']:
        data[key]=rng.normal(size=(2,4)).astype(np.float32)
    matrices={(s,t):np.eye(2,dtype=np.float32)*.7 for s in ['bbox','rgb','joint'] for t in ['bbox','rgb','joint']}
    good={s:np.array([20,20],np.uint8) for s in ['bbox','rgb','joint']}
    return data,matrices,good


def test_RGB_pair_control_runs_without_any_IR_descriptors_or_votes():
    data,matrices,good=frames()
    original=pair_features(data,data,matrices,good,good,1,'rgb')
    del data['ir_hog'];del data['ir_mask_embedding']
    data['quality'][:,1]=1e4;data['point_quality'][:,1]=-1e4
    matrices={k:v for k,v in matrices.items() if 'joint' not in k}
    repeat=pair_features(data,data,matrices,good,good,1,'rgb')
    for first,second in zip(original,repeat):np.testing.assert_array_equal(first,second)
    assert [v.shape for v in repeat]==[(2,2,23),(2,2,17),(2,2,17)]


def test_pair_features_remain_finite_with_missing_current_candidates():
    data,matrices,good=frames();empty={k:v[:0] for k,v in data.items()}
    arrays=pair_features(data,empty,{},good,good,1,'ir')
    assert [v.shape for v in arrays]==[(2,0,23),(2,0,17),(2,0,17)]
    assert all(np.isfinite(v).all() for v in arrays)


def test_competing_equal_ownership_is_ambiguous_and_labels_are_never_inputs():
    vote=np.array([[.5,.5],[.5,0]],np.float32);features=vote_features(vote)
    assert features[0,0,2]==0 and features[0,0,4]==0
    assert features[0,1,2]==0 and features[0,1,4]>.4
    data,matrices,good=frames();before=pair_features(data,data,matrices,good,good,1,'ir')
    data['labels']=np.array([9000,-1])
    after=pair_features(data,data,matrices,good,good,1,'ir')
    for x,y in zip(before,after):np.testing.assert_array_equal(x,y)
