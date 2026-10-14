import numpy as np
from cfp.observable_occlusion import current_overlap,quality_gate


def test_overlap_proxy_is_current_set_permutation_scale_invariant_and_handles_empty():
    boxes=np.array([[0,0,10,10],[5,0,15,10],[30,0,40,10]],float)
    expected=np.array([1/3,1/3,0.]);assert np.allclose(current_overlap(boxes),expected)
    order=np.array([2,0,1]);assert np.array_equal(current_overlap(boxes[order]),current_overlap(boxes)[order])
    assert np.allclose(current_overlap(boxes*7+100),expected)
    assert current_overlap(np.empty((0,4))).shape==(0,)
    assert np.array_equal(current_overlap(boxes[:1]),[0.])
    assert np.array_equal(current_overlap(np.repeat(boxes[:1],2,axis=0)),[1.,1.])


def test_controls_preserve_native_graph_and_row_weight_distribution_without_labels():
    overlap=np.array([.8,.3,.5,0.]);eligible=np.array([[1,0,1,1],[0,1,1,0],[0,0,0,0]],bool)
    adapted=quality_gate(overlap,eligible,'overlap',np.random.default_rng(0));mean=quality_gate(overlap,eligible,'row_mean',np.random.default_rng(0));shuffled=quality_gate(overlap,eligible,'permuted_overlap',np.random.default_rng(1))
    assert np.all(adapted[~eligible]==1) and np.all(mean[~eligible]==1) and np.all(shuffled[~eligible]==1)
    for row in range(2):
        assert np.isclose(adapted[row,eligible[row]].mean(),mean[row,eligible[row]].mean())
        assert np.array_equal(np.sort(adapted[row,eligible[row]]),np.sort(shuffled[row,eligible[row]]))
    assert np.array_equal(quality_gate(overlap,eligible,'native',np.random.default_rng(0)),np.ones(eligible.shape))
