import numpy as np
import pytest

from cfp.identity_evidence import physical_identity_key, pair_discrimination, query_retrieval


def test_physical_ids_are_video_scoped():
    assert physical_identity_key('BT-001', 7) != physical_identity_key('BT-003', 7)
    with pytest.raises(ValueError):
        physical_identity_key('BT-001', -1)


def test_unknown_is_retained_in_pool_but_never_a_known_negative():
    value = query_retrieval([.9,.8,.3], [-1,0,1])
    assert not value['rank1'] and value['unknown_selected']
    assert value['known_only_rank1'] and value['negative_count']==1
    assert value['average_precision']==.5
    assert np.isclose(value['hard_negative_margin'], .5)
    missing = query_retrieval([.2,.1], [1,-1])
    assert not missing['target_present'] and missing['rank1'] is None


def test_auc_and_eer_handle_tied_classes_and_separation():
    tied=pair_discrimination([.5,.5],[.5,.5])
    assert tied['auc']==.5 and tied['eer']==.5
    good=pair_discrimination([.8,.9],[.1,.2])
    assert good['auc']==1 and good['eer']==0
    reverse=pair_discrimination([.1],[.9])
    assert reverse['auc']==0 and reverse['eer']==1
    assert pair_discrimination([], [.3])['auc'] is None


def test_multiple_positive_and_invalid_inputs():
    value=query_retrieval([.9,.8,.7], [0,1,0])
    assert np.isclose(value['average_precision'], (1+2/3)/2)
    assert value['rank1'] and value['rank5']
    with pytest.raises(ValueError): query_retrieval([np.nan],[0])
    with pytest.raises(ValueError): query_retrieval([.5],[2])
