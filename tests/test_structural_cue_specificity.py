import numpy as np
import pytest

from tools.diagnose_structural_cue_specificity import contrast_auc,support_counts


@pytest.mark.parametrize('positive,negative',[
    ([2,3],[-2,0]),([-2,-1],[1,2]),([0,0],[0,0]),([0,1,1,2],[0,0,1,3]),
])
def test_contrast_auc_matches_all_pairs_with_ties(positive,negative):
    exact=np.mean([float(p>n)+.5*float(p==n) for p in positive for n in negative])
    assert contrast_auc(positive,negative)==exact
    assert contrast_auc(positive[::-1],negative[::-1])==exact


def test_empty_region_is_unknown_and_nonfinite_rejected():
    assert contrast_auc([], [1]) is None
    with pytest.raises(ValueError):contrast_auc([np.nan],[0])
    with pytest.raises(ValueError):contrast_auc([[1]],[0])


def test_support_conserves_current_past_and_reference_with_inclusive_threshold():
    row=support_counts([.6,.2,.7,.4],[.6,.8,.2,.4],[.3,.8,.7,.1],.6)
    assert row['current']==row['reference']+row['current_gain_vs_reference']-row['current_loss_vs_reference']
    assert row['current']==row['past']+row['current_gain_vs_past']-row['current_loss_vs_past']
    assert row==dict(reference=2,current=2,past=2,current_gain_vs_reference=1,
        current_loss_vs_reference=1,current_gain_vs_past=1,current_loss_vs_past=1)


@pytest.mark.parametrize('values',[
    ([.5],[.5,.6],[.5]),([1.1],[.5],[.5]),([.5],[float('inf')],[.5]),
])
def test_support_rejects_incomparable_or_invalid_probabilities(values):
    with pytest.raises(ValueError):support_counts(*values,.6)
