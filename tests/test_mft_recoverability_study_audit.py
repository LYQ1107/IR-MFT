import numpy as np
import pytest
from tools.audit_mft_recoverability_studies import linear_quantile,check_frozen_choice


def test_sorted_quantile_handles_empty_chunks_endpoints_and_does_not_mutate_scores():
    a=np.asarray([.91,.1,.85,.2,.99,.5]);before=a.copy()
    for q in [0,.99,.999,1]:assert abs(linear_quantile(a,q)-np.quantile(a,q))<2e-15
    np.testing.assert_array_equal(a,before)
    assert linear_quantile(np.asarray([.3]),.999)==.3
    for bad in [[],[np.nan],[1.01],[-.1]]:
        with pytest.raises(AssertionError):linear_quantile(bad,.999)


def test_Dev_ranking_detects_Test_reselection_and_gate_violations_with_ties():
    def row(name,h,a,i):return dict(case=dict(id=name),combined47=dict(HOTA=h,AssA=a,IDF1=i))
    rows=[row('RGB',50,60,70),row('paired',50,61,69),row('other',49,70,80)]
    winner=rows[1]['case'];freeze=dict(chosen_case=winner,Test_read_for_selection=False)
    chosen,gain=check_frozen_choice(rows,winner,49,freeze);assert chosen['case']==winner and gain==1
    with pytest.raises(AssertionError):check_frozen_choice(rows,rows[0]['case'],49,freeze)
    with pytest.raises(AssertionError):check_frozen_choice(rows,winner,50,freeze)
    with pytest.raises(AssertionError):check_frozen_choice(rows,winner,49,dict(freeze,Test_read_for_selection=True))
    tied=[row('z',50,60,70),row('a',50,60,70)]
    assert check_frozen_choice(tied,tied[1]['case'],49)[0]['case']['id']=='a'
