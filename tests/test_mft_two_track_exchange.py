import numpy as np
from tools.diagnose_mft_two_track_exchange import cycle_scores,PastWindow


def test_closed_exchange_keeps_original_matched_rows_columns_and_geometric_support():
    base=np.asarray([[.1,.7],[.7,.1]]);eligible=np.ones((2,2),bool);pairs=np.asarray([[0,0],[1,1]])
    p=np.asarray([[.1,.95],[.93,.2]])
    result=cycle_scores(base,eligible,pairs,p);assert len(result)==1
    assert result[0]['rows']==(0,1)and result[0]['columns']==(0,1)
    assert abs(result[0]['score']-.73)<1e-12
    base[0,1]=.91;assert cycle_scores(base,eligible,pairs,p)==[]
    base[0,1]=.7;eligible[0,0]=False;assert cycle_scores(base,eligible,pairs,p)==[]


def test_third_eligible_competitor_blocks_an_apparently_clear_two_track_exchange():
    base=np.asarray([[.1,.7,.8],[.7,.1,.8]]);pairs=np.asarray([[0,0],[1,1]])
    p=np.asarray([[.1,.95,.97],[.93,.2,.1]]);eligible=np.ones((2,3),bool)
    result=cycle_scores(base,eligible,pairs,p);assert result[0]['score']<0
    eligible[0,2]=False;assert cycle_scores(base,eligible,pairs,p)[0]['score']>.7


def test_three_frame_window_uses_past_only_and_breaks_at_gaps_or_different_pairs():
    w=PastWindow(3);key=(1,2)
    assert w.add(key,1,.8)is None and w.add(key,2,.7)is None
    assert w.add(key,3,.9)==.7
    prefix=dict(w.memory);w.add(key,4,-.8);assert prefix[key]==[(1,.8),(2,.7),(3,.9)]
    assert w.add(key,6,.9)is None
    assert w.add((3,4),7,.9)is None
    w.prune(7);assert key not in w.memory
