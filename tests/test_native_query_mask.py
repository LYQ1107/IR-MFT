import torch

from yolox.tracker.native_query_mask import remask_appearance_queries


def test_deleted_candidate_cannot_remain_a_hidden_RGB_or_native_competitor():
    base=torch.zeros(1,3,26);base[0,:,23]=torch.tensor([.3,.9,.2]);base[0,:,24]=torch.tensor([-.6,.6,-.7])
    cue=torch.zeros(1,3,9);cue[0,:,0]=torch.tensor([.4,.95,.1]);reference=cue.clone()
    valid=torch.tensor([[True,False,True]])
    first=remask_appearance_queries(base,cue,reference,valid)
    base[0,1,23]=100;cue[0,1,0]=200;reference[0,1,0]=300
    second=remask_appearance_queries(base,cue,reference,valid)
    for a,b in zip(first,second):torch.testing.assert_close(a[valid],b[valid],rtol=0,atol=0)
    torch.testing.assert_close(first[0][0,0,24],torch.tensor(.1),atol=1e-6,rtol=0)


def test_local_point_packet_cannot_be_reused_after_an_unphysical_deletion():
    base=torch.zeros(1,2,26);base[0,0,14]=.5
    try:remask_appearance_queries(base,torch.zeros(1,2,9),torch.zeros(1,2,9),torch.ones(1,2,dtype=torch.bool))
    except ValueError:return
    raise AssertionError('A local ownership packet requires actual recomputation, not a hidden candidate mask')
