import torch

from yolox.models.temporal_correction import TemporalCorrection


def sample(x,amount):
    base=torch.ones(1,2,8,8);corrected=base.clone();corrected[:,:,3,x]+=amount
    trust=torch.ones(1,1,8,8);return base,corrected,trust


def motion():
    backward=torch.zeros(1,2,8,8);backward[:,0]=-1
    return backward,torch.ones(1,1,8,8,dtype=torch.bool)


def test_zero_strength_and_none_preserve_original_objects_without_flow_read():
    base,corrected,trust=sample(2,.2)
    for temporal in [TemporalCorrection(strength=0),TemporalCorrection(mode='none')]:
        assert temporal.update([base],[corrected],[trust],backward_flow='unused')[0]is corrected


def test_rgb_motion_moves_history_to_current_object_instead_of_stationary_background():
    first,old,confidence=sample(2,.2);current,repaired,trust=sample(3,.1);flow,valid=motion()
    temporal=TemporalCorrection();assert temporal.update([first],[old],[confidence])[0]is old
    result=temporal.update([current],[repaired],[trust],flow,valid)[0]
    torch.testing.assert_close(result[:,:,3,3],torch.full((1,2),1.15),atol=1e-6,rtol=0)
    torch.testing.assert_close(result[:,:,3,2],current[:,:,3,2],atol=0,rtol=0)
    stationary=TemporalCorrection(mode='identity');stationary.update([first],[old],[confidence])
    torch.testing.assert_close(stationary.update([current],[repaired],[trust])[0],repaired,atol=0,rtol=0)


def test_invalid_motion_does_not_change_current_features():
    first,old,trust=sample(2,.2);current,repaired,_=sample(3,.1);flow,valid=motion()
    temporal=TemporalCorrection();temporal.update([first],[old],[trust])
    torch.testing.assert_close(temporal.update([current],[repaired],[trust],flow,~valid)[0],repaired,atol=0,rtol=0)


def test_missing_auxiliary_clears_history_before_any_repaired_tensor_access():
    base,old,trust=sample(2,.2);temporal=TemporalCorrection();temporal.update([base],[old],[trust])
    assert temporal.update([base],None,None,available=False)[0]is base
    assert temporal.previous is None
    current,repaired,_=sample(3,.1);flow,valid=motion()
    assert temporal.update([current],[repaired],[trust],flow,valid)[0]is repaired


def test_current_trust_blocks_stale_correction_while_removal_can_carry_it():
    base,old,trust=sample(2,.2);flow,valid=motion();zero=torch.zeros_like(trust)
    protected=TemporalCorrection();protected.update([base],[old],[trust])
    torch.testing.assert_close(protected.update([base],[base],[zero],flow,valid)[0],base,atol=0,rtol=0)
    removed=TemporalCorrection(mode='without_current_evidence');removed.update([base],[old],[trust])
    assert removed.update([base],[base],[zero],flow,valid)[0][0,0,3,3]>1


def test_future_changes_do_not_change_prefix_and_state_does_not_alias_inputs():
    def run(last):
        temporal=TemporalCorrection();outputs=[];flow,valid=motion()
        for x,amount in [(2,.2),(3,.1),(4,last)]:
            base,repaired,trust=sample(x,amount)
            outputs.append(temporal.update([base],[repaired],[trust],flow,valid)[0].clone())
            repaired.fill_(99)
        return outputs
    first,second=run(.1),run(.4)
    for a,b in zip(first[:2],second[:2]):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert first[1][0,0,3,3]<2
