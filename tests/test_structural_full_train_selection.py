import pytest

from tools.export_structural_full_train_dev import select_curve, paired_row


def row(name, epoch, policy, HOTA, bits='111', cue='ir', intervention=None):
    return dict(id=name, config=dict(epoch=epoch, policy=policy, components=bits, cue=cue,
                                     intervention=intervention), metrics=dict(HOTA=HOTA))


def test_Dev_tie_rule_prefers_earlier_checkpoint_then_raw_and_excludes_interventions():
    rows = [row('later_raw',48,'raw',50), row('early_priority',24,'priority',50),
            row('early_raw',24,'raw',50), row('wrong_IR',24,'raw',90,intervention='IR_image_spatial_roll'),
            row('gray',24,'raw',80,cue='gray')]
    assert select_curve(rows,'ir','111')['id']=='early_raw'


def test_paired_removal_keeps_primary_epoch_and_policy_even_when_its_curve_is_worse():
    primary=row('full',48,'priority',50)
    rows=[row('paired',48,'priority',40,bits='011'),row('best',24,'raw',55,bits='011')]
    assert paired_row(rows,'011',primary)['id']=='paired'
    assert select_curve(rows,'ir','011')['id']=='best'
    with pytest.raises(ValueError,match='exact'):
        paired_row(rows,'101',primary)
