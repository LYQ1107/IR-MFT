from tools.register_structural_full_train_test import declared_roles


def test_all_negative_removals_still_require_full_Test_roles():
    analysis = dict(primary='IR', controls=dict(rgb='R',gray='G'),
        removals=[dict(mechanism=m,paired='paired'+m,Dev_best='best'+m,
                       primary_minus_paired_HOTA=-2,primary_minus_Dev_best_HOTA=-3) for m in ['S','C','R']],
        interventions=dict(IR_image_spatial_roll='roll',unavailable='off'))
    result = declared_roles(analysis)
    assert len(result) == 13
    assert all(result['no_'+m+'_paired']=='paired'+m and result['no_'+m+'_Dev_best']=='best'+m for m in ['S','C','R'])
    assert result['primary_IR']=='IR' and result['matched_rgb']=='R' and result['matched_gray']=='G'
