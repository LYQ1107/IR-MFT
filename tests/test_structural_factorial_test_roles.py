from tools.register_structural_factorial_test import declared_roles


def test_all_formal_components_keep_paired_and_best_roles_even_when_Dev_is_negative():
    paired={bits:dict(id=bits+'_paired',metrics={'HOTA':1.})for bits in ['000','001','010','100']}
    best={bits:dict(id=bits+'_best',metrics={'HOTA':2.})for bits in paired}
    roles=declared_roles(dict(paired6_raw=paired,best4_by_components=best))
    assert len(roles)==8
    assert {name for name,row in roles}=={bits+'_'+label for bits in paired for label in ['paired6','Dev_best4']}
    assert all(row is (paired[bits]if label=='paired6'else best[bits])
               for name,row in roles for bits,label in [name.split('_',1)])
