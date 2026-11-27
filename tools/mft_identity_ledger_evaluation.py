"""Prospective ledger cases and fixed missing/permuted modality controls."""
from tools.mft_identity_ledger import LedgerTracker


def policies():
    cases=[dict(id='IL_native_control',kind='reference_native'),
        dict(id='IL_archive_control',kind='reference_archive')]
    for q in ['q95','q99','q999']:
        for shape in [False,True]:
            for kind in ['RGB','critic']:
                cases.append(dict(id='IL_'+kind+'_'+q+('_shape'if shape else''),kind=kind,quantile=q,shape=shape,adaptive=True))
    for kind in ['RGB','critic']:
        cases.append(dict(id='IL_'+kind+'_q99_shape_fixed_memory',kind=kind,quantile='q99',shape=True,adaptive=False))
    return cases


def matched(case,kind):
    assert case in policies()and case['kind']in ['RGB','critic']and kind in ['RGB','critic']
    return next(c for c in policies()if c['kind']==kind and all(c[k]==case[k]for k in ['quantile','shape','adaptive']))


def interventions(case):
    source=matched(case,'critic')
    result=[dict(source,id='IL_control_'+mode,intervention=mode,source_case=source['id'])for mode in ['missing','roll']]
    result.append(dict(case,id='IL_control_fixed_memory',adaptive=False,source_case=case['id']))
    return result


def make_ledger(case,models,thresholds,speed):
    assert case['kind']in ['RGB','critic']
    return LedgerTracker(speed,kind=case['kind'],models=models,thresholds=thresholds,quantile=case['quantile'],
        adaptive=case['adaptive'],shape=case['shape'],intervention=case.get('intervention'))
