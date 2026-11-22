"""Verified corrupted auxiliary packets over the unchanged clean query data."""
import json
from pathlib import Path

import numpy as np

from tools.native_query_data import load_query_group
from tools.run_pseudo_ir_comparison import sha256

ROOT=Path(__file__).resolve().parents[1]


def load_robust_query_group(work,sequence,regime,cue,seed):
    data=load_query_group(work,sequence,regime,cue,seed)
    directory=Path(work)/'native_mismatch_query_bank'/sequence
    meta=json.loads((directory/'complete.json').read_text())
    plan=ROOT/'configs/perceptual_native_robust_query_20261123.json'
    if (meta['status']!='COMPLETE_TRAIN_ONLY_MISMATCH_QUERY_BANK' or meta['Dev_rows'] or meta['Test_rows']
            or meta['config_sha256']!=sha256(plan)):
        raise ValueError('Mismatched cue bank provenance')
    info=meta['outputs'][regime][f'{cue}_s{seed}'];path=Path(info['path'])
    if sha256(path)!=info['sha256']:
        raise ValueError('Mismatched cue packet changed')
    with np.load(path) as bank:
        quality=bank['quality'].copy()
        data['mismatch_cue']=np.c_[bank['cue7'],quality].astype(np.float32)
        valid=bank['mismatch_valid'].copy()
    if len(valid)!=len(data['query']):
        raise ValueError('Mismatched cue query indexing')
    data['mismatch_available']=valid & ((quality[:,0]>=.15)&(quality[:,1]>=.15) if cue=='ir' else True)
    ref=meta['outputs'][regime]['reference'];path=Path(ref['path'])
    if sha256(path)!=ref['sha256']:
        raise ValueError('Mismatched reference packet changed')
    with np.load(path) as bank:
        np.testing.assert_array_equal(bank['RGB3'],data['base'][:,23:26])
        np.testing.assert_array_equal(bank['reference7'],data['reference'][:,:7])
    data['sources']['mismatch']=info['sha256']
    data['sources']['mismatch_reference']=ref['sha256']
    return data
