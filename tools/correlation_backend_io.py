"""Read only frozen, full-Dev correlation streams for a shared backend."""
import json
from pathlib import Path

import numpy as np
from tools.run_pseudo_ir_comparison import sha256

ROOT=Path(__file__).resolve().parents[1]


def proposal_streams(plan,split,sequence):
    if split!='dev':raise ValueError('Only Dev interaction diagnosis registered')
    registration=plan['correlation_backend_registration']
    if sha256(ROOT/'configs/perceptual_mft_20261122.json')!=registration['original_plan_sha256']:
        raise ValueError('Original setting changed')
    streams={};sources={}
    for cue,row in registration['streams'].items():
        folder=Path(row['producer'])/sequence;meta=json.loads((folder/'complete.json').read_text())
        info=meta['outputs'][row['id']];path=Path(info['path'])
        if (meta['status']!='COMPLETE_CORRELATION_OBSERVATIONS'or meta['GT_read']or meta['future_frame_input']
                or info['sha256']!=sha256(path)or meta['IR_intervention']!='none'):
            raise ValueError('Frozen actual-input observation source required')
        with np.load(path)as z:streams[cue]={key:z[key].copy()for key in ['frame_ids','offsets','boxes','scores']}
        sources[cue]=dict(path=str(path),sha256=info['sha256'],producer_manifest_sha256=sha256(folder/'complete.json'))
    if set(streams)!={'rgb','gray','ir'}:raise ValueError('All3 matched observation cues required')
    return streams,sources
