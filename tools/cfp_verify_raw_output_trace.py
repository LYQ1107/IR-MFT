#!/usr/bin/env python3
"""Independent cached-output replay after actual raw inference has completed."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from cfp.lifecycle_crossover import load_bot_reference, make_cfp_lifecycle
from cfp.native_local_identity import RecentIdentityControl
from cfp.predictive import LatentIdentityPredictor
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True); ap.add_argument('--output', type=Path, required=True); a=ap.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New CPU-only independent post-inference proof required')
    m=json.loads((a.run/'manifest.json').read_text()); c=m['config']; trace=a.run/'actual_current_outputs.npz'
    if m['status']!='COMPLETE_WHOLE_RAW_CURRENT_VIDEO_PROFILE' or sha256(trace)!=m['actual_current_output_trace_sha256'] or m['cached_detection_or_feature_used_for_inference']:
        raise ValueError('Actual raw inference must complete first, without cached inference inputs')
    torch.set_num_threads(2); BaseTrack._count=0
    if m['kind']=='legacy_gru':
        predictor=LatentIdentityPredictor().eval(); predictor.load_state_dict(torch.load(c['predictor_checkpoint'],map_location='cpu',weights_only=False)['model'])
        tracker=make_cfp_lifecycle(load_bot_reference('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT'),'gru',predictor,.3)
    else:
        tracker=PredictiveAssociationTracker(SimpleNamespace(track_thresh=.58,track_buffer=45,match_thresh=.97,mot20=False),
            frame_rate=25,predictor=RecentIdentityControl('fifo'),memory='predictor',history_view='recent',appearance_weight=.3,proximity_iou=.1,ema=.9)
    with np.load(trace,allow_pickle=False) as bank:
        frames,offsets,boxes,scores,embedding=[bank[k].copy() for k in ['frame_ids','offsets','boxes','scores','embedding']]
    if not np.array_equal(frames,np.arange(1,len(frames)+1)) or len(frames)!=m['frames'] or offsets[-1]!=len(boxes) or len(embedding)!=len(boxes):
        raise ValueError('All actual current output rows and frames required')
    prediction=a.output.with_suffix('.txt')
    with prediction.open('w') as stream:
        for i,frame in enumerate(frames):
            lo,hi=offsets[i:i+2]; indices=np.arange(lo,hi)
            if m['kind']=='legacy_gru':output=tracker.update(np.c_[boxes[lo:hi],scores[lo:hi]],embedding[lo:hi].copy(),indices)
            else:output=tracker.update(np.c_[boxes[lo:hi],scores[lo:hi]],(1,1),(1,1),embedding[lo:hi].copy(),proposal_indices=indices)
            for t in output:
                if np.prod(t.tlwh[2:])<=100:continue
                index=int(t.current_proposal_index)
                if t.frame_id!=frame or not lo<=index<hi:raise ValueError('Independent replay lost current measurement')
                b=boxes[index].copy();b[2:]-=b[:2]
                stream.write(f'{frame},{int(t.track_id)},{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f},{float(t.score):.6f},-1,-1,-1\n')
    exact=prediction.read_bytes()==(a.run/(m['video']+'.txt')).read_bytes()
    write_json(a.output,dict(status='EXACT_ACTUAL_RAW_OUTPUT_TRACE_REPLAY_VERIFIED' if exact else 'ACTUAL_RAW_TRACE_REPLAY_DIFFERS',
        frames=len(frames),video=m['video'],kind=m['kind'],run_manifest_sha256=sha256(a.run/'manifest.json'),trace_sha256=sha256(trace),
        raw_prediction_sha256=m['prediction_sha256'],replay_prediction_sha256=sha256(prediction),prediction_byte_equal=exact,
        sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__]),
        scope='Cached currentoutputs are used only by this separate post-completion CPU replay, never raw inference. Certifies actualalways-pad8 raw proposal/feature/association consistency; olderglobalunpad3-tail cached benchmark may legitimately differ. Replay is not rawFPS and has no GT/teacher/future input.'))
    if not exact:raise ValueError('Independent post-run replay differs from actual raw outputs')


if __name__=='__main__':main()
