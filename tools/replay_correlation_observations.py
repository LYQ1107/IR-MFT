#!/usr/bin/env python3
"""Original ByteTrack on fully produced, frozen correlation observations."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from tools.eval_mft26ir import run_trackeval,write_sequence_results
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.byte_tracker import BYTETracker


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['dev','test'],default='dev');a=p.parse_args()
    config_path=Path(a.config);config=json.loads(config_path.read_text())
    plan_path=ROOT/'configs/perceptual_mft_20261122.json';plan=json.loads(plan_path.read_text());work=Path(plan['work_root'])
    protocol=json.loads((ROOT/plan['protocol']).read_text());sequences=protocol['dev'if a.split=='dev'else'historical_test']
    fusion_path=ROOT/'configs/perceptual_correlation_repair_20261123.json'
    if sha256(fusion_path)!=config['fusion_plan_sha256']:raise ValueError('Registered fusion plan changed')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if(output/'metrics.json').exists():raise ValueError('Completed fusion accuracy cannot be overwritten')
    sources={};coverage={};started=time.perf_counter()
    for sequence in sequences:
        if config.get('baseline'):
            path=Path(plan['frozen_detection_sources']['rgb_'+a.split])/'detections'/(sequence+'.npz')
        else:
            folder=work/config.get('producer_directory','correlation_predictions')/a.split/sequence
            producer=json.loads((folder/'complete.json').read_text())
            info=producer['outputs'][config['proposal_key']];path=Path(info['path'])
            if (producer['status']!='COMPLETE_CORRELATION_OBSERVATIONS' or producer['GT_read'] or producer['future_frame_input']
                    or producer['config_sha256']!=config['fusion_plan_sha256']or producer['specification_sha256']!=config['specification_sha256']
                    or info['sha256']!=sha256(path)):
                raise ValueError('Complete frozen current-image inference required')
        with np.load(path)as z:data={key:z[key]for key in ['frame_ids','offsets','boxes','scores']}
        np.testing.assert_array_equal(data['frame_ids'],np.arange(1,len(data['frame_ids'])+1))
        tracker=BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),frame_rate=25)
        rows=[]
        for frame in data['frame_ids']:
            lo,hi=data['offsets'][frame-1:frame+1];det=np.c_[data['boxes'][lo:hi],data['scores'][lo:hi]].astype(np.float32)
            rows.extend((int(frame),track.tlwh.copy(),track.track_id,track.score)
                        for track in tracker.update(det,(1,1),(1,1))if track.tlwh[2]*track.tlwh[3]>100)
        target=output/'track_results'/(sequence+'.txt');write_sequence_results(str(target),rows)
        sources[sequence]=dict(path=str(path),sha256=sha256(path))
        coverage[sequence]=dict(frames=len(data['frame_ids']),detections=len(data['boxes']),output_rows=len(rows),output_sha256=sha256(target))
    manifest=dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION',config=config,config_sha256=sha256(config_path),
        plan_sha256=sha256(plan_path),split=a.split,sequences=sequences,frames=sum(x['frames']for x in coverage.values()),
        sources=sources,coverage=coverage,statistics={},GT_inference_input=False,future_frame_input=False,
        previous_outputs_rewritten=False,replay_seconds=time.perf_counter()-started,replay_source_sha256=sha256(__file__),
        byte_tracker_source_sha256=sha256(ROOT/'yolox/tracker/byte_tracker.py'),
        replay_FPS_scope='Cached proposal replay only; global detector/fusion computation excluded, not end-to-end FPS.')
    manifest['replay_FPS']=manifest['frames']/manifest['replay_seconds'];dump(output/'inference_manifest.json',manifest)
    metrics=run_trackeval(str(output/'track_results'),plan['evaluation']['GT_root'],str(output),sequences,
                          plan['evaluation']['TrackEval_root'],data_split='train'if a.split=='dev'else'test')
    dump(output/'metrics.json',dict(status='COMPLETE',trackeval=metrics,manifest=manifest))
    print('CORRELATION_REPLAY_COMPLETE',json.dumps(metrics),flush=True)


if __name__=='__main__':main()
