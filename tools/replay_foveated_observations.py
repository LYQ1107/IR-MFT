#!/usr/bin/env python3
"""Complete causal ByteTrack screen of frozen foveated observation streams."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np

from tools.eval_mft26ir import run_trackeval,write_sequence_results
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.byte_tracker import BYTETracker


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--split',choices=['dev','test'],default='dev');parser.add_argument('--output',required=True);args=parser.parse_args()
    config_path=Path(args.config);config=json.loads(config_path.read_text())
    plan_path=ROOT/'configs/perceptual_mft_20261122.json';plan=json.loads(plan_path.read_text());work=Path(plan['work_root'])
    protocol=json.loads((ROOT/plan['protocol']).read_text());sequences=protocol['dev' if args.split=='dev' else 'historical_test']
    fovea_path=ROOT/'configs/perceptual_foveated_mft_20261123.json'
    if config['fovea_plan_sha256']!=sha256(fovea_path):raise ValueError('Registered foveated policy changed')
    output=Path(args.output)
    if (output/'metrics.json').exists():raise ValueError('Refuse to overwrite completed foveated accuracy')
    output.mkdir(parents=True,exist_ok=True);sources={};coverage={};started=time.perf_counter()
    for sequence in sequences:
        key=config['proposal_key']
        if key.startswith('baseline_'):
            arm=key.removeprefix('baseline_');prefix='new_diffv2ir' if arm=='new' else 'rgb'
            path=Path(plan['frozen_detection_sources'][prefix+'_'+args.split])/'detections'/(sequence+'.npz')
        else:
            folder=work/'foveated_proposals'/args.split/sequence
            producer=json.loads((folder/'complete.json').read_text());info=producer['outputs'][key];path=Path(info['path'])
            if (producer['status']!='COMPLETE_FOVEATED_CURRENT_OBSERVATIONS' or producer['GT_read'] or producer['future_frame_input']
                    or producer['config_sha256']!=config['fovea_plan_sha256'] or sha256(path)!=info['sha256']):
                raise ValueError('Foveated inference provenance mismatch')
        with np.load(path) as source:data={name:source[name] for name in ['frame_ids','offsets','boxes','scores']}
        frames=data['frame_ids']
        np.testing.assert_array_equal(frames,np.arange(1,len(frames)+1))
        tracker=BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),frame_rate=25)
        rows=[]
        for frame in frames:
            lo,hi=data['offsets'][frame-1:frame+1];detection=np.c_[data['boxes'][lo:hi],data['scores'][lo:hi]].astype(np.float32)
            rows.extend((int(frame),track.tlwh.copy(),track.track_id,track.score)
                for track in tracker.update(detection,(1,1),(1,1)) if track.tlwh[2]*track.tlwh[3]>100)
        path_out=output/'track_results'/(sequence+'.txt');write_sequence_results(str(path_out),rows)
        sources[sequence]=dict(path=str(path),sha256=sha256(path));coverage[sequence]=dict(frames=len(frames),detections=len(data['boxes']),output_rows=len(rows),output_sha256=sha256(path_out))
    manifest=dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION',config=config,config_sha256=sha256(config_path),plan_sha256=sha256(plan_path),
        split=args.split,sequences=sequences,frames=sum(value['frames'] for value in coverage.values()),sources=sources,coverage=coverage,
        GT_inference_input=False,future_frame_input=False,previous_outputs_rewritten=False,statistics={},replay_seconds=time.perf_counter()-started,
        runtime_scope='Cached proposal replay; shared ROI production and global detection excluded, not end-to-end FPS.',
        replay_source_sha256=sha256(__file__),byte_tracker_source_sha256=sha256(ROOT/'yolox/tracker/byte_tracker.py'))
    manifest['replay_FPS']=manifest['frames']/manifest['replay_seconds'];dump(output/'inference_manifest.json',manifest)
    metrics=run_trackeval(str(output/'track_results'),plan['evaluation']['GT_root'],str(output),sequences,
        plan['evaluation']['TrackEval_root'],data_split='train' if args.split=='dev' else 'test')
    dump(output/'metrics.json',dict(status='COMPLETE',trackeval=metrics,manifest=manifest))
    print('FOVEATED_REPLAY_COMPLETE',json.dumps(metrics),flush=True)


if __name__=='__main__':main()
