#!/usr/bin/env python3
"""Frozen causal replay; inference completes before the evaluator reads GT."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.perceptual_tracker import PerceptualTracker
from tools.eval_mft26ir import run_trackeval, write_sequence_results
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',default='configs/perceptual_mft_20261122.json')
    p.add_argument('--config',required=True)
    p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    source_hashes=dict(tracker=sha256(ROOT/'yolox/tracker/perceptual_tracker.py'),
                       byte_tracker=sha256(ROOT/'yolox/tracker/byte_tracker.py'),
                       evidence_tracker=sha256(ROOT/'yolox/tracker/evidence_byte_tracker.py'),
                       foreground_tracker=sha256(ROOT/'yolox/tracker/foreground_byte_tracker.py'),
                       candidate_tracker=sha256(ROOT/'yolox/tracker/candidate_foreground_tracker.py'),
                       residual_tracker=sha256(ROOT/'yolox/tracker/residual_foreground_tracker.py'),
                       evidence_adapter=sha256(ROOT/'yolox/tracker/evidence_adapter.py'),
                       replay=sha256(__file__))
    git_head_at_start=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    config_path=Path(args.config)
    config=json.loads(config_path.read_text())
    plan=json.loads(Path(args.plan).read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text())
    sequences=protocol['historical_test' if args.split=='test' else 'dev']
    output=Path(args.output)
    if (output/'metrics.json').exists():
        raise ValueError('Refusing to overwrite completed accuracy experiment')
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    sources,coverage,statistics={},{},{}
    adaptation={}
    arm=config.get('detection_arm','new')
    for sequence in sequences:
        if config.get('backend') in {'byte','foreground_byte','residual_foreground'}:
            source_key=('new_diffv2ir' if arm=='new' else 'rgb')+'_'+args.split
            source=Path(plan['frozen_detection_sources'][source_key])/'detections'/f'{sequence}.npz'
        else:
            source=Path(plan['work_root'])/'features'/args.split/sequence/f'{arm}.npz'
            manifest=json.loads((source.parent/'complete.json').read_text())
            if manifest['status']!='COMPLETE':raise ValueError('Incomplete feature producer')
            if sha256(source)!=manifest['outputs'][arm]['sha256']:raise ValueError('Feature digest mismatch')
        with np.load(source) as z:data={k:z[k] for k in z.files}
        transport_source=None
        if config.get('backend') in {'foreground_byte','residual_foreground'}:
            directory=Path(plan['work_root'])/'transport'/args.split/sequence
            transport_source=directory/f'{arm}.npz'
            producer=json.loads((directory/'complete.json').read_text())
            if producer['status']!='COMPLETE' or sha256(transport_source)!=producer['outputs'][arm]['sha256']:raise ValueError('Incomplete/mismatched point producer')
            with np.load(transport_source) as z:point_data={k:z[k] for k in z.files}
            if not np.array_equal(point_data['detection_offsets'],data['offsets']) or not np.array_equal(point_data['frame_ids'],data['frame_ids']):raise ValueError('Point/proposal indexing mismatch')
            data['foreground_transport']=point_data
        frames=data['frame_ids']
        if not np.array_equal(frames,np.arange(1,len(frames)+1)):raise ValueError('Missing/unordered frames')
        if data['offsets'][-1]!=len(data['boxes']):raise ValueError('Malformed detection indexing')
        if config.get('evidence_adapter'):
            from yolox.tracker.evidence_adapter import apply_adapter
            data,adapter_metadata=apply_adapter(data,config['evidence_adapter'])
            adaptation=dict(adapter_metadata,checkpoint_sha256=sha256(config['evidence_adapter']))
        if config.get('backend')=='byte':
            tracker=BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),frame_rate=25)
        elif config.get('backend')=='safe_byte':
            from yolox.tracker.evidence_byte_tracker import EvidenceByteTracker
            tracker=EvidenceByteTracker(config)
        elif config.get('backend')=='foreground_byte':
            from yolox.tracker.foreground_byte_tracker import ForegroundByteTracker
            tracker=ForegroundByteTracker(config)
        elif config.get('backend')=='residual_foreground':
            from yolox.tracker.residual_foreground_tracker import ResidualForegroundTracker
            tracker=ResidualForegroundTracker(config)
        else:tracker=PerceptualTracker(config)
        rows=[]
        for frame in frames:
            if config.get('backend')=='byte':
                a,b=data['offsets'][frame-1:frame+1]
                det=np.column_stack([data['boxes'][a:b],data['scores'][a:b]]).astype(np.float32)
                rows.extend((int(frame),t.tlwh.copy(),t.track_id,t.score) for t in tracker.update(det,(1,1),(1,1)) if t.tlwh[2]*t.tlwh[3]>100)
            else:rows.extend(tracker.update(int(frame),data))
        write_sequence_results(str(output/'track_results'/f'{sequence}.txt'),rows)
        sources[sequence]=dict(path=str(source),sha256=sha256(source))
        if transport_source is not None:sources[sequence]['foreground_transport']=dict(path=str(transport_source),sha256=sha256(transport_source))
        coverage[sequence]=dict(frames=len(frames),detections=len(data['boxes']),output_rows=len(rows),output_sha256=sha256(output/'track_results'/f'{sequence}.txt'))
        statistics[sequence]=getattr(tracker,'statistics',{})
        print(json.dumps(dict(sequence=sequence,**coverage[sequence])),flush=True)
    runtime=time.perf_counter()-started
    manifest=dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION',config=config,
                  config_path=str(config_path.resolve()),config_sha256=sha256(config_path),
                  plan_sha256=sha256(args.plan),split=args.split,sequences=sequences,
                  frames=sum(x['frames'] for x in coverage.values()),sources=sources,
                  coverage=coverage,statistics=statistics,replay_seconds=runtime,
                  replay_FPS=sum(x['frames'] for x in coverage.values())/runtime,
                  GT_inference_input=False,future_frame_input=False,previous_outputs_rewritten=False,
                  git_commit=git_head_at_start,
                  tracker_source_sha256=source_hashes['tracker'],
                  byte_tracker_source_sha256=source_hashes['byte_tracker'],
                  replay_source_sha256=source_hashes['replay'],CPU_threads=os.environ.get('OMP_NUM_THREADS'))
    if adaptation:
        manifest['evidence_adapter']=adaptation
        manifest['evidence_adapter_source_sha256']=source_hashes['evidence_adapter']
    if config.get('normalizer'):manifest['normalizer_sha256']=sha256(config['normalizer'])
    if config.get('backend')=='safe_byte':
        manifest['evidence_tracker_source_sha256']=source_hashes['evidence_tracker']
    if config.get('backend') in {'foreground_byte','residual_foreground'}:
        manifest['evidence_tracker_source_sha256']=source_hashes['evidence_tracker']
        manifest['foreground_tracker_source_sha256']=source_hashes['foreground_tracker']
    if config.get('backend')=='residual_foreground':
        manifest['candidate_tracker_source_sha256']=source_hashes['candidate_tracker']
        manifest['residual_tracker_source_sha256']=source_hashes['residual_tracker']
    dump(output/'inference_manifest.json',manifest)
    metrics=run_trackeval(str(output/'track_results'),plan['evaluation']['GT_root'],str(output),sequences,
                          plan['evaluation']['TrackEval_root'],data_split='test' if args.split=='test' else 'train')
    dump(output/'metrics.json',dict(status='COMPLETE',trackeval=metrics,manifest=manifest))
    print('REPLAY_COMPLETE',json.dumps(metrics),flush=True)


if __name__=='__main__':main()
