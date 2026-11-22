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
                       cycle_tracker=sha256(ROOT/'yolox/tracker/cycle_candidate_tracker.py'),
                       weak_tracker=sha256(ROOT/'yolox/tracker/weak_recovery_tracker.py'),
                       calibrated_tracker=sha256(ROOT/'yolox/tracker/calibrated_foreground_tracker.py'),
                       strong_RGB_tracker=sha256(ROOT/'yolox/tracker/strong_rgb_tracker.py'),
                       prebirth_tracker=sha256(ROOT/'yolox/tracker/prebirth_identity_tracker.py'),
                       missing_point_tracker=sha256(ROOT/'yolox/tracker/missing_point_tracker.py'),
                       guarded_tracker=sha256(ROOT/'yolox/tracker/guarded_rgb_tracker.py'),
                       native_query_tracker=sha256(ROOT/'yolox/tracker/native_query_tracker.py'),
                       native_query_features=sha256(ROOT/'yolox/tracker/native_query_features.py'),
                       native_query_loader=sha256(ROOT/'tools/native_query_replay.py'),
                       counterfactual_query_tracker=sha256(ROOT/'yolox/tracker/counterfactual_query_tracker.py'),
                       RGB_context=sha256(ROOT/'yolox/tracker/rgb_reference_context.py'),
                       association_model=sha256(ROOT/'yolox/tracker/association_calibrator.py'),
                       association_features=sha256(ROOT/'yolox/tracker/association_features.py'),
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
    raw_backends={'byte','foreground_byte','candidate_foreground','residual_foreground','cycle_foreground','recoverable_foreground'}
    learned_backends={'calibrated_foreground','calibrated_cycle','recoverable_calibrated','strong_RGB_foreground','prebirth_strong_RGB','missing_point_strong_RGB','guarded_strong_RGB','native_query','counterfactual_query'}
    point_backends=(raw_backends-{'byte'})|learned_backends
    for sequence in sequences:
        if config.get('backend') in raw_backends:
            source_key=('new_diffv2ir' if arm=='new' else 'rgb')+'_'+args.split
            source=Path(plan['frozen_detection_sources'][source_key])/'detections'/f'{sequence}.npz'
        else:
            source=Path(plan['work_root'])/'features'/args.split/sequence/f'{arm}.npz'
            manifest=json.loads((source.parent/'complete.json').read_text())
            if manifest['status']!='COMPLETE':raise ValueError('Incomplete feature producer')
            if sha256(source)!=manifest['outputs'][arm]['sha256']:raise ValueError('Feature digest mismatch')
        with np.load(source) as z:data={k:z[k] for k in z.files}
        transport_source=None
        if config.get('backend') in point_backends:
            variant=config.get('transport_variant','raw')
            if variant not in {'raw','registered'}:raise ValueError('Unknown transport producer variant')
            directory=Path(plan['work_root'])/('transport_registered' if variant=='registered' else 'transport')/args.split/sequence
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
        native_evidence=None
        if config.get('backend') in {'native_query','counterfactual_query'}:
            from tools.native_query_replay import load_native_query_evidence
            native_evidence=load_native_query_evidence(data,plan['work_root'],args.split,sequence,config)
        if config.get('backend')=='byte':
            tracker=BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),frame_rate=25)
        elif config.get('backend')=='safe_byte':
            from yolox.tracker.evidence_byte_tracker import EvidenceByteTracker
            tracker=EvidenceByteTracker(config)
        elif config.get('backend')=='foreground_byte':
            from yolox.tracker.foreground_byte_tracker import ForegroundByteTracker
            tracker=ForegroundByteTracker(config)
        elif config.get('backend')=='candidate_foreground':
            from yolox.tracker.candidate_foreground_tracker import CandidateForegroundTracker
            tracker=CandidateForegroundTracker(config)
        elif config.get('backend')=='residual_foreground':
            from yolox.tracker.residual_foreground_tracker import ResidualForegroundTracker
            tracker=ResidualForegroundTracker(config)
        elif config.get('backend')=='cycle_foreground':
            from yolox.tracker.cycle_candidate_tracker import CycleForegroundTracker
            tracker=CycleForegroundTracker(config)
        elif config.get('backend')=='recoverable_foreground':
            from yolox.tracker.weak_recovery_tracker import RecoverableForegroundTracker
            tracker=RecoverableForegroundTracker(config)
        elif config.get('backend')=='strong_RGB_foreground':
            from yolox.tracker.strong_rgb_tracker import StrongRGBForegroundTracker
            tracker=StrongRGBForegroundTracker(config)
        elif config.get('backend')=='prebirth_strong_RGB':
            from yolox.tracker.prebirth_identity_tracker import PrebirthStrongRGBTracker
            tracker=PrebirthStrongRGBTracker(config)
        elif config.get('backend')=='missing_point_strong_RGB':
            from yolox.tracker.missing_point_tracker import MissingPointEvidenceTracker
            tracker=MissingPointEvidenceTracker(config)
        elif config.get('backend')=='guarded_strong_RGB':
            from yolox.tracker.guarded_rgb_tracker import GuardedStrongRGBTracker
            tracker=GuardedStrongRGBTracker(config)
        elif config.get('backend') in {'native_query','counterfactual_query'}:
            from yolox.tracker.native_query_tracker import NativeQueryTracker
            from yolox.tracker.counterfactual_query_tracker import CounterfactualQueryTracker
            tracker=(CounterfactualQueryTracker if config['backend']=='counterfactual_query' else NativeQueryTracker)(config)
            if native_evidence.get('alignment',{}).get('sha256',tracker.native_metadata['alignment_sha256'])!=tracker.native_metadata['alignment_sha256']:
                raise ValueError('Native alignment model mismatch')
        elif config.get('backend') in learned_backends:
            from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker,CalibratedCycleTracker,RecoverableCalibratedTracker
            factory={'calibrated_foreground':CalibratedForegroundTracker,'calibrated_cycle':CalibratedCycleTracker,'recoverable_calibrated':RecoverableCalibratedTracker}[config['backend']]
            tracker=factory(config)
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
        if native_evidence is not None:sources[sequence]['native_identity']=native_evidence
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
    if config.get('backend') in point_backends:
        manifest['evidence_tracker_source_sha256']=source_hashes['evidence_tracker']
        manifest['foreground_tracker_source_sha256']=source_hashes['foreground_tracker']
    if config.get('backend') in point_backends-{'foreground_byte'}:
        manifest['candidate_tracker_source_sha256']=source_hashes['candidate_tracker']
        manifest['residual_tracker_source_sha256']=source_hashes['residual_tracker']
    if config.get('backend') in {'cycle_foreground','recoverable_foreground','calibrated_cycle','recoverable_calibrated'}:
        manifest['cycle_tracker_source_sha256']=source_hashes['cycle_tracker']
    if config.get('backend') in {'recoverable_foreground','recoverable_calibrated'}:
        manifest['weak_tracker_source_sha256']=source_hashes['weak_tracker']
    if config.get('backend') in learned_backends:
        manifest['calibration']=tracker.calibration_metadata
        manifest['calibration_intervention']=dict(disable_learned_increment=config.get('disable_learned_increment',False),shuffle_cue=config.get('shuffle_cue',False),scope='Backend paired cue features only; detector proposals and RGB-reference features held fixed. Within-current-frame candidate shuffle, never future frames.')
        manifest['association_checkpoint_sha256']=sha256(config['association_checkpoint'])
        manifest['calibrated_tracker_source_sha256']=source_hashes['calibrated_tracker']
        manifest['association_model_source_sha256']=source_hashes['association_model']
        manifest['association_features_source_sha256']=source_hashes['association_features']
        if config.get('backend') in {'strong_RGB_foreground','prebirth_strong_RGB','missing_point_strong_RGB','guarded_strong_RGB'}:
            manifest['strong_RGB_tracker_source_sha256']=source_hashes['strong_RGB_tracker']
            manifest['RGB_context_source_sha256']=source_hashes['RGB_context']
            manifest['calibration_intervention']['scope']='Only the newly added cue residual. The full frozen fitted RGB scorer and detector proposals are retained; shuffle rolls auxiliary candidate features within the same current source-query frame, never future frames.'
        if config.get('backend') in {'prebirth_strong_RGB','missing_point_strong_RGB'}:
            manifest['prebirth_tracker_source_sha256']=source_hashes['prebirth_tracker']
        if config.get('backend')=='missing_point_strong_RGB':
            manifest['missing_point_tracker_source_sha256']=source_hashes['missing_point_tracker']
        if config.get('backend')=='guarded_strong_RGB':
            manifest['guarded_tracker_source_sha256']=source_hashes['guarded_tracker']
        if config.get('backend') in {'native_query','counterfactual_query'}:
            manifest['native_query']=tracker.native_metadata
            manifest['native_query_checkpoint_sha256']=sha256(config['native_query_checkpoint'])
            for name in ['native_query_tracker','native_query_features','native_query_loader','prebirth_tracker']:
                manifest[name+'_source_sha256']=source_hashes[name]
            if config.get('backend')=='counterfactual_query':
                manifest['counterfactual_query_tracker_source_sha256']=source_hashes['counterfactual_query_tracker']
            manifest['calibration_intervention']=dict(disable_native_increment=config.get('disable_native_increment',False),shuffle_native_cue=config.get('shuffle_native_cue',False),
                scope='Only native backend identity evidence; the complete frozen RGB scorer, real/null prior, RGB reference and detector proposals remain. Shuffle uses current-frame candidates only.')
            manifest['replay_FPS_scope']='Cached features only; detector, crop encoders and point extraction excluded. This is not end-to-end FPS.'
    dump(output/'inference_manifest.json',manifest)
    metrics=run_trackeval(str(output/'track_results'),plan['evaluation']['GT_root'],str(output),sequences,
                          plan['evaluation']['TrackEval_root'],data_split='test' if args.split=='test' else 'train')
    dump(output/'metrics.json',dict(status='COMPLETE',trackeval=metrics,manifest=manifest))
    print('REPLAY_COMPLETE',json.dumps(metrics),flush=True)


if __name__=='__main__':main()
