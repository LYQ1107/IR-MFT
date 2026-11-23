#!/usr/bin/env python3
"""Frozen original graph, ROI-confirmed rescue, GT evaluated afterward."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from tools.eval_mft26ir import run_trackeval,write_sequence_results
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.roi_confirmed_recovery import ROIConfirmedRecovery


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    config_path=Path(a.config);config=json.loads(config_path.read_text())
    plan_path=ROOT/'configs/perceptual_mft_20261122.json';plan=json.loads(plan_path.read_text());work=Path(plan['work_root'])
    research_path=ROOT/'configs/perceptual_roi_confirmed_recovery_20261123.json'
    if config['recovery_plan_sha256']!=sha256(research_path):raise ValueError('Registered recovery plan changed')
    protocol=json.loads((ROOT/plan['protocol']).read_text());output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if(output/'metrics.json').exists():raise ValueError('Completed recovery accuracy exists')
    arm=config['detection_arm'];sources={};coverage={};statistics={};started=time.perf_counter()
    for sequence in protocol['dev']:
        feature=work/'features/dev'/sequence/(arm+'.npz');feature_meta=json.loads((feature.parent/'complete.json').read_text())
        point=work/'transport/dev'/sequence/(arm+'.npz');point_meta=json.loads((point.parent/'complete.json').read_text())
        if (feature_meta['status']!='COMPLETE'or point_meta['status']!='COMPLETE'
                or feature_meta['outputs'][arm]['sha256']!=sha256(feature)or point_meta['outputs'][arm]['sha256']!=sha256(point)):
            raise ValueError('Frozen complete original evidence required')
        with np.load(feature)as z:data={key:z[key]for key in z.files}
        with np.load(point)as z:transport={key:z[key]for key in z.files}
        np.testing.assert_array_equal(transport['detection_offsets'],data['offsets'])
        data['foreground_transport']=transport;data['ROI_confirmation']=np.zeros(len(data['boxes']),np.float32)
        roi_source=None
        if not config.get('disable_ROI_confirmation',False):
            directory=work/'foveated_priority_regions_proposals/dev'/sequence
            meta=json.loads((directory/'complete.json').read_text());key=f'{arm}_rgb_half_k4_priority';info=meta['outputs'][key]
            roi_source=Path(info['path'])
            if meta['GT_read']or info['sha256']!=sha256(roi_source):raise ValueError('Immutable ROI evidence mismatch')
            with np.load(roi_source)as z:proposal={key:z[key]for key in z.files}
            mapping=proposal['original_index'];known=mapping>=0
            np.testing.assert_array_equal(proposal['boxes'][known],data['boxes'][mapping[known]])
            if sorted(mapping[known].tolist())!=list(range(len(data['boxes']))):raise ValueError('Every original row must remain once')
            data['ROI_confirmation'][mapping[known]]=proposal['scores'][known]
        tracker=ROIConfirmedRecovery(config);rows=[]
        for frame in data['frame_ids']:rows.extend(tracker.update(int(frame),data))
        target=output/'track_results'/(sequence+'.txt');write_sequence_results(str(target),rows)
        sources[sequence]=dict(features=str(feature),feature_sha256=sha256(feature),points=str(point),point_sha256=sha256(point),
                               ROI_source=None if roi_source is None else str(roi_source),ROI_sha256=None if roi_source is None else sha256(roi_source))
        coverage[sequence]=dict(frames=len(data['frame_ids']),detections=len(data['boxes']),output_rows=len(rows),output_sha256=sha256(target))
        statistics[sequence]=tracker.statistics
    manifest=dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION',config=config,config_sha256=sha256(config_path),
        plan_sha256=sha256(plan_path),split='dev',sequences=protocol['dev'],frames=sum(x['frames']for x in coverage.values()),
        sources=sources,coverage=coverage,statistics=statistics,GT_inference_input=False,future_frame_input=False,
        previous_outputs_rewritten=False,original_confidence_partition_preserved=True,
        original_candidate_hypotheses_preserved=True,primary_before_rescue=True,calibration=tracker.calibration_metadata,
        auxiliary_calibration=tracker.auxiliary_metadata,primary_checkpoint_sha256=sha256(config['association_checkpoint']),
        auxiliary_checkpoint_sha256=sha256(config['auxiliary_checkpoint']),replay_source_sha256=sha256(__file__),
        recovery_source_sha256=sha256(ROOT/'yolox/tracker/roi_confirmed_recovery.py'),
        original_recovery_source_sha256=sha256(ROOT/'yolox/tracker/weak_recovery_tracker.py'),replay_seconds=time.perf_counter()-started)
    manifest['replay_FPS']=manifest['frames']/manifest['replay_seconds']
    manifest['replay_FPS_scope']='Cached evidence; ROI/global detector/encoder extraction excluded, not end-to-end FPS.'
    dump(output/'inference_manifest.json',manifest)
    metrics=run_trackeval(str(output/'track_results'),plan['evaluation']['GT_root'],str(output),protocol['dev'],
                          plan['evaluation']['TrackEval_root'],data_split='train')
    dump(output/'metrics.json',dict(status='COMPLETE',trackeval=metrics,manifest=manifest))
    print('ROI_RECOVERY_COMPLETE',json.dumps(metrics),flush=True)


if __name__=='__main__':main()
