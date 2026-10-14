#!/usr/bin/env python3
"""All sixteen complete-six-video geometry controls, raw metrics/provenance."""
import argparse,json
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('Newimmutable complete report required')
    config=Path('configs/cfp/phase_vii_fish_geometry_controls_v1.json');c=json.loads(config.read_text());runs={};summary={};rows=[]
    for worker in [0,1]:
        state=json.loads((a.root/f'fish_geometry_worker{worker}_v1.json').read_text());assert state['status']=='COMPLETE_REGISTERED_SIX_VIDEO_FISH_GEOMETRY_CONTROLS';rows+=state['completed']
    if len(rows)!=16:raise ValueError('Exactall16controls required')
    for row in rows:
        path=Path(row['output']);m=json.loads((path/'manifest.json').read_text());metrics=Path(row['official_metrics_path']);official=json.loads(metrics.read_text())
        assert sha256(path/'manifest.json')==row['manifest_sha256'] and sha256(metrics)==row['official_metrics_sha256'];assert m['status']=='COMPLETE' and sum(v['frames'] for v in m['videos'].values())==27251
        for video,v in m['videos'].items():assert sha256(path/(video+'.txt'))==v['prediction_sha256']==official['artifacts'][video]['prediction_sha256']
        label='gru' if m['appearance_weight'] else 'geometry';kind=m['geometry_kind'];summary.setdefault(label,{})[kind]=dict(per_video=official['metrics'],threshold_crossings={v:x['geometry_counts']['native_threshold_crossings'] for v,x in m['videos'].items()},association_seconds={v:x['association_seconds'] for v,x in m['videos'].items()});runs[row['name']]=dict(row,scope=m['config']['scope'],manifest=dict(config=m['config'],detection_manifest_sha256=m['detection_manifest_sha256'],feature_manifest_sha256=m['feature_manifest_sha256'],baseline_checkpoint_sha256=m['baseline_checkpoint_sha256'],source_commit=m['source_commit'],source_snapshots=m['source_snapshots'],videos=m['videos']))
    proof=a.root/'fish_geometry_source_parity_v1.json';p=json.loads(proof.read_text());assert p['status']=='EXACT_LICENSED_GEOMETRY_FUNCTION_PARITY_VERIFIED'
    write_json(a.output,dict(status='IN_PROGRESS',completed_stage='COMPLETE_16_GEOMETRY_CONTROLS_ALL_SIX_NATIVE_VIDEOS',config=c,config_sha256=sha256(config),runs=runs,summary=summary,completed_configurations=16,complete_video_replays=96,native_frames_executed=16*27251,source_parity=dict(path=str(proof),sha256=sha256(proof),evidence=p),sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,config]),scope='Everyconfiguration all6nativeframes onidenticalfrozenV3/postneck512/GRUseed0/currentproposal rows; geometry-only vsGRU weights0/.3, nativeKF/lifecycle/lowstage/duplicate/measurementoutputs fixed. Deterministicgeometrycontrols conditionalonoldencoder/GRUexposure, not3independentnewencoderseeds. SourceFishcentraldefect/repair/continuousvsinclusiveIoU/ordinaryDIoU/finiteAGCv1/affine+.3/+0.05controls explicit. No sourcefulltracker/paperAGCIoU/nonlinearity/anatomy/groundtruthocclusion/novelty/rawFPS/finalgoal claim.'))
    print(json.dumps(dict(configurations=16,complete_videos=96,frames=16*27251)))

if __name__=='__main__':main()
