#!/usr/bin/env python3
"""Fixed-cache deployment risk replay for full online output parity, not FPS."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
import torch
from cfp.online import CurrentFrameFrontend
from cfp.detector_runtime_device_pad import DevicePadDetectorRuntime
from cfp.predictive import LatentIdentityPredictor
from cfp.recoverability_ablation import risk_model_from_config
from cfp.online_recovery_write_quota import install_quota_risk_memory
from cfp.lifecycle_crossover import load_bot_reference, make_cfp_lifecycle
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-frames',type=int,default=0)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve online evidence')
    config=json.loads(args.config.read_text())
    protocol_path=Path('configs/cfp/protocol_v2.json')
    protocol=json.loads(protocol_path.read_text())
    if config['state']!='FROZEN' or config['protocol_sha256']!=sha256(protocol_path) or config['sequences']!=protocol['dev'] or args.max_frames<0:raise ValueError('Frozen canonical protocol required')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.manual_seed(config['seed'])
    start_free=shutil.disk_usage(args.output.parent).free
    def reserve():
        free=shutil.disk_usage(args.output.parent).free
        if free<40*2**30 or start_free-free>config.get('output_budget_gib',2)*2**30:raise OSError('40GiB reserve/output budget')
        return free
    reserve()
    checkpoint=Path(config['predictor_checkpoint'])
    metadata=json.loads(checkpoint.with_name('manifest.json').read_text())
    if metadata['status']!='COMPLETE' or sha256(checkpoint)!=metadata['checkpoint_sha256']:
        raise ValueError('Canonical ordinary GRU drift')
    encoder_metadata=json.loads((Path(metadata['features_root'])/'manifest.json').read_text())['rgb_encoder']
    if encoder_metadata['checkpoint_sha256']!=sha256(config['rgb_checkpoint']):
        raise ValueError('Canonical RGB coordinate-space drift')
    model=LatentIdentityPredictor().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
    risk_checkpoint=Path(config['risk_checkpoint'])
    risk_metadata=json.loads(risk_checkpoint.with_name('manifest.json').read_text())
    if risk_metadata['status']!='COMPLETE' or sha256(risk_checkpoint)!=risk_metadata['inference_sha256']:
        raise ValueError('Deployment risk export drift')
    if risk_metadata['config']['validation_video'] is not None or set(risk_metadata['config']['fit_videos']) & set(config['sequences']) or risk_metadata['config']['calibration_video'] in config['sequences']:
        raise ValueError('Deployment dev references must not enter risk fitting/calibration')
    query_metadata=json.loads((Path(risk_metadata['config']['queries'])/'manifest.json').read_text())
    if query_metadata['historical_gru_checkpoint_sha256']!=sha256(checkpoint) or query_metadata['seed']!=config['seed']:
        raise ValueError('Deployment risk producer/coordinate seed differs')
    for source,digest in risk_metadata['guard']['sources'].items():
        if sha256(source)!=digest:raise ValueError('Deployment risk source drift: '+source)
    saved=torch.load(risk_checkpoint,map_location='cpu',weights_only=False)
    estimator=risk_model_from_config(saved['config']).eval()
    estimator.load_state_dict(saved['model'],strict=True)
    calibration=saved['calibration']
    native=load_bot_reference('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT')
    root=Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    detection_root=root/'phase0/causal_v3_dev_detections'
    feature_root=root/'phase1'/f'causal_dev_detection_features_postneck_seed{config["seed"]}'
    dm=json.loads((detection_root/'manifest.json').read_text())
    fm=json.loads((feature_root/'manifest.json').read_text())
    if fm['rgb_encoder']['checkpoint_sha256']!=sha256(config['rgb_checkpoint']):
        raise ValueError('Frozen encoder coordinate drift')
    args.output.mkdir()
    sources=[__file__,args.config,protocol_path,'cfp/online_recovery_write_quota.py','cfp/online_recovery_write_isolated.py',
             'cfp/predictive.py','cfp/lifecycle_crossover.py','cfp/recoverability.py','cfp/recoverability_ablation.py',
             'cfp/candidate_graph.py','cfp/association.py','cfp/predictive_association.py','cfp/history.py']
    report=dict(status='RUNNING',config=config,config_sha256=sha256(args.config),videos={},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_snapshots=snapshot_sources(args.output,sources),
        ordinary_checkpoint_sha256=sha256(checkpoint),risk_checkpoint_sha256=sha256(risk_checkpoint),
        detection_manifest_sha256=sha256(detection_root/'manifest.json'),feature_manifest_sha256=sha256(feature_root/'manifest.json'),
        scope='Fixed frozen detection/feature replay for canonical deployment risk method; no new training/tuning. No cache FPS claimed. Exposed dev references and warm producer fitting exposure disclosed.')
    write_json(args.output/'manifest.json',report)
    for seq in config['sequences']:
        dp,fp=detection_root/(seq+'.npz'),feature_root/(seq+'.npz')
        provenance=json.loads(fp.with_suffix('.provenance.json').read_text())
        if sha256(dp)!=dm['caches'][seq]['sha256'] or sha256(fp)!=provenance['features_sha256']:
            raise ValueError('Frozen cache drift')
        with np.load(dp,allow_pickle=False) as bank:
            frames,offsets,boxes,scores=[bank[key] for key in ('frame_ids','offsets','boxes','scores')]
        with np.load(fp,allow_pickle=False) as bank:features=bank['rgb']
        BaseTrack._count=0
        tracker=make_cfp_lifecycle(native,'gru',model,config['appearance_weight'],memory_mode='predictor')
        tracker=install_quota_risk_memory(tracker,estimator,calibration,fraction=config['reject_fraction'],seed=config['risk_initialization_seed'])
        rows=0
        with (args.output/(seq+'.txt')).open('x') as stream:
            for i,frame in enumerate(frames):
                if i%128==0:reserve()
                lo,hi=offsets[i:i+2]
                output=tracker.update(np.c_[boxes[lo:hi],scores[lo:hi]],features[lo:hi].copy(),np.arange(lo,hi))
                if len({int(t.track_id) for t in output})!=len(output):raise ValueError('Duplicate current identity')
                for track in output:
                    if np.prod(track.tlwh[2:])<=100:continue
                    index=track.current_proposal_index
                    if track.frame_id!=frame or not lo<=index<hi:raise ValueError('Current measured output required')
                    measured=boxes[index].copy();measured[2:]-=measured[:2]
                    stream.write(f'{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n');rows+=1
        report['videos'][seq]=dict(frames=len(frames),output_rows=rows,prediction_sha256=sha256(args.output/(seq+'.txt')),
            risk_memory_counts=tracker.phase_vi_memory_counts,quota_counts=tracker.quota_counts)
        write_json(args.output/'manifest.json',report)
        print(seq,report['videos'][seq],flush=True)
    report['status']='COMPLETE';write_json(args.output/'manifest.json',report)


if __name__=='__main__':main()
