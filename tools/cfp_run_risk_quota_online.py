#!/usr/bin/env python3
"""Exact device padding and RGB-only crop preparation; same frozen online inputs/model."""
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
    if torch.cuda.device_count()!=1:raise ValueError('One explicitly visible physical GPU')
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.cuda.set_per_process_memory_fraction(.5)
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
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
    frontend=CurrentFrameFrontend(config['detector_checkpoint'],config['scripts_root'],rgb_checkpoint=config['rgb_checkpoint'],
      detector_batch=8,protocol_sha256=sha256(protocol_path),input_regime='causal',preprocessing='lookup_uint8',detector_runtime_python=config['detector_runtime_python'], detector_runtime_factory=DevicePadDetectorRuntime, skip_unused_ir_crops=True)
    args.output.mkdir()
    sources=[__file__,'cfp/recoverability.py','cfp/recoverability_ablation.py','cfp/candidate_graph.py','cfp/online_recovery_write_isolated.py','cfp/online_recovery_write_quota.py','cfp/predictive.py','cfp/lifecycle_crossover.py','cfp/association.py','cfp/predictive_association.py','cfp/history.py',
      'cfp/online.py','cfp/pseudo_ir.py','cfp/reid.py','cfp/identity_data.py','cfp/detector_runtime.py','tools/cfp_detector_worker_device_pad.py','cfp/detector_runtime_device_pad.py',args.config,protocol_path]
    report=dict(status='RUNNING',probe=bool(args.max_frames),config=config,config_sha256=sha256(args.config),protocol_sha256=sha256(protocol_path),
      source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_snapshots=snapshot_sources(args.output,sources),
      ordinary_manifest_sha256=sha256(checkpoint.with_name('manifest.json')),risk_manifest_sha256=sha256(risk_checkpoint.with_name('manifest.json')),inference_teacher_removed=True,
      rgb_encoder=frontend.rgb_metadata,detector_runtime=frontend.detector_worker.metadata,
      runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')),
      parameters=dict(ordinary=sum(p.numel() for p in model.parameters()),risk=risk_metadata['parameters'],detector=frontend.detector_parameters,rgb_encoder=sum(p.numel() for p in frontend.rgb_encoder.parameters())),
      starting_free_bytes=start_free,reserve_gib=40,videos={},scope='Full online risk method: decode+causal IR generation+V3/NMS+current RGB crops/encoding+canonical ordinary GRU+matched10% risk write quota+association+output. No detector/feature cache, GT, future or teacher. Current frame copied8times in detector; canonical finalSN013 batch2 numerical difference remains disclosed. Synthetic IR feeds detector only. Shared GPU load, not dedicated deployment timing.')
    write_json(args.output/'manifest.json',report)
    try:
        for seq in config['sequences']:
            frontend.reset();BaseTrack._count=0
            tracker=make_cfp_lifecycle(native,'gru',model,config['appearance_weight'],memory_mode='predictor')
            tracker=install_quota_risk_memory(tracker,estimator,calibration,fraction=config['reject_fraction'],seed=config['risk_initialization_seed'])
            paths=sorted((Path(config['data_root'])/'train'/seq/'visible').glob('*.jpg'))
            if not paths or any(path.stem!=f'{i:06d}' for i,path in enumerate(paths,1)):raise ValueError('Consecutive current frames required')
            if args.max_frames:paths=paths[:args.max_frames]
            times=[];stage_rows={};rows=0;proposal_count=0
            prediction=args.output/(seq+'.txt')
            with prediction.open('x') as output,(args.output/(seq+'.timing.jsonl')).open('x') as timing:
                for frame,path in enumerate(paths,1):
                    if frame%128==1:reserve()
                    started=time.perf_counter();rgb=cv2.imread(str(path))
                    if rgb is None:raise ValueError('Current frame decode failed')
                    stages=dict(decode=time.perf_counter()-started)
                    result=frontend.process(rgb);stages.update(result['stage_seconds'])
                    lo=proposal_count;hi=lo+len(result['boxes']);proposal_count=hi
                    t=time.perf_counter()
                    proposals=np.concatenate([result['boxes'],result['scores'][:,None]],axis=1)
                    targets=tracker.update(proposals,result['rgb_features'],np.arange(lo,hi))
                    stages['student_and_association']=time.perf_counter()-t;t=time.perf_counter()
                    if len({int(x.track_id) for x in targets})!=len(targets):raise ValueError('Duplicate online ID')
                    for target in targets:
                        if np.prod(target.tlwh[2:])<=100:continue
                        index=target.current_proposal_index
                        if target.frame_id!=frame or not lo<=index<hi:raise ValueError('Output must be a current accepted measurement')
                        box=result['boxes'][index-lo].copy();box[2:]-=box[:2]
                        output.write(f'{frame},{target.track_id},{box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f},{float(target.score):.6f},-1,-1,-1\n');rows+=1
                    stages['output']=time.perf_counter()-t
                    elapsed=time.perf_counter()-started;times.append(elapsed)
                    for name,value in stages.items():stage_rows.setdefault(name,[]).append(value)
                    timing.write(json.dumps(dict(frame=frame,total_seconds=elapsed,stage_seconds=stages))+'\n')
                    if frame%128==0:print(seq,frame,len(paths),round(elapsed,3),'seconds',flush=True)
            report['videos'][seq]=dict(frames=len(paths),output_rows=rows,prediction_sha256=sha256(prediction),timing_sha256=sha256(args.output/(seq+'.timing.jsonl')),
              pipeline_seconds=sum(times),fps=len(times)/sum(times),latency_p50_p95_ms=(np.quantile(times,[.5,.95])*1000).tolist(),
              stage_latency_p50_p95_ms={k:(np.quantile(v,[.5,.95])*1000).tolist() for k,v in stage_rows.items()},write_counts=tracker.write_counts,risk_memory_counts=tracker.phase_vi_memory_counts,quota_counts=tracker.quota_counts)
            write_json(args.output/'manifest.json',report)
        report.update(status='COMPLETE',ending_free_bytes=reserve(),peak_parent_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        report.update(status='FAILED',error=str(error));raise
    finally:
        report['detector_runtime']=frontend.close();write_json(args.output/'manifest.json',report)


if __name__=='__main__':main()
