#!/usr/bin/env python3
"""Guarded raw full-six-video runtime campaign; no cached inference inputs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def active_raw_process(output,tool):
    for line in subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines():
        parts=line.strip().split(None,1)
        if len(parts)==2 and parts[1].startswith('/data3/liuyeqiang/.venvs/cfp-fishmot/bin/python') and ('python '+tool) in parts[1] and ('--output '+str(output)+' ') in parts[1]:
            return int(parts[0])
    return None


def main():
    ap=argparse.ArgumentParser()
    for name in ['root','state']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--kind',choices=['legacy_gru','dino_frozen_fifo'],required=True);a=ap.parse_args()
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') not in ['0','2']:raise ValueError('NewapprovedGPUworker required')
    config=Path('configs/cfp/phase_vii_whole_raw_current_profile_v1.json');c=json.loads(config.read_text());tool='tools/cfp_profile_raw_current_video_v2.py' if a.kind=='legacy_gru' else 'tools/cfp_profile_raw_current_video.py';version='v2' if a.kind=='legacy_gru' else 'v1';paths=[__file__,tool,str(config),'cfp/raw_profile_runtime.py','tools/cfp_detector_worker_raw_profile.py','cfp/detector_runtime.py','cfp/detector_runtime_device_pad.py','cfp/online.py','cfp/pseudo_ir.py','cfp/reid.py','cfp/dinov2_native_encoder.py','cfp/dinov2_identity.py','cfp/dinov2_fish_encoder.py','cfp/native_local_identity.py','cfp/lifecycle_crossover.py','cfp/predictive.py','cfp/association.py','cfp/predictive_association.py','cfp/history.py','cfp/provenance.py'];hashes={p:sha256(p) for p in paths};tasks=c['videos'];state=dict(status='RUNNING',pid=os.getpid(),kind=a.kind,tasks=tasks,completed=[],active=None,sources=snapshot_sources(a.state.with_suffix('.sources'),paths),source_guards=hashes,scope=c['scope']);write_json(a.state,state)
    try:
        for video in tasks:
            if any(sha256(p)!=v for p,v in hashes.items()):raise ValueError('Active rawpipeline source changed')
            out=a.root/f'raw_current_{a.kind}_{video}_{version}';state['active']=dict(output=str(out));write_json(a.state,state)
            if out.exists() or active_raw_process(out,tool):
                deadline=time.monotonic()+21600
                while True:
                    if not (out/'manifest.json').exists():
                        pid=active_raw_process(out,tool)
                        if pid is None:raise RuntimeError('Actualownedproducer endedbefore manifestcreation; inspectlog')
                        state.update(status='WAITING_ACTUAL_RAW_PRODUCER',active=dict(output=str(out),verified_existing_pid=pid));write_json(a.state,state);time.sleep(10);continue
                    m=json.loads((out/'manifest.json').read_text());status=m['status']
                    if status=='COMPLETE_WHOLE_RAW_CURRENT_VIDEO_PROFILE':break
                    if status.startswith('FAILED'):raise RuntimeError('Actualrawproducer failed, inspectinstead ofrestart')
                    if time.monotonic()>deadline:raise TimeoutError('Actualrawproducer wait exceeded6hours')
                    state['status']='WAITING_ACTUAL_RAW_PRODUCER';write_json(a.state,state);time.sleep(10)
                # Its stop acknowledgement may precede CUDA teardown. Poll the
                # specifically matched owned raw process, never restart it.
                while active_raw_process(out,tool):time.sleep(2)
            else:
                state['status']='RUNNING';write_json(a.state,state)
                with Path(str(out)+'.log').open('x') as log:
                    p=subprocess.Popen([sys.executable,tool,'--config',str(config),'--root',str(a.root),'--output',str(out),'--video',video,'--kind',a.kind],stdout=log,stderr=subprocess.STDOUT);state['active']['pid']=p.pid;write_json(a.state,state);code=p.wait()
                if code:raise RuntimeError('Actualrawpipeline exited '+str(code))
                m=json.loads((out/'manifest.json').read_text())
            if m['status']!='COMPLETE_WHOLE_RAW_CURRENT_VIDEO_PROFILE' or m['config_sha256']!=hashes[str(config)] or m['cached_detection_or_feature_used_for_inference'] or m['loaded_GT_annotation_labels'] or m['precomputed_IR_used_for_inference']:raise ValueError('ActualfullcurrentRGB-only profile required')
            if m['reference_parity']['proposal_byte_equal_frames']!=m['frames'] or m['reference_parity']['current_IR_pixel_equal_frames']!=m['frames']:raise ValueError('Allframeinput/proposal parity required')
            evaluation=Path(str(out)+'.trackeval');evaluation.mkdir(exist_ok=False);parent=Path('configs/cfp/protocol_v2.json');protocol=json.loads(parent.read_text());protocol.update(train=[video],dev=[],historical_test=[],role='Offline scoring ofcompletedfullraw current-only profile; nooutertuning');write_json(evaluation/'protocol.json',protocol)
            cmd=[sys.executable,'tools/cfp_evaluate.py','--data-root',c['data_root'],'--derived-gt-root','/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt','--protocol',str(evaluation/'protocol.json'),'--partition','train','--predictions',str(out),'--output',str(evaluation),'--trackeval-root','/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
            with (evaluation/'evaluation.log').open('x') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError('Actualrawofficial evaluation failed')
            metrics=json.loads((evaluation/'metrics.json').read_text());assert metrics['artifacts'][video]['prediction_sha256']==m['prediction_sha256']
            state['completed'].append(dict(output=str(out),manifest_sha256=sha256(out/'manifest.json'),whole_raw_FPS=m['whole_raw_FPS'],observed_with_validation_FPS=m['frames']/m['elapsed_with_validation_seconds'],cache_reference_prediction_byte_equal=m['cache_reference_prediction_byte_equal'],official_metrics=str(evaluation/'metrics.json'),metrics_sha256=sha256(evaluation/'metrics.json'),metrics=metrics['metrics'][video]));state['active']=None;write_json(a.state,state);print(json.dumps(dict(kind=a.kind,video=video,completed=len(state['completed']),FPS=m['whole_raw_FPS'])),flush=True)
        state['status']='COMPLETE_WHOLE_RAW_SIX_VIDEO_PROFILE';write_json(a.state,state)
    except Exception as e:state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
