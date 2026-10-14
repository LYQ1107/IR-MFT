#!/usr/bin/env python3
"""Wait for actual crop fits, then current-frame production and 54 native controls."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);ap.add_argument('--worker',choices=['jpm','controls'],required=True);a=ap.parse_args()
    actual_proof=a.root/'native_jpm_encoder_actual_cpu_proof_v1.json'
    if json.loads(actual_proof.read_text())['status']!='VERIFIED_ACTUAL_CURRENT_JPM_ENCODER_AND_EXACT_PILOT_PIXELS':raise ValueError('Actual currentpixel/model coordinate proof required')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES');expected='1' if a.worker=='jpm' else '3'
    if gpu!=expected or a.state.exists():raise ValueError('New assigned GPU1/3 queue required')
    dependencies=[a.root/'transreid_dino_jpm_firstfold_worker_v1.json',a.root/'transreid_dino_jpm_controls_firstfold_worker_v1.json'];modes=['jpm'] if a.worker=='jpm' else ['global','cls_capacity'];videos=['BT-001','SN-015'];tasks=[(mode,seed,video) for mode in modes for seed in range(3) for video in videos]
    config=Path('configs/cfp/phase_vii_native_jpm_features_v1.json');replay_config=Path('configs/cfp/phase_vii_native_current_identity_replay_v1.json');dets=Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections');sources=[__file__,'tools/cfp_extract_native_jpm_features.py','tools/cfp_replay_native_jpm_identity.py','cfp/transreid_native_encoder.py','cfp/transreid_dino_jpm.py','third_party/transreid_jpm/shuffle.py','cfp/masked_identity_checkpoint.py','cfp/dinov2_native_encoder.py','cfp/dinov2_fish_encoder.py','cfp/dinov2_identity.py','cfp/identity_data.py','cfp/local_identity_encoder.py','cfp/native_local_identity.py','cfp/predictive_association.py','cfp/association.py','cfp/predictive.py',str(config),str(replay_config)];guards={p:sha256(p) for p in sources};state=dict(status='WAITING_ACTUAL_CROP_FIT_PROCESSES',pid=os.getpid(),physical_gpu=gpu,worker=a.worker,tasks=tasks,completed=[],active=None,dependencies=list(map(str,dependencies)),source_guards=guards,sources=snapshot_sources(a.state.with_suffix('.sources'),sources),scope='Firstfold3modes/3seeds/twoexcluded fullnativevideos=18currentfeaturebanks/54last-FIFO-EMA officialcontrols, fixedmemorysettings. Onlyexactpastmemory; all6outer/finalmethod/rawFPS/sourceotherbaselines stillrequired. No tuning fromexposedoutermetrics.')
    write_json(a.state,state)
    def alive(pid):
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    def check_sources():
        if any(sha256(p)!=h for p,h in guards.items()):raise ValueError('Native guarded source changed')
    def remaining_bytes():
        total=0
        for mode in ['global','jpm','cls_capacity']:
            dim=384 if mode=='global' else 1920
            for seed in range(3):
                for video in videos:
                    path=a.root/f'native_jpm_features_{mode}_{video}_s{seed}_v1';f=path/'manifest.json'
                    if f.exists() and json.loads(f.read_text()).get('status')=='COMPLETE_NATIVE_CURRENT_JPM_FEATURES':continue
                    with np.load(dets/(video+'.npz'),allow_pickle=False) as data:total+=len(data['boxes'])*(dim*4+1)+len(data['offsets'])*8+len(data['frame_ids'])*8
        return total
    def run(cmd,output,phase,cpu=False):
        check_sources();log=Path(str(output)+'.'+phase+'.log');state.update(status='RUNNING_NATIVE_JPM_CONTROLS',active=dict(output=str(output),phase=phase,command=cmd,log=str(log)));write_json(a.state,state)
        with log.open('x') as stream:
            child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else gpu));state['active']['pid']=child.pid;write_json(a.state,state);code=child.wait()
        if code:raise RuntimeError('Actual native child failed: '+str(log))
        state['active']=None;write_json(a.state,state)
    state['actual_deployment_proof_sha256']=sha256(actual_proof)
    state['estimated_total_feature_bytes']=remaining_bytes()
    write_json(a.state,state)
    try:
        while True:
            check_sources();deps=[json.loads(p.read_text()) for p in dependencies]
            if any(d['status'].startswith(('HALTED','FAILED')) for d in deps):raise ValueError('Actual prerequisite fit failed; no launch')
            if all(d['status']=='COMPLETE_REGISTERED_JPM_FIRST_FOLD_WORKER' and not alive(d['pid']) for d in deps):break
            if any(d['status']!='COMPLETE_REGISTERED_JPM_FIRST_FOLD_WORKER' and not alive(d['pid']) for d in deps):raise ValueError('Incomplete prerequisite has missing process; no synthetic completion orrestart')
            state['verified_wait_pids']=[d['pid'] for d in deps if alive(d['pid'])];write_json(a.state,state);time.sleep(20)
        parent=Path('configs/cfp/phase_vii_joint_evaluation_subset_v1.json');base=json.loads(parent.read_text())
        for mode,seed,video in tasks:
            check_sources();feature=a.root/f'native_jpm_features_{mode}_{video}_s{seed}_v1';training=a.root/f'transreid_dino_{mode}_SN-015_s{seed}_v1'
            if shutil.disk_usage(a.root).free-remaining_bytes()<41*2**30:raise OSError('All18plannedfeaturebanks plus40GiBreserve/1GiBrawmargin unavailable')
            # Explicit fit process teardown and actual idle check immediately before use.
            for attempt in range(15):
                memory,util=map(int,subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','))
                if memory<=64 and util==0:break
                time.sleep(2)
            else:raise ValueError('Assigned GPU notactuallyidle; stop withoutinterference')
            cmd=[sys.executable,'tools/cfp_extract_native_jpm_features.py','--config',str(config),'--data-root','/data3/liuyeqiang/datasets/mft26-ir-causal-v1','--frame-index',str(a.root/'identity_two_crop_index_v1'),'--detections',str(dets),'--reference','/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2','--weights',str(a.root/'external_models/dinov2_vits14_v1/dinov2_vits14_pretrain.pth'),'--source-proof',str(a.root/'dinov2_small_inference_source_smoke_v1.json'),'--training',str(training),'--output',str(feature),'--video',video]
            run(cmd,feature,'features');fm=json.loads((feature/'manifest.json').read_text());tm=json.loads((training/'manifest.json').read_text())
            if fm['status']!='COMPLETE_NATIVE_CURRENT_JPM_FEATURES' or fm['encoder']['coordinate_sha256']!=tm['checkpoint_sha256']:raise ValueError('Native producer coordinate differs')
            results=[]
            for memory in ['last','fifo','ema']:
                out=a.root/f'native_jpm_identity_{mode}_{video}_{memory}_s{seed}_v1';cmd=[sys.executable,'tools/cfp_replay_native_jpm_identity.py','--config',str(replay_config),'--features',str(feature),'--detections',str(dets),'--output',str(out),'--video',video,'--memory',memory];run(cmd,out,'replay',True)
                rm=json.loads((out/'manifest.json').read_text());prediction=out/(video+'.txt')
                if rm['status']!='COMPLETE_NATIVE_JPM_IDENTITY_REPLAY' or rm['completed_frames']!=fm['frames'] or sha256(prediction)!=rm['prediction_sha256']:raise ValueError('Exact complete native replay required')
                evaluation=Path(str(out)+'.trackeval');evaluation.mkdir();protocol=dict(base,train=[video],dev=[],historical_test=[],role='Offline completed sourceJPM native controls, no model/threshold/seed selection',parent_protocol_sha256=sha256(parent));write_json(evaluation/'protocol.json',protocol)
                cmd=[sys.executable,'tools/cfp_evaluate.py','--data-root','/data3/liuyeqiang/datasets/mft26-ir-causal-v1','--derived-gt-root','/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt','--protocol',str(evaluation/'protocol.json'),'--partition','train','--predictions',str(out),'--output',str(evaluation),'--trackeval-root','/data3/liuyeqiang/CFP-FishMOT_references/TrackEval'];run(cmd,evaluation,'evaluation',True)
                official=json.loads((evaluation/'metrics.json').read_text())
                if official['artifacts'][video]['prediction_sha256']!=rm['prediction_sha256'] or official['artifacts'][video]['gt_sha256']!=base['derived_gt_sha256']['train/'+video]:raise ValueError('Official coordinate/prediction/GT provenance differs')
                results.append(dict(memory=memory,output=str(out),manifest_sha256=sha256(out/'manifest.json'),metrics=str(evaluation/'metrics.json'),metrics_sha256=sha256(evaluation/'metrics.json')))
            state['completed'].append(dict(mode=mode,seed=seed,video=video,feature=str(feature),feature_manifest_sha256=sha256(feature/'manifest.json'),replays=results));write_json(a.state,state)
        state['status']='COMPLETE_FIRST_FOLD_NATIVE_JPM_WORKER';write_json(a.state,state)
    except Exception as e:state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
