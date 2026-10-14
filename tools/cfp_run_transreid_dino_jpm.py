#!/usr/bin/env python3
"""Two owned GPU queues: nine registered JPM crop fits and actual restart proof."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);ap.add_argument('--worker',choices=['jpm','controls'],required=True);a=ap.parse_args()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES');expected='1' if a.worker=='jpm' else '3'
    if gpu!=expected or a.state.exists():raise ValueError('New registered GPU1/3 owner only')
    config=Path('configs/cfp/phase_vii_transreid_dino_jpm_v1.json');c=json.loads(config.read_text());smoke=a.root/'transreid_dino_jpm_actual_cpu_smoke_v1.json'
    if json.loads(smoke.read_text())['status']!='VERIFIED_ACTUAL_PINNED_DINO_JPM_SOURCE_GRADIENT_AND_DELTA':raise ValueError('Actual licensed source/gradient proof required')
    paths=[__file__,'tools/cfp_train_transreid_dino_jpm.py','tools/cfp_verify_transreid_jpm_restart.py',str(config),'cfp/transreid_dino_jpm.py','third_party/transreid_jpm/shuffle.py','cfp/masked_identity_checkpoint.py','cfp/dinov2_fish_data.py','cfp/dinov2_identity.py','cfp/local_identity_training.py','cfp/local_identity_encoder.py','cfp/crop_retrieval.py','cfp/reid.py'];guards={x:sha256(x) for x in paths}
    tasks=[('jpm',0),('cls_capacity',0),('jpm',1),('jpm',2)] if a.worker=='jpm' else [('global',0),('global',1),('global',2),('cls_capacity',1),('cls_capacity',2)]
    state=dict(status='RUNNING',pid=os.getpid(),worker=a.worker,physical_gpu=gpu,tasks=tasks,completed=[],active=None,launches=[],source_guards=guards,sources=snapshot_sources(a.state.with_suffix('.sources'),paths),scope='Nine distinct3mode/3seed fixedfinal firstfold fits, plus separate real JPMseed0 epoch2exit/freshresume duplicate. No outerselect, native-MOT/allfold/fullmethod/novelty claim. PreviousrawGPU0/2 sources unchanged; GPU4-9 neverused.')
    write_json(a.state,state)
    def launch(output,mode,seed,phase):
        if any(sha256(p)!=h for p,h in guards.items()):raise ValueError('Guarded source changed while queue active')
        if shutil.disk_usage(a.root).free<46*2**30:raise OSError('40GiB reserve plus6GiB parallel/remainingfit margin')
        for attempt in range(15):
            query=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);mem,util=map(int,query.split(','))
            if mem<=64 and util==0:break
            time.sleep(2)
        else:raise ValueError('GPU not actuallyidle after ownedchildexit; stop without interfering')
        cmd=[sys.executable,'tools/cfp_train_transreid_dino_jpm.py','--config',str(config),'--crops',str(a.root/'identity_pilot_crop_cache_v1'),'--features',str(a.root/'dinov2_two_crop_evidence_v1'),'--index',str(a.root/'identity_two_crop_index_v1'),'--reference','/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2','--weights',str(a.root/'external_models/dinov2_vits14_v1/dinov2_vits14_pretrain.pth'),'--source-proof',str(a.root/'dinov2_small_inference_source_smoke_v1.json'),'--output',str(output),'--outer','SN-015','--mode',mode,'--local-count','4','--seed',str(seed)]
        if phase=='stop':cmd+=['--stop-after-epoch','2']
        if phase=='resume':cmd+=['--resume']
        log=Path(str(output)+'.'+phase+'.log');entry=dict(output=str(output),mode=mode,seed=seed,phase=phase,command=cmd,log=str(log));state['active']=entry;write_json(a.state,state)
        with log.open('x') as stream:
            child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'));entry['pid']=child.pid;state['launches'].append(entry);write_json(a.state,state);code=child.wait()
        entry.update(exit_code=code,log_sha256=sha256(log));write_json(a.state,state)
        if code:raise RuntimeError('Actual child failure; preserve outputs '+str(output))
        m=json.loads((output/'manifest.json').read_text());expected_status='CHECKPOINTED_EARLY_STOP' if phase=='stop' else 'COMPLETE'
        if m['status']!=expected_status:raise ValueError('Actual terminal stage differs')
        state['active']=None;write_json(a.state,state);return m
    try:
        for mode,seed in tasks:
            output=a.root/f'transreid_dino_{mode}_SN-015_s{seed}_v1';m=launch(output,mode,seed,'continuous');state['completed'].append(dict(output=str(output),manifest_sha256=sha256(output/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256']));write_json(a.state,state)
            if mode=='jpm' and seed==0:
                restarted=a.root/'transreid_dino_jpm_SN-015_s0_v1_restarted';launch(restarted,mode,seed,'stop');launch(restarted,mode,seed,'resume');proof=a.root/'transreid_dino_jpm_exact_restart_v1.json'
                subprocess.run([sys.executable,'tools/cfp_verify_transreid_jpm_restart.py','--continuous',str(output),'--restarted',str(restarted),'--stop-log',str(restarted)+'.stop.log','--launch-proof',str(a.state),'--output',str(proof)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True);state['restart_proof']=dict(path=str(proof),sha256=sha256(proof));write_json(a.state,state)
        state['status']='COMPLETE_REGISTERED_JPM_FIRST_FOLD_WORKER';write_json(a.state,state)
    except Exception as e:
        state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
