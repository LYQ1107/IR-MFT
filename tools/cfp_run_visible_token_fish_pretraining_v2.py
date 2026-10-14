#!/usr/bin/env python3
"""Six preregistered unlabeled all-block fits, wait for actual GPU1 owner teardown."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);a=ap.parse_args()
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('New registered GPU1 nextstage queue required')
    config=Path('configs/cfp/phase_vii_visible_token_fish_mae_pretraining_v1.json');c=json.loads(config.read_text());dependency=a.root/'native_jpm_firstfold_worker_v1.json';pool=a.root/'unlabeled_detector_crop_fit_pool_v2';smoke=a.root/'visible_token_fish_mae_actual_cpu_smoke_v3.json'
    if json.loads(smoke.read_text())['status']!='VERIFIED_ACTUAL_ALL_BLOCK_VISIBLE_TOKEN_MAE_GRADIENT_PIXEL_EXCLUSION_RESTORE':raise ValueError('Actualnearfull visible-token gradient/pixel/restore proof required')
    paths=[__file__,'tools/cfp_pretrain_visible_token_fish_mae.py','tools/cfp_verify_visible_token_pretrain_restart.py','cfp/visible_token_fish_mae.py','cfp/unlabeled_crop_pool.py','cfp/dinov2_fish_data.py','cfp/dinov2_identity.py','cfp/local_identity_encoder.py','cfp/fish_pixel_reconstruction.py','cfp/masked_identity_checkpoint.py',str(config)];guards={p:sha256(p) for p in paths};tasks=[(mode,seed) for seed in c['seed_plan'] for mode in c['modes']];state=dict(status='WAITING_ACTUAL_GPU1_NATIVE_OWNER',pid=os.getpid(),dependency=str(dependency),tasks=tasks,completed=[],active=None,launches=[],source_guards=guards,sources=snapshot_sources(a.state.with_suffix('.sources'),paths),scope=c['scope'],storage_policy=c['storage_policy']);write_json(a.state,state)
    def alive(pid):
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    def source_check():
        if any(sha256(p)!=h for p,h in guards.items()):raise ValueError('Guarded nearfullpretraining source changed')
    def reserve_check():
        dets=Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections');raw=0
        for kind,dim in [('legacy_gru',512),('dino_frozen_fifo',384)]:
            driver=json.loads((a.root/f'raw_whole_{kind}_driver_v3.json').read_text());done={Path(x['output']).name.split('_')[-2] for x in driver['completed']}
            for video in driver['tasks']:
                if video in done:continue
                with np.load(dets/(video+'.npz'),allow_pickle=False) as data:
                    # Upper bound: actualv3trace fields from profiler, 1% ZIP
                    # overhead plus1MiB each; compression isnotassumed.
                    raw+=int((len(data['boxes'])*(dim*4+20)+(len(data['frame_ids'])+len(data['offsets']))*8)*1.01)+2**20
        proof=json.loads(smoke.read_text());model_bytes=proof['checkpoint_bytes'];opt_bytes=2*model_bytes
        # Four fullstate-equivalents cover two retainedrestart states and
        # old/new atomiclive copies. Encoderfinal bound <=model checkpoint.
        planned=(len(tasks)-len(state['completed']))*model_bytes+4*(model_bytes+opt_bytes+2**20)
        required=40*2**30+raw+planned+.125*2**30
        state['storage_forecast']=dict(free_bytes=shutil.disk_usage(a.root).free,remaining_raw_upper_bound_bytes=raw,remaining_encoder_restart_atomic_bytes=planned,safety_margin_bytes=.125*2**30,required_bytes=required)
        write_json(a.state,state)
        if shutil.disk_usage(a.root).free<required:raise OSError('Exactremaining-raw/final/restart/atomic forecast cannotpreserve40GiB')
    def launch(output,mode,seed,phase):
        source_check();reserve_check()
        for attempt in range(15):
            mem,util=map(int,subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','))
            if mem<=64 and util==0:break
            time.sleep(2)
        else:raise ValueError('GPU1notactuallyidle; stopwithoutinterference')
        cmd=[sys.executable,'tools/cfp_pretrain_visible_token_fish_mae.py','--root',str(a.root),'--config',str(config),'--pool',str(pool),'--output',str(output),'--mode',mode,'--seed',str(seed)]
        if phase=='stop':cmd+=['--stop-after-epoch','2']
        if phase=='resume':cmd+=['--resume']
        log=Path(str(output)+'.'+phase+'.log');entry=dict(output=str(output),mode=mode,seed=seed,phase=phase,command=cmd,log=str(log));state.update(status='RUNNING_UNLABELED_VISIBLE_TOKEN_PRETRAINING',active=entry);write_json(a.state,state)
        with log.open('x') as stream:
            child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'));entry['pid']=child.pid;state['launches'].append(entry);write_json(a.state,state);code=child.wait()
        entry.update(exit_code=code,log_sha256=sha256(log));write_json(a.state,state)
        if code:raise RuntimeError('Actualunlabeled childfailed; preserveall files '+str(log))
        result=json.loads((output/'manifest.json').read_text());expected='CHECKPOINTED_EARLY_STOP' if phase=='stop' else 'COMPLETE_UNLABELED_VISIBLE_TOKEN_PRETRAINING'
        if result['status']!=expected:raise ValueError('Actualstage terminal status mismatch')
        state['active']=None;write_json(a.state,state);return result
    try:
        predecessor=a.root/'visible_token_fish_pretraining_all6_driver_v1.json'
        while True:
            source_check();previous=json.loads(predecessor.read_text())
            if previous['launches']:
                state.update(status='ADOPT_ACTIVE_OR_TERMINAL_PREDECESSOR_NO_DUPLICATE',predecessor=str(predecessor));write_json(a.state,state);return
            if previous['status']=='HALTED_ACTUAL_FAILURE' and not alive(previous['pid']):
                if 'budgets cannotpreserve40GiB' not in previous.get('error',''):raise ValueError('Prerequisite hasotherfailure, notforecastreplacement')
                break
            if not alive(previous['pid']):raise ValueError('Unfinishedpredecessor processmissing; noautomaticrestart')
            state.update(status='WAITING_ACTUAL_CONSERVATIVE_FORECAST_PREDECESSOR',verified_wait_pid=previous['pid']);write_json(a.state,state);time.sleep(20)
        dependencies=[dependency,a.root/'native_jpm_controls_firstfold_worker_v1.json']
        while True:
            source_check();deps=[json.loads(p.read_text()) for p in dependencies]
            if all(d['status']=='COMPLETE_FIRST_FOLD_NATIVE_JPM_WORKER' and not alive(d['pid']) for d in deps):break
            if any(d['status'].startswith(('FAILED','HALTED')) or d['status']!='COMPLETE_FIRST_FOLD_NATIVE_JPM_WORKER' and not alive(d['pid']) for d in deps):raise ValueError('Actualnativeprerequisite failed/missing')
            state.update(status='WAITING_BOTH_ACTUAL_NATIVE_OWNERS',verified_wait_pids=[d['pid'] for d in deps if alive(d['pid'])]);write_json(a.state,state);time.sleep(20)
        for mode,seed in tasks:
            out=a.root/f'visible_token_fish_pretrain_{mode}_s{seed}_v1';m=launch(out,mode,seed,'continuous')
            if mode=='masked_visible_tokens' and seed==0:
                restarted=a.root/'visible_token_fish_pretrain_masked_visible_tokens_s0_v1_restarted';launch(restarted,mode,seed,'stop');launch(restarted,mode,seed,'resume');proof=a.root/'visible_token_fish_pretrain_exact_restart_v1.json';subprocess.run([sys.executable,'tools/cfp_verify_visible_token_pretrain_restart.py','--continuous',str(out),'--restarted',str(restarted),'--launch-proof',str(a.state),'--output',str(proof)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True);state['restart_proof']=dict(path=str(proof),sha256=sha256(proof));write_json(a.state,state)
            else:
                # Only this queue's new explicitlytemporary optimizer state;
                # decoder/fullstate digest recorded, finalencoder alwaysretained.
                temp=out/'.scratch-last.pt'
                if sha256(temp)!=m['full_training_scratch_sha256']:raise ValueError('Cannotretire changed ownscratch checkpoint')
                temp.unlink();m['own_temporary_optimizer_scratch_retired']=True;m['full_training_scratch_retired_sha256']=m['full_training_scratch_sha256'];write_json(out/'manifest.json',m)
            state['completed'].append(dict(mode=mode,seed=seed,output=str(out),manifest_sha256=sha256(out/'manifest.json'),encoder_checkpoint_sha256=m['encoder_checkpoint_sha256']));write_json(a.state,state)
        state['status']='COMPLETE_SIX_UNLABELED_VISIBLE_TOKEN_PRETRAINS';write_json(a.state,state)
    except Exception as e:state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
