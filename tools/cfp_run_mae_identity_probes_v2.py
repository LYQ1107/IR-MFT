#!/usr/bin/env python3
"""27 registered small frozen-encoder identity fits afteractual unlabeledpretraining."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);a=ap.parse_args()
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('New assignedGPU1 identityqueue only')
    config=Path('configs/cfp/phase_vii_mae_identity_probe_v1.json');c=json.loads(config.read_text());dependency=a.root/'visible_token_fish_pretraining_all6_driver_v3.json';paths=[__file__,'tools/cfp_train_mae_identity_probe.py','cfp/mae_identity_probe.py','cfp/dinov2_fish_data.py','cfp/dinov2_identity.py','cfp/local_identity_training.py','cfp/local_identity_encoder.py','cfp/crop_retrieval.py','cfp/reid.py',str(config)];guards={p:sha256(p) for p in paths};tasks=[(init,loss,seed) for init in c['modes'] for loss in c['loss_modes'] for seed in c['seed_plan']];state=dict(status='WAITING_ACTUAL_UNLABELED_PRETRAINING',pid=os.getpid(),dependency=str(dependency),tasks=tasks,completed=[],active=None,source_guards=guards,sources=snapshot_sources(a.state.with_suffix('.sources'),paths),scope=c['scope']);write_json(a.state,state)
    def alive(pid):
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    try:
        while True:
            if any(sha256(p)!=h for p,h in guards.items()):raise ValueError('Guardedidentityprobe source changed')
            d=json.loads(dependency.read_text())
            if d['status']=='ADOPT_ACTIVE_OR_TERMINAL_PREDECESSOR_NO_DUPLICATE':dependency=Path(d['predecessor']);state['dependency']=str(dependency);write_json(a.state,state);continue
            if d['status']=='COMPLETE_SIX_UNLABELED_VISIBLE_TOKEN_PRETRAINS' and not alive(d['pid']):break
            if d['status'].startswith(('FAILED','HALTED')) or not alive(d['pid']):raise ValueError('Actualpretrain failed/missing; nofakecompletion orrestart')
            state['verified_wait_pid']=d['pid'];write_json(a.state,state);time.sleep(20)
        for init,loss,seed in tasks:
            if any(sha256(p)!=h for p,h in guards.items()):raise ValueError('Identitysource drift')
            if shutil.disk_usage(a.root).free<40.2*2**30:raise OSError('40GiB plusremaining smallhead/RAMretrieval margin')
            out=a.root/f'mae_identity_probe_{init}_{loss}_SN-015_s{seed}_v1'
            if out.exists():
                m=json.loads((out/'manifest.json').read_text());g=m['guard']
                if m['status']!='COMPLETE' or g['mode']!=init or g['loss_mode']!=loss or g['seed']!=seed or g['config_sha256']!=sha256(config) or sha256(out/'final.pt')!=m['checkpoint_sha256'] or any(sha256(p)!=h for p,h in g['source_sha256'].items()):raise ValueError('Existingprobe incomplete/changed; donotrestart')
            else:
                for attempt in range(15):
                    memory,util=map(int,subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','))
                    if memory<=64 and util==0:break
                    time.sleep(2)
                else:raise ValueError('GPU1notactuallyidle')
                cmd=[sys.executable,'tools/cfp_train_mae_identity_probe.py','--config',str(config),'--crops',str(a.root/'identity_pilot_crop_cache_v1'),'--features',str(a.root/'dinov2_two_crop_evidence_v1'),'--index',str(a.root/'identity_two_crop_index_v1'),'--reference','/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2','--weights',str(a.root/'external_models/dinov2_vits14_v1/dinov2_vits14_pretrain.pth'),'--source-proof',str(a.root/'dinov2_small_inference_source_smoke_v1.json'),'--output',str(out),'--outer','SN-015','--mode',init,'--loss-mode',loss,'--local-count','4','--seed',str(seed)]
                if init!='original':cmd+=['--encoder-init',str(a.root/f'visible_token_fish_pretrain_{init}_s{seed}_v1')]
                log=Path(str(out)+'.continuous.log');state['active']=dict(output=str(out),command=cmd,log=str(log));write_json(a.state,state)
                with log.open('x') as stream:
                    child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'));state['active']['pid']=child.pid;write_json(a.state,state);code=child.wait()
                if code:raise RuntimeError('Actualidentity childfailure; preservefiles')
                m=json.loads((out/'manifest.json').read_text())
            state['completed'].append(dict(init=init,loss=loss,seed=seed,output=str(out),manifest_sha256=sha256(out/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256']));state['active']=None;write_json(a.state,state)
        state['status']='COMPLETE_27_FROZEN_ENCODER_IDENTITY_PROBES';write_json(a.state,state)
    except Exception as e:state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
