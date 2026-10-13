#!/usr/bin/env python3
"""Three-seed on-policy local identity replay; wait for actual producers only."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);a=ap.parse_args()
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU-only state required')
    modes=['global','local_temporal','full','global_gru','local_attention','local_contrastive','no_ema','no_prediction','no_id']
    tasks=[(m,s,False) for s in range(3) for m in modes]+[('full',s,True) for s in range(3)]
    paths=[__file__,'tools/cfp_replay_native_current_identity.py','cfp/native_local_identity.py','cfp/local_dense_jepa.py','cfp/local_dense_context_identity.py','cfp/predictive_association.py','cfp/association.py','cfp/predictive.py','configs/cfp/phase_vii_native_current_identity_replay_v1.json']
    hashes={p:sha256(p) for p in paths};state=dict(status='RUNNING',pid=os.getpid(),tasks=tasks,completed=[],active=None,sources=snapshot_sources(a.state.with_suffix('.sources'),paths),source_guards=hashes,scope='Full native SN015 firstfold allnineV1 controls three seeds plus V2full three seeds. Actual assigned history; teacher/GT/future excluded. Offline crop supports are not used as native memories.')
    write_json(a.state,state)
    try:
        for mode,seed,context in tasks:
            if any(sha256(p)!=v for p,v in hashes.items()):raise ValueError('Replay source drift')
            model=a.root/(f'local_dense_context_{mode}_SN-015_k4_s{seed}_v2' if context else f'local_dense_{mode}_SN-015_k4_s{seed}_v1')
            out=a.root/(f'native_identity_context_{mode}_SN-015_s{seed}_v2' if context else f'native_identity_local_{mode}_SN-015_s{seed}_v1')
            deadline=time.monotonic()+21600
            while True:
                mm=json.loads((model/'manifest.json').read_text()) if (model/'manifest.json').exists() else {}
                om=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else {}
                if mm.get('status')=='COMPLETE' and (not out.exists() or om.get('status')=='COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY'):break
                if str(mm.get('status','')).startswith('FAILED') or str(om.get('status','')).startswith('FAILED'):raise ValueError('Actual producer failed')
                if time.monotonic()>deadline:raise TimeoutError('Producer wait exceeded six hours')
                state.update(status='WAITING_ACTUAL_PRODUCER',active=dict(model=str(model),output=str(out)));write_json(a.state,state);time.sleep(30)
            state.update(status='RUNNING',active=dict(model=str(model),output=str(out)));write_json(a.state,state)
            if not out.exists():
                cmd=[sys.executable,'tools/cfp_replay_native_current_identity.py','--config','configs/cfp/phase_vii_native_current_identity_replay_v1.json','--features',str(a.root/'native_dino_frozen_SN-015_k4_v1'),'--detections','/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections','--output',str(out),'--video','SN-015','--model',str(model)]
                if context:cmd+=['--context-identity']
                with Path(str(out)+'.log').open('x') as log:
                    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);state['active']['pid']=p.pid;write_json(a.state,state);code=p.wait()
                if code:raise RuntimeError(f'Actual replay child exit {code}')
                om=json.loads((out/'manifest.json').read_text())
            if om['model_checkpoint_sha256']!=mm['checkpoint_sha256'] or sha256(out/'SN-015.txt')!=om['prediction_sha256']:raise ValueError('Checkpoint or prediction drift')
            state['completed'].append(dict(output=str(out),manifest_sha256=sha256(out/'manifest.json')));state['active']=None;write_json(a.state,state)
            print(json.dumps(dict(mode=mode,seed=seed,context=context,completed=len(state['completed']))),flush=True)
        state['status']='COMPLETE_NATIVE_LOCAL_IDENTITY_CONTROLS';write_json(a.state,state)
    except Exception as e:state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
