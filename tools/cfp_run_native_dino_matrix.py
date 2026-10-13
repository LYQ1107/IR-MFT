#!/usr/bin/env python3
"""Guarded four-idle-GPU serial native feature producers, all registered folds."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--worker',type=int,choices=range(4),required=True)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--state',type=Path,required=True)
    a=ap.parse_args()
    config_path=Path('configs/cfp/phase_vii_native_dino_features_v1.json')
    config=json.loads(config_path.read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.worker) or a.state.exists() or a.root.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError('New outsideGit state and exactly assigned GPU required')
    tasks=[(v,'frozen',None) for v in config['videos']]
    tasks += [(v,m,s) for s in config['seed_plan'] for m in ('global','patch_attention') for v in config['videos']]
    tasks=[t for i,t in enumerate(tasks) if i%4==a.worker]
    sources=[__file__,'tools/cfp_extract_native_dino_features.py','cfp/dinov2_native_encoder.py','cfp/dinov2_identity.py','cfp/dinov2_fish_encoder.py','cfp/local_identity_encoder.py','cfp/identity_data.py','cfp/provenance.py',str(config_path)]
    hashes={p:sha256(p) for p in sources}
    state=dict(status='RUNNING',pid=os.getpid(),worker=a.worker,tasks=tasks,completed=[],active=None,sources=snapshot_sources(a.state.with_suffix('.sources'),sources),source_guards=hashes)
    write_json(a.state,state)
    try:
        for video,mode,seed in tasks:
            if any(sha256(p)!=v for p,v in hashes.items()): raise ValueError('Producer source drift')
            if shutil.disk_usage(a.root).free<42*2**30: raise OSError('Space reserve with parallel margin')
            name=f'native_dino_{mode}_{video}_k4'+(f'_s{seed}' if seed is not None else '')+'_v1'
            out=a.root/name
            train=a.root/f'dinov2_fish_{mode}_{video}_k4_s{seed}_v2' if seed is not None else None
            state['active']=dict(output=str(out));write_json(a.state,state)
            if out.exists():
                fm=json.loads((out/'manifest.json').read_text())
                if fm['status']!='COMPLETE_NATIVE_CURRENT_DINO_FEATURES' or fm['video']!=video or fm['config_sha256']!=hashes[str(config_path)] or fm['sources']['cfp/dinov2_native_encoder.py']!=hashes['cfp/dinov2_native_encoder.py']:
                    raise ValueError('Existing incomplete or changed native producer: '+name)
                trained=fm['encoder']['trained_encoder']
                if train is None and trained is not None or train is not None and trained['checkpoint_sha256']!=sha256(train/'final.pt'):
                    raise ValueError('Wrong native encoder coordinates')
            else:
                cmd=[sys.executable,'tools/cfp_extract_native_dino_features.py','--config',str(config_path),'--data-root','/data3/liuyeqiang/datasets/mft26-ir-causal-v1','--frame-index',str(a.root/'identity_two_crop_index_v1'),'--detections','/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections','--reference','/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2','--weights',str(a.root/'external_models/dinov2_vits14_v1/dinov2_vits14_pretrain.pth'),'--source-proof',str(a.root/'dinov2_small_inference_source_smoke_v1.json'),'--output',str(out),'--video',video]
                if train is not None: cmd+=['--training',str(train)]
                with Path(str(out)+'.log').open('x') as log:
                    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);state['active']['pid']=p.pid;write_json(a.state,state);code=p.wait()
                if code: raise RuntimeError(f'Actual child exit {code}: {name}')
                fm=json.loads((out/'manifest.json').read_text())
            if fm['status']!='COMPLETE_NATIVE_CURRENT_DINO_FEATURES': raise ValueError('Actual completion required')
            state['completed'].append(dict(output=str(out),manifest_sha256=sha256(out/'manifest.json')));state['active']=None;write_json(a.state,state)
            print(json.dumps(dict(worker=a.worker,name=name,completed=len(state['completed']))),flush=True)
        state['status']='COMPLETE_REGISTERED_NATIVE_DINO_MATRIX';write_json(a.state,state)
    except Exception as e:
        state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__': main()
