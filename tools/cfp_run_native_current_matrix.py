#!/usr/bin/env python3
"""All six/three-seed native ordinary-memory controls and official metrics."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);a=ap.parse_args()
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU-only state required')
    config=Path('configs/cfp/phase_vii_native_dino_features_v1.json');c=json.loads(config.read_text())
    feature_tasks=[(v,'frozen',None) for v in c['videos']]+[(v,m,s) for s in c['seed_plan'] for m in ['global','patch_attention'] for v in c['videos']]
    tasks=[(v,m,s,mem) for v,m,s in feature_tasks for mem in ['last','fifo','ema']]
    sources=[__file__,'tools/cfp_replay_native_current_identity.py','cfp/native_local_identity.py','cfp/predictive_association.py','cfp/association.py','cfp/predictive.py','tools/cfp_evaluate.py','configs/cfp/phase_vii_native_current_identity_replay_v1.json',str(config),'configs/cfp/phase_vii_joint_evaluation_subset_v1.json']
    hashes={p:sha256(p) for p in sources};state=dict(status='RUNNING',pid=os.getpid(),tasks=tasks,completed=[],active=None,sources=snapshot_sources(a.state.with_suffix('.sources'),sources),source_guards=hashes,scope='42current feature producers x three ordinary memories=126 fullnative video replays; six videos three seeds for each fitted global/local encoder, frozen deterministic once. Offline official scoring after current-only replay, no outer threshold selection. Cache timing is not raw FPS.')
    write_json(a.state,state)
    try:
        for video,mode,seed,memory in tasks:
            if any(sha256(p)!=v for p,v in hashes.items()):raise ValueError('Guarded source drift')
            features=a.root/(f'native_dino_{mode}_{video}_k4'+(f'_s{seed}' if seed is not None else '')+'_v1')
            out=a.root/f'native_identity_{mode}_{video}_{memory}_s{seed if seed is not None else 0}_v1'
            deadline=time.monotonic()+21600
            while True:
                fm=json.loads((features/'manifest.json').read_text()) if (features/'manifest.json').exists() else {}
                om=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else {}
                if fm.get('status')=='COMPLETE_NATIVE_CURRENT_DINO_FEATURES' and (not out.exists() or om.get('status')=='COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY'):break
                if str(fm.get('status','')).startswith('FAILED') or str(om.get('status','')).startswith('FAILED'):raise ValueError('Actual producer failed')
                if time.monotonic()>deadline:raise TimeoutError('Actual producer wait exceeded six hours')
                state.update(status='WAITING_ACTUAL_PRODUCER',active=dict(features=str(features),output=str(out)));write_json(a.state,state);time.sleep(30)
            state.update(status='RUNNING',active=dict(features=str(features),output=str(out)));write_json(a.state,state)
            if not out.exists():
                cmd=[sys.executable,'tools/cfp_replay_native_current_identity.py','--config','configs/cfp/phase_vii_native_current_identity_replay_v1.json','--features',str(features),'--detections','/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections','--output',str(out),'--video',video,'--memory',memory]
                with Path(str(out)+'.log').open('x') as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);state['active']['pid']=child.pid;write_json(a.state,state);code=child.wait()
                if code:raise RuntimeError('Actual replay exit '+str(code))
                om=json.loads((out/'manifest.json').read_text())
            if om['feature_manifest_sha256']!=sha256(features/'manifest.json') or om['memory']!=memory or sha256(out/(video+'.txt'))!=om['prediction_sha256']:raise ValueError('Actual producer coordinates/output changed')
            evaluation=Path(str(out)+'.trackeval');parent=Path('configs/cfp/phase_vii_joint_evaluation_subset_v1.json');p=json.loads(parent.read_text())
            if not (evaluation/'metrics.json').exists():
                evaluation.mkdir(exist_ok=False);write_json(evaluation/'protocol.json',dict(p,train=[video],dev=[],historical_test=[],role='Offline completed current-only fullnative scoring; no outerselection',parent_protocol_sha256=sha256(parent)))
                cmd=[sys.executable,'tools/cfp_evaluate.py','--data-root','/data3/liuyeqiang/datasets/mft26-ir-causal-v1','--derived-gt-root','/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt','--protocol',str(evaluation/'protocol.json'),'--partition','train','--predictions',str(out),'--output',str(evaluation),'--trackeval-root','/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
                with (evaluation/'evaluation.log').open('x') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
                if code:raise RuntimeError('Actual TrackEval exit '+str(code))
            metrics=json.loads((evaluation/'metrics.json').read_text())
            if metrics['artifacts'][video]['prediction_sha256']!=om['prediction_sha256'] or metrics['artifacts'][video]['gt_sha256']!=p['derived_gt_sha256']['train/'+video]:raise ValueError('Offline scoring provenance drift')
            state['completed'].append(dict(output=str(out),manifest_sha256=sha256(out/'manifest.json'),official_metrics_sha256=sha256(evaluation/'metrics.json'),metrics=metrics['metrics'][video]));state['active']=None;write_json(a.state,state)
            print(json.dumps(dict(video=video,mode=mode,seed=seed,memory=memory,completed=len(state['completed']),HOTA=metrics['metrics'][video]['HOTA'])),flush=True)
        state['status']='COMPLETE_REGISTERED_NATIVE_CURRENT_IDENTITY_MATRIX';write_json(a.state,state)
    except Exception as e:state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
