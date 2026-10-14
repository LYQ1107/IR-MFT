#!/usr/bin/env python3
"""Seven configurations times six full videos, no online GT or outer tuning."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);a=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.state.exists():raise ValueError('New CPU-only controlqueue required')
    config=Path('configs/cfp/phase_vii_observable_occlusion_controls_v1.json');cfg=json.loads(config.read_text());tasks=[(mode,seed) for mode in cfg['occlusion_modes'] for seed in (cfg['permutation_seed_plan'] if mode=='permuted_overlap' else [0])]
    paths=[__file__,'tools/cfp_replay_observable_occlusion.py','cfp/observable_occlusion.py','cfp/fish_geometry.py','cfp/fish_motion.py','cfp/lifecycle_crossover.py','cfp/predictive.py','cfp/history.py','cfp/association.py','cfp/predictive_association.py','yolox/tracker/kalman_filter.py','yolox/tracker/byte_tracker.py','yolox/tracker/matching.py',str(config)];guards={p:sha256(p) for p in paths};state=dict(status='RUNNING',pid=os.getpid(),tasks=tasks,completed=[],active=None,source_guards=guards,sources=snapshot_sources(a.state.with_suffix('.sources'),paths),scope=cfg['scope']);write_json(a.state,state)
    try:
        for mode,seed in tasks:
            if any(sha256(p)!=h for p,h in guards.items()):raise ValueError('Guarded association source changed')
            if shutil.disk_usage(a.root).free<41*2**30:raise OSError('40GiBreserve plus1GiBmargin')
            output=a.root/f'observable_occlusion_{mode}_s{seed}_v1';p=a.root/f'observable_occlusion_{mode}_s{seed}_v1.config.json';run=dict(cfg,occlusion_mode=mode,occlusion_seed=seed);write_json(p,run);cmd=[sys.executable,'tools/cfp_replay_observable_occlusion.py','--config',str(p),'--output',str(output)];log=Path(str(output)+'.log');state['active']=dict(output=str(output),phase='native_replay',command=cmd,log=str(log));write_json(a.state,state)
            with log.open('x') as stream:
                child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT);state['active']['pid']=child.pid;write_json(a.state,state);code=child.wait()
            if code:raise RuntimeError('Actual observablecontrol replay failed '+str(log))
            m=json.loads((output/'manifest.json').read_text())
            if m['status']!='COMPLETE' or set(m['videos'])!=set(cfg['sequences']):raise ValueError('Everyregistered nativevideo mustcomplete')
            evaluation=Path(str(output)+'.trackeval');evaluation.mkdir();cmd=[sys.executable,'tools/cfp_evaluate.py','--data-root','/data3/liuyeqiang/datasets/mft26-ir-causal-v1','--derived-gt-root','/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt','--protocol','configs/cfp/protocol_v2.json','--partition','train','--predictions',str(output),'--output',str(evaluation),'--trackeval-root','/data3/liuyeqiang/CFP-FishMOT_references/TrackEval'];state['active'].update(phase='official_evaluation',output=str(evaluation),command=cmd);write_json(a.state,state)
            with (evaluation/'evaluation.log').open('x') as stream:
                child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT);state['active']['pid']=child.pid;write_json(a.state,state);code=child.wait()
            if code:raise RuntimeError('Actual officialobservable scoring failed')
            official=json.loads((evaluation/'metrics.json').read_text())
            for video,row in m['videos'].items():
                if sha256(output/(video+'.txt'))!=row['prediction_sha256'] or official['artifacts'][video]['prediction_sha256']!=row['prediction_sha256']:raise ValueError('Actual prediction/official provenance mismatch')
                if mode=='native' and not row['baseline_byte_equal']:raise ValueError('Disabledobserver baseline must matchfull referencebytes')
            state['completed'].append(dict(mode=mode,seed=seed,output=str(output),manifest_sha256=sha256(output/'manifest.json'),metrics=str(evaluation/'metrics.json'),metrics_sha256=sha256(evaluation/'metrics.json')));state['active']=None;write_json(a.state,state)
        state['status']='COMPLETE_42_NATIVE_OBSERVABLE_OCCLUSION_CONTROLS';write_json(a.state,state)
    except Exception as e:state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
