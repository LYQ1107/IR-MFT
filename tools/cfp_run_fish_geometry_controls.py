#!/usr/bin/env python3
"""Two isolated CPUworkers, sixteen complete-six-video geometry controls."""
import argparse,json,os,subprocess,sys,shutil
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);ap.add_argument('--worker',type=int,choices=[0,1],required=True);a=ap.parse_args()
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('NewCPUworker required')
    plan=Path('configs/cfp/phase_vii_fish_geometry_controls_v1.json');c=json.loads(plan.read_text());tasks=[(g,w) for w in [.3,0.] for g in c['geometry_kinds']][a.worker::2]
    paths=[__file__,str(plan),'tools/cfp_replay_fish_geometry.py','cfp/fish_geometry.py','cfp/fish_motion.py','cfp/lifecycle_crossover.py','cfp/predictive.py','cfp/history.py','cfp/association.py','cfp/predictive_association.py','yolox/tracker/kalman_filter.py','yolox/tracker/byte_tracker.py','yolox/tracker/matching.py','cfp/provenance.py'];hashes={p:sha256(p) for p in paths};state=dict(status='RUNNING',pid=os.getpid(),tasks=tasks,completed=[],active=None,sources=snapshot_sources(a.state.with_suffix('.sources'),paths),source_guards=hashes,scope=c['scope']);write_json(a.state,state)
    try:
        for kind,weight in tasks:
            if any(sha256(p)!=v for p,v in hashes.items()):raise ValueError('Guardedgeometry/native source changed')
            if shutil.disk_usage(a.root).free<42*2**30:raise OSError('40GiBreserve plusmargin')
            name=f'fish_geometry_{kind}_{"gru" if weight else "geometry"}_v1';out=a.root/name;config=dict(c,geometry_kind=kind,appearance_weight=weight);cp=a.root/(name+'.config.json');write_json(cp,config)
            if out.exists():raise ValueError('Existingoutput requires actualinspection, notautomaticrestart')
            state['active']=dict(output=str(out));write_json(a.state,state)
            with (a.root/(name+'.log')).open('x') as log:
                p=subprocess.Popen([sys.executable,'tools/cfp_replay_fish_geometry.py','--config',str(cp),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT);state['active']['pid']=p.pid;write_json(a.state,state);code=p.wait()
            if code:raise RuntimeError('Actualnative child failed '+str(code))
            m=json.loads((out/'manifest.json').read_text());assert m['status']=='COMPLETE' and sum(v['frames'] for v in m['videos'].values())==27251
            evaluation=Path(str(out)+'.trackeval');evaluation.mkdir(exist_ok=False)
            cmd=[sys.executable,'tools/cfp_evaluate.py','--data-root','/data3/liuyeqiang/datasets/mft26-ir-causal-v1','--derived-gt-root','/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt','--protocol','configs/cfp/protocol_v2.json','--partition','train','--predictions',str(out),'--output',str(evaluation),'--trackeval-root','/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
            with (evaluation/'evaluation.log').open('x') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError('Actualofficial scoring failed '+str(code))
            metrics=json.loads((evaluation/'metrics.json').read_text())
            for video in c['sequences']:assert metrics['artifacts'][video]['prediction_sha256']==m['videos'][video]['prediction_sha256']
            state['completed'].append(dict(name=name,output=str(out),manifest_sha256=sha256(out/'manifest.json'),official_metrics_path=str(evaluation/'metrics.json'),official_metrics_sha256=sha256(evaluation/'metrics.json'),metrics=metrics['metrics']));state['active']=None;write_json(a.state,state);print(json.dumps(dict(name=name,completed=len(state['completed']))),flush=True)
        state['status']='COMPLETE_REGISTERED_SIX_VIDEO_FISH_GEOMETRY_CONTROLS';write_json(a.state,state)
    except Exception as e:state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
