#!/usr/bin/env python3
"""Fixed trained controls, both native routes, complete heldout-head videos."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);a=ap.parse_args()
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU-only driver required')
    tasks=[(mode,seed,route) for seed in range(3) for mode in ['joint','pair','cross_transformer'] for route in ['bounded','full']]
    sources=[__file__,'tools/cfp_replay_centered_joint_identity.py','cfp/online_centered_joint_identity.py','cfp/centered_joint_identity.py','cfp/joint_identity_decoder.py','cfp/joint_identity_controls.py','cfp/lifecycle_crossover.py','cfp/predictive.py','cfp/history.py','cfp/provenance.py','configs/cfp/phase_vii_centered_joint_probe_v2.json','configs/cfp/phase_vii_joint_evaluation_subset_v1.json'];hashes={p:sha256(p) for p in sources}
    state=dict(status='RUNNING',pid=os.getpid(),tasks=tasks,completed=[],active=None,sources=snapshot_sources(a.state.with_suffix('.sources'),sources),source_guards=hashes,scope='Ninefitted models x twonative routes onbothfullSN001/SN015; inherited512encoder/GRU conditionalheadheldout. Uniformcenteredcontrol separately requires exactdisabledbytes. No outerselection or fullnewmethod/native384 independent fit claim.')
    write_json(a.state,state)
    try:
        for mode,seed,route in tasks:
            if any(sha256(p)!=v for p,v in hashes.items()):raise ValueError('Active source drift')
            model=a.root/f'centered_joint_probe_{mode}_s{seed}_v2';out=a.root/f'centered_joint_native_{mode}_{route}_s{seed}_v2';deadline=time.monotonic()+21600
            while True:
                m=json.loads((model/'manifest.json').read_text()) if (model/'manifest.json').exists() else {}
                if m.get('status')=='COMPLETE':break
                if str(m.get('status','')).startswith('FAILED'):raise ValueError('Actualmodel failed')
                if time.monotonic()>deadline:raise TimeoutError('Actual model wait exceeded six hours')
                state.update(status='WAITING_ACTUAL_TRAINING',active=dict(model=str(model)));write_json(a.state,state);time.sleep(30)
            config=dict(state='FROZEN',partition='train',seed=seed,sequences=['SN-001','SN-015'],route=route,model_root=str(model),maximum_delta=.03,cost_version='PER_CANDIDATE_CENTERED_CONDITIONAL_RANKING_V2',output_policy='Unchangednative confirmation/low/unconfirmed/writes, measured currentproposals',scope=state['scope']);config_path=a.root/f'centered_joint_replay_{mode}_{route}_s{seed}_v2.json'
            if config_path.exists():
                if json.loads(config_path.read_text())!=config:raise ValueError('Frozen replay config changed')
            else:write_json(config_path,config)
            state.update(status='RUNNING',active=dict(model=str(model),output=str(out)));write_json(a.state,state)
            if out.exists():raise ValueError('Existing native replay requires explicit inspection, not automatic restart')
            cmd=[sys.executable,'tools/cfp_replay_centered_joint_identity.py','--config',str(config_path),'--output',str(out)]
            with Path(str(out)+'.log').open('x') as log:
                p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);state['active']['pid']=p.pid;write_json(a.state,state);code=p.wait()
            if code:raise RuntimeError('Actualnative replay failed '+str(code))
            rm=json.loads((out/'manifest.json').read_text());assert rm['status']=='COMPLETE' and rm['joint_checkpoint_sha256']==m['checkpoint_sha256']
            evaluation=Path(str(out)+'.trackeval');cmd=[sys.executable,'tools/cfp_evaluate.py','--data-root','/data3/liuyeqiang/datasets/mft26-ir-causal-v1','--derived-gt-root','/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt','--protocol','configs/cfp/phase_vii_joint_evaluation_subset_v1.json','--partition','train','--predictions',str(out),'--output',str(evaluation),'--trackeval-root','/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
            evaluation.mkdir(exist_ok=False)
            with (evaluation/'evaluation.log').open('x') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError('Actual official scoring failed '+str(code))
            metrics=json.loads((evaluation/'metrics.json').read_text())
            for video in config['sequences']:
                if metrics['artifacts'][video]['prediction_sha256']!=rm['videos'][video]['prediction_sha256']:raise ValueError('Prediction scoring drift')
            state['completed'].append(dict(output=str(out),manifest_sha256=sha256(out/'manifest.json'),official_metrics=str(evaluation/'metrics.json'),metrics_sha256=sha256(evaluation/'metrics.json'),metrics=metrics['metrics']));state['active']=None;write_json(a.state,state);print(json.dumps(dict(mode=mode,seed=seed,route=route,completed=len(state['completed']))),flush=True)
        state['status']='COMPLETE_REGISTERED_CENTERED_JOINT_NATIVE_CONTROLS';write_json(a.state,state)
    except Exception as e:state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
