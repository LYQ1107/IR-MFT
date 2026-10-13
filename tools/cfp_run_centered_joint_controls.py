#!/usr/bin/env python3
"""Two serial CPU workers, three same-input architectures and three seeds."""
import argparse,json,os,subprocess,sys,shutil
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);ap.add_argument('--worker',type=int,choices=[0,1],required=True);a=ap.parse_args()
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU worker required')
    tasks=[('pair',0),('cross_transformer',0),('pair',1),('cross_transformer',1),('joint',1),('pair',2),('cross_transformer',2),('joint',2),('joint',0)][a.worker::2]
    config=Path('configs/cfp/phase_vii_centered_joint_probe_v2.json');paths=[__file__,'tools/cfp_train_centered_joint_identity.py','cfp/centered_joint_identity.py','cfp/centered_joint_training.py','cfp/joint_identity_decoder.py','cfp/joint_identity_supervision.py','cfp/joint_identity_training_data.py','cfp/predictive.py','cfp/provenance.py',str(config)];hashes={p:sha256(p) for p in paths}
    state=dict(status='RUNNING',pid=os.getpid(),tasks=tasks,completed=[],active=None,sources=snapshot_sources(a.state.with_suffix('.sources'),paths),source_guards=hashes,scope='Repairround1 centeredconditionalranking/separatebalancedNULL, original512conditional warmstates only, inheritedencoder/GRU exposure; allthree controls/seeds, no OOF/new384/native/finalgoal success claim.')
    write_json(a.state,state)
    try:
        for mode,seed in tasks:
            if any(sha256(p)!=v for p,v in hashes.items()):raise ValueError('Active sources changed')
            if shutil.disk_usage(a.root).free<42*2**30:raise OSError('40GiBreserve with margin')
            out=a.root/f'centered_joint_probe_{mode}_s{seed}_v2';state['active']=dict(output=str(out));write_json(a.state,state)
            if not out.exists():
                command=[sys.executable,'tools/cfp_train_centered_joint_identity.py','--queries',str(a.root/'joint_actual_queries_s0_v2'),'--config',str(config),'--architecture',mode,'--seed',str(seed),'--output',str(out)]
                with Path(str(out)+'.log').open('x') as log:
                    p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);state['active']['pid']=p.pid;write_json(a.state,state);code=p.wait()
                if code:raise RuntimeError('Actual child failure '+str(code))
            m=json.loads((out/'manifest.json').read_text());g=m['guard']
            if m['status']!='COMPLETE' or g['architecture']!=mode or g['seed']!=seed or g['config_sha256']!=hashes[str(config)] or sha256(out/'epoch8.pt')!=m['checkpoint_sha256'] or any(sha256(p)!=v for p,v in g['source_sha256'].items()):raise ValueError('Actualcomplete sameconfig/source/checkpoint required')
            state['completed'].append(dict(output=str(out),manifest_sha256=sha256(out/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256']));state['active']=None;write_json(a.state,state);print(json.dumps(dict(mode=mode,seed=seed,completed=len(state['completed']))),flush=True)
        state['status']='COMPLETE_REGISTERED_CENTERED_JOINT_CONTROLS';write_json(a.state,state)
    except Exception as e:state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
