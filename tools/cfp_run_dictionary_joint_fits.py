#!/usr/bin/env python3
"""Four CPU workers, all three seeds/matched source and sampling controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap=argparse.ArgumentParser()
    for name in ['root','queries','config','state']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--mode',choices=['pair','cross_transformer','joint','coded_joint'],required=True);a=ap.parse_args()
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU-only worker state required')
    c=json.loads(a.config.read_text());paths=[__file__,'tools/cfp_train_dictionary_separated_joint.py',str(a.config),
        'cfp/dictionary_separated_joint.py','cfp/joint_identity_decoder.py','cfp/centered_joint_identity.py',
        'cfp/joint_onpolicy_data.py','cfp/joint_set_supervision.py','cfp/joint_set_augmentation.py','cfp/predictive.py','cfp/native_local_identity.py','cfp/association.py']
    hashes={p:sha256(p) for p in paths};state=dict(status='RUNNING',pid=os.getpid(),mode=a.mode,tasks=c['seed_plan'],
        completed=[],active=None,sources=snapshot_sources(a.state.with_suffix('.sources'),paths),source_guards=hashes)
    write_json(a.state,state)
    try:
        for seed in c['seed_plan']:
            if any(sha256(p)!=h for p,h in hashes.items()):raise ValueError('Active source drift')
            output=a.root/f'dictionary_separated_joint_{a.mode}_s{seed}_v1';state['active']=dict(output=str(output));write_json(a.state,state)
            if output.exists():
                m=json.loads((output/'manifest.json').read_text())
                if m['status']!='COMPLETE':raise ValueError('Do not restart existing incomplete/active fit')
            else:
                with Path(str(output)+'.log').open('x') as log:
                    process=subprocess.Popen([sys.executable,'tools/cfp_train_dictionary_separated_joint.py','--queries',str(a.queries),
                        '--config',str(a.config),'--mode',a.mode,'--seed',str(seed),'--output',str(output)],stdout=log,stderr=subprocess.STDOUT)
                    state['active']['pid']=process.pid;write_json(a.state,state);code=process.wait()
                if code:raise RuntimeError('Actual fit failed '+str(code))
                m=json.loads((output/'manifest.json').read_text())
            if m['status']!='COMPLETE' or m['guard']['mode']!=a.mode or m['guard']['seed']!=seed or m['guard']['config_sha256']!=hashes[str(a.config)] or sha256(output/'final.pt')!=m['checkpoint_sha256'] or m['heldout_data_loaded'] or m['cuda_initialized']:
                raise ValueError('Exact complete four-fit-only CPU result required')
            state['completed'].append(dict(path=str(output),manifest_sha256=sha256(output/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256']));state['active']=None;write_json(a.state,state)
        state['status']='COMPLETE_ALL_THREE_DICTIONARY_JOINT_FITS';write_json(a.state,state)
    except Exception as error:
        state.update(status='HALTED_ACTUAL_FAILURE',error=str(error));write_json(a.state,state);raise


if __name__=='__main__':main()
