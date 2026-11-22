#!/usr/bin/env python3
"""A Dev witness forces the same extension for robust and clean controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.run_pseudo_ir_comparison import dump,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--from-epoch',type=int,required=True)
    parser.add_argument('--witness',required=True);parser.add_argument('--earlier',nargs='+',required=True);args=parser.parse_args()
    if args.from_epoch not in [18,24,30]:raise ValueError('Only common six-epoch extensions')
    work=Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    path=work/f'native_query_common_e{args.from_epoch}_extension.json'
    lock=path.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Common extension already decided')
    records=[]
    for name in [args.witness]+args.earlier:
        p=Path(name);v=json.loads(p.read_text());m=v['manifest']
        if v['status']!='COMPLETE' or m['split']!='dev' or m['frames']!=4834:raise ValueError('Complete Dev only')
        c=dict(m['config'])
        for field in ['id','native_query_checkpoint','registration']:c.pop(field,None)
        records.append(dict(path=str(p),sha256=sha256(p),HOTA=v['trackeval']['HOTA'],epoch=m['native_query']['epoch'],identity=c))
    witness=records[0];earlier=records[1:]
    if any(item['identity']!=witness['identity'] for item in earlier):raise ValueError('One algorithm/frontend/cue/seed curve required')
    if {item['epoch'] for item in earlier}!=set(range(6,args.from_epoch-12+1,6)):raise ValueError('All lookback checkpoints required')
    if witness['epoch']>args.from_epoch:raise ValueError('Future witness')
    gain=witness['HOTA']-max(item['HOTA'] for item in earlier)
    if gain<=.2:raise ValueError('No forced common extension')
    state=dict(status='WAITING_FOR_ALL_MATCHED_FITS',from_epoch=args.from_epoch,to_epoch=args.from_epoch+6,
               gain_lower_bound_pp=gain,evidence=records,Test_selection=False,
               rule='Any complete Dev witness improving best over the complete12-epoch lookback by>0.2pp forces all matched fits to continue. Continue clean-only controls as well, preserving all earlier stages.')
    dump(path,state)
    families=[('native_query','tools/launch_native_query_fits.py'),('native_robust_query','tools/launch_native_robust_query_fits.py')]
    while True:
        dependencies=[json.loads((work/f'{family}_fit_e{args.from_epoch}_state.json').read_text()) for family,_ in families]
        if any(d['status'] in ['FAILED','DEPENDENCY_FAILED'] for d in dependencies):
            state['status']='DEPENDENCY_FAILED';dump(path,state);return
        if all(d['status']=='COMPLETE' for d in dependencies):break
        time.sleep(5)
    for family,_ in families:
        for front in ['new','rgb']:
            for cue in ['rgb','ir']:
                for seed in range(3):
                    folder=work/family/f'{front}_{cue}_s{seed}'
                    completed=json.loads((folder/'complete.json').read_text())
                    if completed['epochs']!=args.from_epoch:raise ValueError('Actual matched budget mismatch')
                    for name in ['complete','history']:
                        source=folder/(name+'.json');destination=folder/f'stage{args.from_epoch}_{name}.json'
                        if destination.exists():
                            if sha256(source)!=sha256(destination):raise ValueError('Different stage backup')
                        else:shutil.copy2(source,destination)
    next_epoch=args.from_epoch+6;children=[]
    for family,launcher in families:
        if (work/f'{family}_fit_e{next_epoch}_state.json').exists():raise ValueError('Next fit already scheduled')
        log=work/f'{family}_fit_e{next_epoch}.controller.log'
        command=[sys.executable,'-u',launcher,'--epochs',str(next_epoch),'--stage',f'e{next_epoch}','--resume']
        with log.open('ab',buffering=0) as handle:
            child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(family=family,pid=child.pid,command=command,log=str(log)))
    state.update(status='EXECUTED_COMMON_EXTENSION',children=children);dump(path,state)
    print(json.dumps(dict(status=state['status'],to_epoch=next_epoch,matched_heads=24,gain_lower_bound_pp=gain)))


if __name__=='__main__':main()
