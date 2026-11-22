#!/usr/bin/env python3
"""Execute a forced common-budget extension once all six fits complete."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['pair','strong'],required=True)
    p.add_argument('--from-epoch',type=int,required=True);p.add_argument('--witness',required=True)
    p.add_argument('--earlier',nargs='+',required=True);a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root'])
    if a.from_epoch not in [18,24,30]:raise ValueError('Only registered six-epoch common extensions')
    to_epoch=a.from_epoch+6
    prefix,bank,launcher=('association','association_metric','tools/launch_association_calibration.py') if a.family=='pair' else ('strong_rgb','strong_rgb_residual','tools/launch_strong_rgb_residual.py')
    decision=w/f'{prefix}_epoch{a.from_epoch}_forced_extension.json'
    lock=decision.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if decision.exists():raise ValueError('Extension decision already exists')
    evidence=[]
    for path in [a.witness]+a.earlier:
        path=Path(path);v=json.loads(path.read_text());m=v['manifest']
        if v['status']!='COMPLETE' or m['split']!='dev' or m['frames']!=4834:raise ValueError('Only complete Dev evidence')
        evidence.append(dict(path=str(path),sha256=digest(path),HOTA=v['trackeval']['HOTA'],epoch=m['calibration']['epoch'],config=m['config']))
    witness=evidence[0];earlier=evidence[1:]
    identity=lambda cfg:(cfg['detection_arm'],cfg['calibration_cue'],Path(cfg['association_checkpoint']).parent.name)
    if any(identity(x['config'])!=identity(witness['config']) for x in earlier):raise ValueError('Lookback evidence must be the same frontend/cue/seed')
    if witness['epoch']>a.from_epoch or any(x['epoch']>a.from_epoch-12 for x in earlier):raise ValueError('Lookback window mismatch')
    if {x['epoch'] for x in earlier}!=set(range(6,a.from_epoch-12+1,6)):raise ValueError('All earlier checkpoints are required')
    gain=witness['HOTA']-max(x['HOTA'] for x in earlier)
    if gain<=.2:raise ValueError('Witness does not force a common extension')
    record=dict(status='WAITING_FOR_ALL_SIX_FITS',from_epoch=a.from_epoch,to_epoch=to_epoch,lower_bound_gain_pp=gain,
                rule='An already complete Dev witness establishes a monotone lower bound on best-through-current gain over all checkpoints through12 epochs earlier. No pending current Dev outcome can reverse this forced extension.',evidence=evidence,Test_selection=False)
    def save():
        temporary=decision.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');os.replace(temporary,decision)
    save();state_path=w/f'{prefix}_fit_e{a.from_epoch}_state.json'
    while True:
        state=json.loads(state_path.read_text())
        if state['status']=='COMPLETE':break
        if state['status'] in {'FAILED','DEPENDENCY_FAILED'}:record['status']='DEPENDENCY_FAILED';save();return
        time.sleep(5)
    for cue in ['rgb','ir']:
        for seed in [0,1,2]:
            folder=w/bank/f'{cue}_s{seed}';completion=json.loads((folder/'complete.json').read_text())
            if completion['epochs']!=a.from_epoch:raise ValueError('Every fit must reach the common boundary')
            for name in ['history','complete']:
                source=folder/(name+'.json');target=folder/f'stage{a.from_epoch}_{name}.json'
                if target.exists():
                    if digest(source)!=digest(target):raise ValueError('Existing stage backup differs')
                else:shutil.copy2(source,target)
    next_state=w/f'{prefix}_fit_e{to_epoch}_state.json'
    if next_state.exists():raise ValueError('Next fit stage already exists')
    log=w/f'{prefix}_fit_e{to_epoch}_controller.log'
    command=[sys.executable,'-u',launcher,'--epochs',str(to_epoch),'--resume','--stage',f'e{to_epoch}']
    with log.open('ab',buffering=0) as handle:
        child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    record.update(status='EXECUTED_COMMON_EXTENSION',pid=child.pid,command=command,log=str(log));save()
    print(json.dumps(dict(status=record['status'],family=a.family,epochs=to_epoch,pid=child.pid)),flush=True)


if __name__=='__main__':main()
