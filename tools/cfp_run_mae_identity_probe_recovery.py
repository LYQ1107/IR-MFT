#!/usr/bin/env python3
"""Observe actual continuousprobe, then realepoch2exit/resume onowned idleGPU1."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from cfp.provenance import sha256,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();proof=a.root/'mae_identity_probe_original_restart_launches_v1.json';d=json.loads(proof.read_text());entry=d['launches'][0];pid=entry['pid']
    while True:
        try:os.kill(pid,0)
        except ProcessLookupError:break
        time.sleep(5)
    continuous=Path(entry['output']);m=json.loads((continuous/'manifest.json').read_text())
    if m['status']!='COMPLETE':raise ValueError('Actualcontinuousprobe mustcomplete; norestart-on-timeout')
    entry.update(exit_code=0,log_sha256=sha256(entry['log']));write_json(proof,d)
    out=a.root/'mae_identity_probe_original_ce_triplet_SN-015_s0_v1_restarted'
    for phase in ['stop','resume']:
        # ActualownerGPU3remainingwork guaranteespretrainstillwaitingbeforethisshortaudit.
        queued=json.loads((a.root/'visible_token_fish_pretraining_all6_driver_v2.json').read_text())
        if queued['status']!='WAITING_BOTH_ACTUAL_NATIVE_OWNERS' or queued['launches']:raise ValueError('NewGPU1pretrainingowner active; do notinterfere')
        argv=list(entry['command']);argv[argv.index('--output')+1]=str(out)
        if phase=='stop':argv+=['--stop-after-epoch','2']
        else:argv+=['--resume']
        log=Path(str(out)+'.'+phase+'.log');event=dict(output=str(out),phase=phase,command=argv,log=str(log))
        with log.open('x') as stream:
            child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'));event['pid']=child.pid;d['launches'].append(event);write_json(proof,d);code=child.wait()
        event.update(exit_code=code,log_sha256=sha256(log));write_json(proof,d)
        if code:raise RuntimeError('Actualprobe restartchild failed')
    d['status']='COMPLETE_ACTUAL_THREE_PROBE_LAUNCHES';write_json(proof,d)

if __name__=='__main__':main()
