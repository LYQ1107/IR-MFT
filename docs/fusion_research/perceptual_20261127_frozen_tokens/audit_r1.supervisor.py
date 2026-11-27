#!/usr/bin/env python3
"""Record actual ledger-worker identity, immutable sources and real exit status."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data3/liuyeqiang/IR-MFT')
WORK=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['engineer','fit','run','audit'])
    p.add_argument('--attempt',default='r1');a=p.parse_args();assert a.attempt.isalnum()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    study='frozen_tokens_20261127'
    base=WORK/study;base.mkdir(parents=True,exist_ok=True)
    entry='analyse_mft_frozen_tokens.py'
    mode='engineering'if a.mode=='engineer'else a.mode
    target=base/(mode+'_'+a.attempt+'.execution.json');log=base/(mode+'_'+a.attempt+'.stdout.txt')
    assert not target.exists()and not log.exists()
    names=['tools/analyse_mft_frozen_tokens.py','tools/mft_frozen_tokens.py','tools/run_mft_frozen_tokens.py','tests/test_mft_frozen_tokens.py','configs/perceptual_frozen_tokens_20261127.json']
    sources={n:sha(ROOT/n)for n in names};command=[sys.executable,'-u',str(ROOT/'tools'/entry),a.mode]
    started=time.monotonic();snapshots=[]
    def save(v):
        temporary=target.with_suffix('.tmp.json');temporary.write_text(json.dumps(v,indent=2)+'\n');os.replace(temporary,target)
    with log.open('xb',buffering=0)as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        expected=b'\0'.join(os.fsencode(s)for s in command)+b'\0'
        for _ in range(100):
            proc=Path('/proc')/str(child.pid);raw=(proc/'cmdline').read_bytes()
            snapshots.append(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            if raw==expected:
                fields=(proc/'stat').read_text().rsplit(') ',1)[1].split()
                visible=next((v.decode().split('=',1)[1]for v in(proc/'environ').read_bytes().split(b'\0')if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
                assert visible==''
                actual=dict(PID=child.pid,start_ticks=int(fields[19]),raw_argv_sha256=sha(proc/'cmdline'),CUDA_VISIBLE_DEVICES=visible)
                break
            time.sleep(.01)
        else:raise RuntimeError('Preserve worker; expected actual argv was not observed')
        value=dict(status='ACTUAL_LEDGER_WORKER_RUNNING',mode=a.mode,UTC_started=utc(),command=command,
            source_hashes=sources,supervisor_sha256=sha(Path(__file__)),actual_child_exit_code=None,
            identity=actual,startup_argv_snapshots=snapshots,log=str(log),GPU_work=False)
        save(value);print(json.dumps(value),flush=True);code=child.wait()
    value.update(status='CLOSED_ACTUAL_LEDGER_WORKER',actual_child_exit_code=code,UTC_finished=utc(),
        runtime_seconds=time.monotonic()-started,source_unchanged=all(sha(ROOT/n)==h for n,h in sources.items()),stdout_sha256=sha(log))
    save(value);print(json.dumps(value),flush=True)
    raise SystemExit(code if value['source_unchanged']else 1)


if __name__=='__main__':main()
