"""Bounded two-slot subprocess phases with actual identities and durable results."""
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import re
import subprocess
import time

from tools.native_pyramid_experiment import ROOT
from tools.neck72_runtime import require_host_memory,require_storage
from tools.run_pseudo_ir_comparison import dump


def process_identity(pid,command=None):
    """Bind the actual exec argv, not the transient pre-exec parent argv."""
    path=Path('/proc')/str(pid);expected=None if command is None else b'\0'.join(os.fsencode(s) for s in command)+b'\0'
    for _ in range(100):
        stat=(path/'stat').read_text().rsplit(')',1)[1].split();raw=(path/'cmdline').read_bytes()
        if stat[0]=='Z':raise RuntimeError('Child exited before its executable identity was observed')
        if expected is None or raw==expected:
            return dict(pid=pid,start_ticks=int(stat[19]),command_sha256=hashlib.sha256(raw).hexdigest())
        time.sleep(.01)
    raise RuntimeError('Actual child argv did not match its registered launch command')


class Queue:
    def __init__(self,work,base,state,data_root,guard,*,poll_seconds=5,storage_guard=require_storage,
                 memory_guard=require_host_memory,identity_reader=process_identity,spawn=subprocess.Popen,
                 cwd=ROOT,force_CPU_for_artificial_test=False):
        self.work,self.base,self.state,self.data_root=Path(work),Path(base),state,str(data_root)
        self.guard,self.poll_seconds=guard,poll_seconds
        self.storage_guard,self.memory_guard=storage_guard,memory_guard
        self.identity_reader,self.spawn,self.cwd=identity_reader,spawn,Path(cwd)
        self.force_CPU_for_artificial_test=force_CPU_for_artificial_test
        if force_CPU_for_artificial_test and not state.get('artificial_fixture'):
            raise ValueError('CPU fixture override is unavailable to a formal controller')
        self.active={}
    def save(self):dump(self.base/'state.json',self.state)
    def jobs(self,phase,tasks,gpu,check):
        if not phase or not phase.replace('_','').isalnum() or phase in self.state['phases'] or self.active:
            raise ValueError('Every new phase requires a complete prior-phase barrier')
        entries={t['id']:dict(t,status='PENDING') for t in tasks}
        if not entries or len(entries)!=len(tasks) or any(not re.fullmatch(r'[A-Za-z0-9_-]+',name) for name in entries):
            raise ValueError('Unique safe nonempty job identifiers required')
        if any(gpu and t.get('physical_gpu','4') not in ['4','5'] for t in tasks):
            raise ValueError('Only physical GPU4/5 are permitted')
        self.state['phases'][phase]=entries;self.state['status']=phase;self.save();failure=None
        while True:
            try:self.guard()
            except Exception as error:failure=repr(error)
            for slot,(name,process) in list(self.active.items()):
                item=entries[name];observed=time.monotonic();code=process.poll()
                if code is None:
                    item['last_live_monotonic']=observed
                    try:
                        lines=(Path('/proc')/str(process.pid)/'status').read_text().splitlines()
                        rss=next(int(line.split()[1])*1024 for line in lines if line.startswith('VmRSS:'))
                        item['observed_peak_RSS_bytes']=max(item.get('observed_peak_RSS_bytes',0),rss)
                        if rss>8*2**30:
                            process.terminate();failure='Own child exceeded8GiB;healthy sibling retained'
                    except (FileNotFoundError,StopIteration):pass
                    continue
                del self.active[slot];item['exit_code']=code
                item['completion_observed_UTC']=datetime.now(timezone.utc).isoformat()
                item['worker_wall_upper_seconds']=observed-item['started_monotonic']
                item['worker_wall_lower_seconds']=max(0.,item.get('last_live_monotonic',item['started_monotonic'])-item['started_monotonic'])
                if code:
                    item['status']='FAILED';failure='Preserve failed child: '+item['log']
                else:
                    try:
                        if check(item) is False:raise ValueError('Child returned0without complete original evidence')
                        item['status']='COMPLETE'
                    except Exception as error:
                        item.update(status='FAILED_EVIDENCE',error=repr(error));failure=repr(error)
            if failure is None:
                for slot in (['4','5'] if gpu else ['CPU0','CPU1']):
                    if slot in self.active:continue
                    pending=[(name,t) for name,t in entries.items() if t['status']=='PENDING' and
                        (not gpu or t.get('physical_gpu',slot)==slot)]
                    if not pending:continue
                    try:
                        self.storage_guard(self.work,growth_bytes=int(.5*2**30))
                        self.memory_guard(8*2**30,maximum_workers=2 if not self.active and len(pending)>1 else 1)
                    except RuntimeError as error:
                        self.state['resource_wait_reason']=str(error);continue
                    name,item=pending[0];log=self.base/'logs'/(phase+'_'+name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=slot if gpu and not self.force_CPU_for_artificial_test else '',
                        MFT26IR_ROOT=self.data_root,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    launched=time.monotonic();launch_UTC=datetime.now(timezone.utc).isoformat()
                    with log.open('xb',buffering=0) as stream:
                        process=self.spawn(item['command'],cwd=self.cwd,env=env,stdin=subprocess.DEVNULL,
                            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    item.update(status='RUNNING',pid=process.pid,slot=slot,log=str(log),started_monotonic=launched,
                        started_UTC=launch_UTC,argv_identity_observed=False)
                    self.active[slot]=name,process;self.save()
                    try:
                        item.update(self.identity_reader(process.pid,item['command']),argv_identity_observed=True)
                    except Exception as error:
                        failure='Launched child identity could not be verified: '+repr(error)
                        item['identity_observation_error']=repr(error)
                    self.state.pop('resource_wait_reason',None)
                    if failure:break
            if failure:self.state['failure']=failure
            self.save()
            if failure and not self.active:raise RuntimeError(failure)
            if not self.active and all(t['status']=='COMPLETE' for t in entries.values()):return
            time.sleep(self.poll_seconds)
