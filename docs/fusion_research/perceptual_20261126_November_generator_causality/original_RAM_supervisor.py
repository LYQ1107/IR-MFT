import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/data3/liuyeqiang/IR-MFT')
WORK=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
ARCHIVE=ROOT/'diffv2ir_final_20261104'
STEM=WORK/'fixed_November_actual_generator_causality_20261126_r3'
RAM=WORK/'fixed_November_private_verified_RAM_20261126_r1'
PLAN=ROOT/'configs/perceptual_fixed_November_causality_probe_20261126.json'
SOURCE=ROOT/'tools/audit_fixed_November_generator_causality.py'
RECEIPT=Path(str(STEM)+'.execution.json')
LOG=Path(str(STEM)+'.log')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save():RECEIPT.write_text(json.dumps(record,indent=2)+'\n')
def identity(p,argv):
    expected=b'\0'.join(x.encode() for x in argv)+b'\0'
    for _ in range(200):
        try:
            folder=Path('/proc')/str(p.pid);raw=(folder/'cmdline').read_bytes();s=(folder/'stat').read_text();f=s[s.rfind(')')+2:].split()
            if raw==expected:return dict(PID=p.pid,start_ticks=int(f[19]),raw_argv_sha256=hashlib.sha256(raw).hexdigest())
        except FileNotFoundError:pass
        time.sleep(.01)
    raise RuntimeError('Cannot bind owned process after exec')
def original_phase():
    p=Path('/proc')/str(plan['original_controller']['PID']);s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split();raw=(p/'cmdline').read_bytes();pin=plan['original_controller']
    assert int(f[19])==pin['start_ticks'] and hashlib.sha256(raw).hexdigest()==pin['raw_argv_sha256'] and f[0]!='Z'
    state=read(WORK/'native_pyramid_study_state.json');assert state['status'] in plan['allowed_original_phases'],state['status'];return state['status']
def stop_owned(p,group=False):
    if p.poll() is not None:return
    if group:os.killpg(p.pid,signal.SIGTERM)
    else:p.terminate()
    try:p.wait(timeout=12)
    except subprocess.TimeoutExpired:
        if group:os.killpg(p.pid,signal.SIGKILL)
        else:p.kill()
        p.wait(timeout=10)

plan=read(PLAN);assert not RECEIPT.exists() and not LOG.exists() and not STEM.exists() and not RAM.exists()
phase=original_phase();memory={k:int(v.strip().split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())};assert memory['MemAvailable']>48*1024**2
RAM.mkdir();(RAM/'cache/ram').mkdir(parents=True);(RAM/'evidence').mkdir();(RAM/'cache/ram/clip_text').symlink_to(ARCHIVE/'cache/ram/clip_text',target_is_directory=True)
artifacts=ROOT/'thermalgen_probe2_20261102/evidence/artifacts.jsonl';entries=[json.loads(s) for s in artifacts.read_text().splitlines() if s.strip()]
expected={name:next(r for r in entries if r['filename']==filename and r['verified']) for name,filename in [('model','IR-500k/finetuned_checkpoints/M3FD.ckpt'),('sam','sam_vit_b_01ec64.pth')]}
assert sum(r['size_bytes'] for r in expected.values())<plan['maximum_anonymous_RAM_weight_cache_GiB']*2**30
started=time.monotonic();owned={};handles=[];gpu=None;exit_code=1
record=dict(status='CPU_ONLY_VERIFIED_ANONYMOUS_RAM_PRELOAD_RUNNING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(SOURCE),plan_sha256=sha(PLAN),supervisor_source_sha256=sha(__file__),original_stage_ram_source_sha256=sha(ARCHIVE/'scripts/stage_ram.py'),original_artifact_manifest_sha256=sha(artifacts),private_RAM_root=str(RAM),output_directory=str(STEM),expected_RAM_bytes=sum(r['size_bytes'] for r in expected.values()),RAM_limit_GiB=8,disk_weight_copies=0,original_cache_links_modified=False,original_training_signalled=False,original_phase_at_start=phase,staging_processes={},GPU_probe_started=False,actual_NVML_samples=[],exit_code=None)
save();print(json.dumps(record),flush=True)
try:
    for name in ('model','sam'):
        code="from pathlib import Path; import sys; sys.path.insert(0,"+repr(str(ARCHIVE/'scripts'))+"); import stage_ram; stage_ram.R=Path("+repr(str(RAM))+"); stage_ram.os.sys.argv=['private-original-stage-ram',"+repr(name)+"]; stage_ram.main()"
        argv=['/usr/bin/python3','-u','-c',code];log=(RAM/(name+'.log')).open('xb');handles.append(log)
        p=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owned[name]=p;record['staging_processes'][name]=dict(identity(p,argv),command=argv,log=str(RAM/(name+'.log')),CPU_only=True)
    save()
    while True:
        record['last_original_phase']=original_phase();ready=[]
        for name,p in owned.items():
            assert p.poll() is None,'Owned RAM holder exited before inference: '+name
            path=RAM/'evidence'/('ram_'+name+'_ready.json')
            if not path.exists():continue
            value=read(path);assert value['pid']==p.pid and len(value['files'])==1
            item=value['files'][0];assert item['sha256']==expected[name]['sha256'] and item['bytes']==expected[name]['size_bytes'] and Path(item['path']).is_symlink() and Path(item['path']).stat().st_size==item['bytes']
            ready.append(name)
        record.update(last_observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),RAM_ready=ready,staging_seconds=time.monotonic()-started);save()
        if len(ready)==2:break
        assert time.monotonic()-started<plan['maximum_CPU_RAM_staging_seconds'],'Bounded CPU RAM staging expired'
        time.sleep(5)
    record['RAM_ready_original_receipts']={n:read(RAM/'evidence'/('ram_'+n+'_ready.json')) for n in owned}
    record.update(status='VERIFIED_RAM_READY_STARTING_GPU4_PROBE',RAM_staging_seconds=time.monotonic()-started);save();print(json.dumps({k:v for k,v in record.items() if k not in ['staging_processes','actual_NVML_samples']}),flush=True)
    scratch=Path(str(STEM)+'_environment');scratch.mkdir()
    for name in ['tmp','huggingface','torch','cache','cuda','triton','matplotlib']:(scratch/name).mkdir()
    argv=[str(ARCHIVE/'environments/diffv2ir/bin/python'),'-u',str(SOURCE),'--asset-cache-root',str(RAM/'cache/ram'),'--output-directory',str(STEM)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',PROBE_NVML_PHYSICAL_GPU='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',HF_HOME=str(scratch/'huggingface'),TORCH_HOME=str(scratch/'torch'),XDG_CACHE_HOME=str(scratch/'cache'),CUDA_CACHE_PATH=str(scratch/'cuda'),TRITON_CACHE_DIR=str(scratch/'triton'),MPLCONFIGDIR=str(scratch/'matplotlib'),TMPDIR=str(scratch/'tmp'),PYTHONPATH=':'.join(str(ARCHIVE/p) for p in ['scripts','DiffV2IR','DiffV2IR/stable_diffusion']))
    log=LOG.open('xb');handles.append(log);gpu_start=time.monotonic();gpu=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record.update(status='BOUNDED_GPU4_GENERATOR_PROBE_RUNNING_WITH_VERIFIED_RAM',GPU_probe_started=True,GPU_child_identity=identity(gpu,argv),command=argv,source_sha256=sha(SOURCE),GPU_probe_started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
    while gpu.poll() is None:
        record['last_original_phase']=original_phase();assert time.monotonic()-gpu_start<plan['maximum_child_wall_seconds'],'Bounded GPU4 probe wall expired'
        q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True)
        record['actual_NVML_samples'].append(dict(seconds=time.monotonic()-gpu_start,project_original_and_auxiliary=[s.strip() for s in q.stdout.splitlines() if s.split(',')[0].strip()==str(gpu.pid) or ('GPU-daa9b388-4540-242c-83d7-3261bb232a7c' in s or 'GPU-f07a0798-7ddc-ec7c-a27c-d47124bbfe81' in s)]));save();time.sleep(2)
    exit_code=gpu.wait();record.update(GPU_probe_exit_code=exit_code,complete_report_exists=(STEM/'complete.json').is_file(),GPU_probe_wall_seconds=time.monotonic()-gpu_start,log_sha256=sha(LOG));assert exit_code==0 and record['complete_report_exists'],'Preserve generator probe failure'
    record['status']='COMPLETE_RAM_STAGING_AND_EIGHT_TRAIN_GPU4_GENERATOR_PROBE';exit_code=0
except BaseException as error:
    record.update(status='FAILED_OR_CONTEXT_CHANGED_AUXILIARY_PROBE_PRESERVED',error=repr(error))
finally:
    if gpu is not None:stop_owned(gpu,group=True)
    for name in owned:(RAM/'cache'/('STOP_RAM_'+name)).touch()
    for name,p in owned.items():
        try:p.wait(timeout=7)
        except subprocess.TimeoutExpired:stop_owned(p,group=True)
        record['staging_processes'][name]['actual_exit_code']=p.returncode
    for h in handles:h.close()
    record.update(exit_code=exit_code,total_wall_seconds=time.monotonic()-started,owned_RAM_holders_released=True,original_training_signalled=False);save();print(json.dumps({k:v for k,v in record.items() if k!='actual_NVML_samples'}),flush=True)
raise SystemExit(exit_code)
