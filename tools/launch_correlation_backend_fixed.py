#!/usr/bin/env python3
"""All3 observation cues x identical frozen association cues/seeds, Dev only."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.run_pseudo_ir_comparison import dump,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    main=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(main['work_root'])
    plan_path=ROOT/'configs/perceptual_correlation_backend_fixed_20261123.json';plan=json.loads(plan_path.read_text())
    evidence=Path(plan['work_root']);path=work/'correlation_backend_fixed_state.json';lock=path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Fixed correlation backend already scheduled')
    smoke=json.loads((evidence/'transport/dev/BT-005/smoke_8/complete.json').read_text())
    point_source=ROOT/'tools/extract_correlation_backend_points.py';feature_source=ROOT/'tools/extract_correlation_backend_features.py'
    if(smoke['status']!='SMOKE_COMPLETE'or smoke['GT_read']or smoke['future_input']or set(smoke['outputs'])!={'rgb','gray','ir'}
            or smoke['source_hashes']['extractor']!=sha256(point_source)):
        raise ValueError('Actual8-frame all-cue point check required')
    folder=ROOT/'configs/perceptual_correlation_backend_fixed_dev_20261123';folder.mkdir(exist_ok=False);jobs={}
    for observation in ['rgb','gray','ir']:
        for cue in ['rgb','ir']:
            for seed in range(3):
                source=ROOT/f'configs/perceptual_pair_selected_20261123/K_rgb_{cue}_s{seed}.json';config=json.loads(source.read_text())
                name=f'CB_{observation}_{cue}_s{seed}'
                config.update(id=name,detection_arm=observation,observation_cue=observation,source_config_sha256=sha256(source),
                    association_checkpoint_sha256=sha256(config['association_checkpoint']),
                    correlation_backend_plan_sha256=sha256(plan_path),
                    registration='Same exact RGB-frontend-selected frozen F0 head for each cue/seed across all3 observation streams; no new fit/threshold search. Full Dev interaction diagnosis, no Test or supported contribution.')
                target=folder/(name+'.json');dump(target,config);jobs[name]=dict(status='PENDING',config=str(target),sha256=sha256(target))
    state=dict(status='PRODUCING_UNLABELED_EVIDENCE',jobs=jobs,evidence_jobs={},plan_sha256=sha256(plan_path),Test_read=False,
        source_sha256={str(p.relative_to(ROOT)):sha256(p)for p in [point_source,feature_source,ROOT/'tools/correlation_backend_io.py']})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MFT26IR_ROOT=main['data_root'])
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    active={}
    def start(key,command,gpu=None,registry=None):
        registry=state['evidence_jobs']if registry is None else registry
        if sha256(plan_path)!=state['plan_sha256']or any(sha256(ROOT/p)!=value for p,value in state['source_sha256'].items()):
            raise ValueError('Frozen evidence source changed')
        log=work/(key+'.correlation_backend_fixed.log')
        with log.open('ab',buffering=0)as handle:
            proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpu or''),stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        active[key]=(proc,registry);registry.setdefault(key,{}).update(status='RUNNING',pid=proc.pid,command=command,log=str(log),physical_gpu=gpu)
    def reap():
        for key,(proc,registry)in list(active.items()):
            code=proc.poll()
            if code is not None:registry[key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del active[key]
    for seq in ['BT-005','SN-013']:
        start('points_'+seq,[sys.executable,'-u',str(point_source),'--split','dev','--sequence',seq])
    pending=['BT-005','SN-013']
    while active or pending:
        reap()
        if any(j['status']=='FAILED'for j in state['evidence_jobs'].values()):
            if not active:state['status']='FAILED';dump(path,state);return
            dump(path,state);time.sleep(5);continue
        fitting=json.loads((work/'correlation_recoverability_fit_state.json').read_text())
        ready=all(fitting['jobs'].get(cue+'_full24',{}).get('status')=='COMPLETE'for cue in ['ir','gray'])
        occupied={j['physical_gpu']for j in fitting['jobs'].values()if j['status']=='RUNNING'}
        dev_path=work/'correlation_recoverability_pilot_dev_state.json'
        if dev_path.exists():
            dev=json.loads(dev_path.read_text());occupied|={j['physical_gpu']for j in dev.get('producers',{}).values()if j['status']=='RUNNING'}
        feature_busy=any(key.startswith('features_')for key in active)
        free=[gpu for gpu in ['2','3']if gpu not in occupied]
        if pending and ready and not feature_busy and free:
            seq=pending.pop(0);start('features_'+seq,[sys.executable,'-u',str(feature_source),'--split','dev','--sequence',seq],free[0])
        dump(path,state);time.sleep(5)
    for seq in ['BT-005','SN-013']:
        for kind in ['features','transport']:
            manifest=json.loads((evidence/kind/'dev'/seq/'complete.json').read_text())
            if manifest['status']!='COMPLETE'or manifest['GT_read']or set(manifest['outputs'])!={'rgb','gray','ir'}:
                raise ValueError('Complete all-cue unlabeled evidence required')
    state['status']='RUNNING'
    while True:
        reap()
        for key,job in jobs.items():
            if len(active)>=2 or any(j['status']=='FAILED'for j in jobs.values()):break
            if job['status']!='PENDING':continue
            if sha256(job['config'])!=job['sha256']:raise ValueError('Frozen config changed')
            command=[sys.executable,'-u','tools/replay_perceptual_mft.py','--plan',str(plan_path),'--config',job['config'],
                '--split','dev','--output',str(work/'correlation_backend_fixed_dev'/key)]
            start(key,command,registry=jobs)
        if not active:
            if all(j['status']=='COMPLETE'for j in jobs.values()):state['status']='COMPLETE'
            elif any(j['status']=='FAILED'for j in jobs.values()):state['status']='FAILED'
        dump(path,state)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
