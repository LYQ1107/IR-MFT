#!/usr/bin/env python3
"""Complete Dev, frozen model runtime, then mandatory full Test; no Dev-positive gate."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_pseudo_ir_comparison import dump,sha256


def training_ledger(work,study,study_path):
    fits=json.loads((work/'structural_full_train_fit_state.json').read_text())
    assert fits['status']=='COMPLETE' and len(fits['jobs'])==6
    rows=[]
    for name,job in fits['jobs'].items():
        assert job['status']=='COMPLETE'
        folder=work/'structural_full_train_fit'/name
        path=folder/'full_train_complete.json'
        complete=json.loads(path.read_text())
        assert complete['status']=='COMPLETE_THREE_FULL_TRAIN_TRAVERSALS'
        assert complete['successful_steps_this_invocation']==20736
        assert complete['logical_Train_epochs_completed']==3 and len(complete['history'])==72
        assert complete['post_loop_source_verified'] and complete['full_frame_address_delivery_verified']
        weights={str(epoch):dict(checkpoint_sha256=sha256(folder/f'epoch{epoch:02d}.pt'),
            bytes=(folder/f'epoch{epoch:02d}.pt').stat().st_size) for epoch in study['snapshots']}
        rows.append(dict(id=name,cue=job['cue'],components=job['components'],fixed_seed=0,
            full_train_complete_sha256=sha256(path),logical_Train_epochs=3,optimizer_macro_passes=72,
            updates=20736,unique_Train_frames=27251,trainable_parameters=complete['trainable_parameters'],
            address_sha256=complete['full_train_address_sha256'],snapshots=weights,
            history=complete['history'],live_frame_passes=complete['live_frame_passes'],
            source_hashes=complete['source_hashes'],Train_GT_hashes=complete['Train_GT_hashes'],
            source_RGB_sha256=complete['source_RGB_sha256'],GPU_input_optimizer_proof_sha256=complete['GPU_input_optimizer_proof_sha256'],
            image_reader_statistics=complete['image_reader_statistics'],seconds=complete['seconds'],
            peak_own_GPU_GiB=complete['peak_gpu_GiB']))
    destination=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_FULL_TRAIN_TRAINING_LEDGER.json'
    assert not destination.exists()
    dump(destination,dict(status='COMPLETE_SIX_MATCHED_THREE_FULL_TRAIN_FITS',rows=rows,
        study_sha256=sha256(study_path),fixed_seed=0,additional_seed_work=False,
        scope='Every27251Trainframe in each of3logical traversals;72optimizer macro passes,not72fullTrainepochs. '
              'RGB/gray/IR and removals share exact actual addresses. No image/FPN disk copies or weights in Git.',
        goal_met=False,supported_innovations=0))


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work=Path(plan['work_root'])
    path=work/'structural_full_train_next_stage_state.json'
    lock=path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not path.exists()
    names=['tools/advance_structural_full_train.py','tools/export_structural_full_train_dev.py',
        'tools/register_structural_full_train_test.py','tools/benchmark_structural_full_train.py',
        'tools/benchmark_structural_factorial.py','tools/extract_structural_full_train.py',
        'tools/launch_structural_full_train_test.py','tools/replay_structural_full_train.py',
        'tools/export_structural_full_train_test.py','tools/structural_gate_observer.py']
    state=dict(status='WAITING_FOR_ALL50_FULL_TRAIN_DEV',controller_pid=os.getpid(),
        source_hashes={name:sha256(ROOT/name) for name in names},
        fixed_seed=0,additional_seed_work=False,goal_met=False,supported_innovations=0)
    dump(path,state)
    try:
        while True:
            dev_path=work/'structural_full_train_dev_state.json'
            dev=json.loads(dev_path.read_text()) if dev_path.exists() else {}
            if 'FAIL' in dev.get('status',''):
                raise RuntimeError('Preserve failed complete Dev family')
            if dev.get('status')=='COMPLETE':
                break
            time.sleep(5)
        assert all(sha256(ROOT/name)==digest for name,digest in state['source_hashes'].items())
        study_path=ROOT/'configs/perceptual_structural_fulltrain_study_20261123.json'
        study=json.loads(study_path.read_text())
        training_ledger(work,study,study_path)
        subprocess.run([sys.executable,'tools/export_structural_full_train_dev.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'tools/register_structural_full_train_test.py'],cwd=ROOT,check=True)
        state['status']='PROFILING_FROZEN_DEV_CHOICES_BEFORE_TEST'
        dump(path,state)
        for memory in [False,True]:
            command=[sys.executable,'-u','tools/benchmark_structural_full_train.py']
            if memory:
                command+=['--memory-inputs-and-components']
            log=work/('full_train_runtime_'+('memory' if memory else 'live')+'.log')
            with log.open('xb',buffering=0) as stream:
                process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='4'),
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(runtime_pid=process.pid,runtime_memory_inputs=memory,runtime_log=str(log))
            dump(path,state)
            assert process.wait()==0,str(log)
        state['status']='RUNNING_ALL_FROZEN_FULL_TRAIN_TEST'
        log=work/'launch_structural_full_train_test.log'
        with log.open('xb',buffering=0) as stream:
            process=subprocess.Popen([sys.executable,'-u','tools/launch_structural_full_train_test.py'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(Test_controller_pid=process.pid,Test_log=str(log))
        dump(path,state)
        assert process.wait()==0,str(log)
        subprocess.run([sys.executable,'tools/export_structural_full_train_test.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'tools/export_perceptual_official_tables.py'],cwd=ROOT,check=True)
        state['status']='COMPLETE_FULL_TRAIN_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT'
        dump(path,state)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_FULL_TRAIN_ADVANCER',error=repr(error))
        dump(path,state)
        raise


if __name__=='__main__':
    main()
