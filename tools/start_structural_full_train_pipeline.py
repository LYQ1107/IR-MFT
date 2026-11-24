#!/usr/bin/env python3
"""Activate the complete registered pipeline; GPU work still waits for old jobs/proof."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work=Path(plan['work_root'])
    ready_path=work/'structural_full_train_pipeline_registered.json'
    lock=ready_path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not ready_path.exists()
    for name in ['structural_full_train_fit_state.json','structural_full_train_dev_state.json',
                 'structural_full_train_next_stage_state.json']:
        assert not (work/name).exists(),'Inspect existing live/terminal handles; do not restart on timeout'
    study_path=ROOT/'configs/perceptual_structural_fulltrain_study_20261123.json'
    study=json.loads(study_path.read_text())
    assert sha256(ROOT/study['configuration'])==study['configuration_sha256']
    schedule_path=work/'structural_full_train_schedule/complete.json'
    schedule=json.loads(schedule_path.read_text())
    assert sha256(schedule_path)==study['schedule_proof_sha256']
    assert all(row['unique_Train_frames']==27251 and row['sample_exposures']==27648
               for row in schedule['logical_epochs'])
    proof_path=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_FULL_TRAIN_REPLAYER_PARITY.json'
    proof=json.loads(proof_path.read_text())
    assert proof['status']=='COMPLETE_NEW_FULL_TRAIN_REPLAYER_EXACT_CANONICAL_RGB_PARITY'
    assert proof['source_sha256']==sha256(ROOT/'tools/replay_structural_full_train.py')
    assert proof['all39metrics_exact'] and proof['official276column_CSV_exact'] and proof['full_track_files_exact']
    tests=work/'pytest_full_train_pipeline_full_20261123.log'
    assert '196 passed' in tests.read_text()
    tools=['structural_full_train_schedule','structural_full_train_inputs','register_structural_full_train',
        'register_structural_full_train_dev','train_structural_full_train','launch_structural_full_train_fits',
        'extract_structural_full_train','replay_structural_full_train','launch_structural_full_train_dev',
        'export_structural_full_train_dev','register_structural_full_train_test',
        'launch_structural_full_train_test','export_structural_full_train_test',
        'benchmark_structural_full_train','advance_structural_full_train','start_structural_full_train_pipeline']
    names=['tools/'+name+'.py' for name in tools]
    names+=['tools/structural_stream_loop.py','tools/structural_stream_training.py',
            'tools/structural_train_image_reader.py','tools/benchmark_structural_factorial.py',
            'tools/extract_structural_relative.py','tools/structural_relative_inference.py',
            'tools/structural_grid.py','tools/structural_correspondence.py','tools/build_correlation_cache.py',
            'tools/indexed_postprocess.py','tools/structural_gate_observer.py',
            'yolox/models/structural_factorial_repair.py','yolox/models/structural_relative_repair.py',
            'yolox/models/structural_relative_losses.py','yolox/models/frozen_head_repair.py',
            'yolox/models/recoverable_curriculum.py','yolox/models/yolo_head.py',
            'yolox/models/yolo_pafpn.py','yolox/models/darknet.py','yolox/models/network_blocks.py',
            'yolox/models/yolox.py','exps/yolox_mft26ir_visible.py',
            'tools/eval_mft26ir.py','tools/run_pseudo_ir_comparison.py',
            'yolox/tracker/byte_tracker.py','yolox/tracker/matching.py','yolox/tracker/kalman_filter.py',
            'yolox/tracker/basetrack.py','yolox/tracker/foveated_regions.py','yolox/tracker/priority_observations.py']
    subprocess.run([sys.executable,'-m','py_compile',*names],cwd=ROOT,check=True)
    names+=['configs/perceptual_mft_20261122.json',plan['protocol'],study['configuration'],
        'configs/perceptual_structural_fulltrain_study_20261123.json',
        'configs/perceptual_structural_fulltrain_dev_20261123.json',
        'configs/perceptual_single_seed_policy_20261123.json']
    names += [str(path.relative_to(ROOT)) for path in sorted(
        (ROOT/'configs/perceptual_structural_fulltrain_dev_cases_20261123').glob('*.json'))]
    source_hashes={name:sha256(ROOT/name) for name in names}
    dev_spec=ROOT/'configs/perceptual_structural_fulltrain_dev_20261123.json'
    assert json.loads(dev_spec.read_text())['registered_cases']==50
    assert len(list((ROOT/'configs/perceptual_structural_fulltrain_dev_cases_20261123').glob('*.json')))==50
    ready=dict(status='COMPLETE_REGISTERED_DEV_TEST_PIPELINE',study_sha256=sha256(study_path),
        source_hashes=source_hashes,full50Dev_specification_sha256=sha256(dev_spec),
        canonical_replay_proof_sha256=sha256(proof_path),full196test_log_sha256=sha256(tests),
        fullTrain_fits_started=False,exact_Test_choices_pending_all50Dev=True,
        mandatory_Test_roles_registered=13,physical_GPUs=['4','5'],
        GPU_gate='Current factorial Test NN completion, then queued liveGPU2step optimizer parity must complete first.',
        scope='Pipeline tooling/roles registered and CPUvalidated, not future trained-model accuracy or completed GPUstages.',
        goal_met=False,supported_innovations=0)
    dump(ready_path,ready)
    pids={}
    for name in ['launch_structural_full_train_fits','launch_structural_full_train_dev','advance_structural_full_train']:
        log=work/(name+'.log')
        with log.open('xb',buffering=0) as stream:
            process=subprocess.Popen([sys.executable,'-u','tools/'+name+'.py'],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        pids[name]=dict(pid=process.pid,log=str(log))
    dump(work/'structural_full_train_pipeline_pids.json',pids)
    print(json.dumps(dict(status=ready['status'],controllers=pids,formal_GPU_work_started=False)))


if __name__=='__main__':
    main()
