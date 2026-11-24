#!/usr/bin/env python3
"""Freeze full workflow CPU evidence; actual GPU gates remain mandatory."""
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from tools.launch_spatial_native import source_files
from tools.native_pyramid_experiment import CANONICAL_PROOF
from tools.neck72_runtime import require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.spatial_native_study import STUDY, arm_path, context, read


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    started = time.monotonic()
    plan, study, protocol, calibration, work, checkpoint, base = context()
    require_host_memory(2*2**30, maximum_workers=1)
    require_storage(work)
    output = work/'native_spatial_full_workflow_CPU_proof.json'
    assert not output.exists()
    sources = {name: sha256(ROOT/name) for name in source_files()}
    worker_path = work/'native_spatial_worker_CPU_proof.json'
    worker = read(worker_path)
    assert worker['status'] == 'COMPLETE_NINE_ARM_SPATIAL_WORKER_CPU_CANARY_NOT_FULL_PIPELINE'
    assert worker['study_sha256'] == sha256(STUDY)
    assert all(sources[name] == digest for name, digest in worker['source_hashes'].items())
    assert worker['actual_updates'] == 27 and len(worker['records']) == 9
    for row in worker['records']:
        assert worker['normal_arms'][row['fit_id']] == row['arm_sha256'] == sha256(arm_path(base, row['fit_id']))
        record = row['record']
        assert record['delivered_Train_frames'] == 12 and record['actual_updates'] == 3
        assert len(record['optimizer_rows']) == 3 and record['inference_B1_frames'] == 2
        assert len(record['source_zero_missing_checks']) == 2
        assert all(all(r[k] for k in ['fresh_direct_packets_exact', 'zero_full_head_exact',
            'learned_missing_full_head_exact']) for r in record['source_zero_missing_checks'])
        assert record['frozen_RGB_weights_BN_unchanged_before_explicit_head_precision_conversion']
        assert record['inference_RGB_state_unchanged']
    prototype = read(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_SPATIAL_CPU_PROOF.json')
    assert prototype['tests_passed'] == 10 and worker['tests_passed'] == 15
    log = work/'native_spatial_pipeline_pytest_R2.log'
    match = re.search(r'\b(\d+) passed\b', log.read_text())
    assert match and int(match.group(1)) >= 13
    first_log = work/'native_spatial_pipeline_pytest.log'
    assert '1 failed, 10 passed' in first_log.read_text()
    canonical = read(CANONICAL_PROOF)
    assert canonical['status'] == 'COMPLETE_NATIVE_STREAM_CANONICAL_ALL4834DEV_ALL39_CSV_EXACT'
    assert all(sha256(ROOT/name) == digest for name, digest in canonical['source_hashes'].items())
    assert all(sha256(ROOT/name) == digest for name, digest in sources.items())
    assert not torch.cuda.is_initialized()
    proof = dict(status='COMPLETE_SPATIAL_FULL_WORKFLOW_CPU_PROOF_PENDING_GPU',
        study_sha256=sha256(STUDY), source_hashes=sources, worker_proof_sha256=sha256(worker_path),
        new_pipeline_unit_checks=int(match.group(1)), distinct_unit_checks_with_unchanged_prototype_worker=25+int(match.group(1)),
        pipeline_test_log_sha256=sha256(log), preserved_initial_test_log_sha256=sha256(first_log),
        initial_test_failure='Missing-Dev test stub omitted keyword forwarding; fixed stub, retained first10passes and1failure.',
        actual_worker_records=9, real_Train_updates_in_reused_worker_evidence=27,
        real_Train_frames_unique=12, real_B1_frame_conditions_in_worker_evidence=18,
        inherited_canonical_all4834Dev_39metrics_CSV_proof_sha256=sha256(CANONICAL_PROOF),
        complete_Train_Dev_Test_cost_pipeline_implemented=True,
        CPU_contracts='All9negative fits retained; incomplete Dev and failed GPU prerequisites stop progression; '
            'semantic reuse, source/CSV precision, Test freeze, actual cached-prefix integrity, healthy sibling preservation, '
            'all47fields/original276CSV publication and negative wrong-IR numeric guard.',
        CPU_orchestration_fixtures_are_not_real_fits_or_accuracy=True,
        actual_GPU_engineering_pending=True, formal_fits_started=False, Dev_Test_accuracy_evaluated=False,
        new_neural_training_in_this_wrapper=False, GPU_work=False, image_or_FPN_disk_cache=False,
        saved_weights=False, fixed_seed=0, additional_seed_work=False,
        torch_version=str(torch.__version__), CUDA_runtime=torch.version.cuda,
        python_version=sys.version, peak_wrapper_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        seconds=time.monotonic()-started, goal_met=False, supported_innovations=0)
    dump(output, proof)
    print(json.dumps(dict(status=proof['status'], sources=len(sources), tests=int(match.group(1)))), flush=True)


if __name__ == '__main__':
    main()
