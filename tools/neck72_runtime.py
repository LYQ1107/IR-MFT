"""Seed0 resource guards for the unfinished neck72 downstream study.

No scheduler or implicit GPU launch. The existing structural/native/CGR chain
must finish and its actual processes must exit before these workers use CUDA.
"""
import json
import os
from pathlib import Path
import subprocess

from tools.run_pseudo_ir_comparison import sha256

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = 'neck72_worker_interfaces_registered.json'
RESOURCE_GUARD_REVISION = 2


def read(path):
    return json.loads(Path(path).read_text())


def require_seed0(seed):
    if seed != 0:
        raise ValueError('Current user policy permits only seed0; no seed1/2 inference or fitting')


def predecessor_state_ready(state, report, process_finished):
    if 'FAIL' in state.get('status', ''):
        raise RuntimeError('Preserve the failed CGR predecessor before any neck72 GPU work')
    if (state.get('status') != 'COMPLETE_CGR_BACKFILL_REQUIRES_SCIENTIFIC_ASSESSMENT'
            or report.get('status') != 'COMPLETE_ALL6_TRAINED_CGR_FITS_FROZEN12_FULL7TEST_BACKFILL'):
        return False
    if not state.get('workflow_sha256') or state['workflow_sha256'] != report.get('workflow_sha256'):
        raise ValueError('CGR state and published report version disagree')
    if not state.get('controller_pid') or not process_finished(state['controller_pid']):
        return False
    phases = state.get('phases', {})
    if not phases or any(not jobs for jobs in phases.values()):
        return False
    for jobs in phases.values():
        for job in jobs.values():
            if (job.get('status') != 'COMPLETE' or job.get('exit_code') != 0
                    or not job.get('pid') or not process_finished(job['pid'])):
                return False
    return True


def gpu_predecessor_ready(work):
    from tools.correlation_recovery_workflow import predecessor_ready
    from tools.native_remaining_test_backfill import finished
    if not predecessor_ready(work):
        return False
    state_path = Path(work) / 'correlation_recovery_workflow_state.json'
    report_path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_CGR_FORMAL_BACKFILL.json'
    if not state_path.exists() or not report_path.exists():
        return False
    state = read(state_path)
    workflow = ROOT / 'configs/perceptual_correlation_recovery_workflow_20261124.json'
    if state.get('workflow_sha256') != sha256(workflow):
        raise ValueError('Expected the currently pinned CGR predecessor version')
    return predecessor_state_ready(state, read(report_path), finished)


def require_storage(work, growth_bytes=0):
    from tools.native_pyramid_training_loop import guard_storage
    stats = guard_storage(work)
    if (growth_bytes < 0 or stats['work_file_bytes'] + growth_bytes > stats['work_cap_bytes']
            or stats['free_bytes'] - growth_bytes < stats['reserve_bytes']):
        raise RuntimeError('neck72 output bound would exceed the64GiB cap or40GiB independent reserve')
    return stats


def require_host_memory(incoming_bytes, maximum_workers=4):
    """Reserve40GiB after the caller's bounded incoming worker allocation.

    The legacy default remains conservative. The phase controller admits at
    most two workers and never overlaps CPU/GPU phases; each child reserves its
    own allocation because MemAvailable already excludes a resident peer's RAM.
    No image cache or scientific training-budget change is made here.
    """
    from tools.structural_train_image_reader import available_memory_bytes
    if incoming_bytes < 0 or not 1 <= maximum_workers <= 4:
        raise ValueError('Invalid bounded producer memory request')
    available = available_memory_bytes()
    reserve = 40 * 2**30
    if available - maximum_workers * incoming_bytes < reserve:
        raise RuntimeError('neck72 producers must retain40GiB of available host memory')
    return dict(MemAvailable_bytes=available, reserve_bytes=reserve,
                incoming_bytes_each=incoming_bytes, maximum_workers=maximum_workers)


def require_gpu(work, seed=0, role='data'):
    require_seed0(seed)
    if role not in {'data', 'probe', 'fit'}:
        raise ValueError('Unknown neck72 GPU worker role')
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'4', '5'}:
        raise ValueError('neck72 workers permit physicalGPU4/5 only')
    if not gpu_predecessor_ready(work):
        raise RuntimeError('Wait for complete structural/native/CGR exports and actual process exit')
    require_host_memory(8 * 2**30, maximum_workers=1)
    registry = read(Path(work) / REGISTRY)
    assert registry['status'] == 'COMPLETE_CPU_NECK72_WORKER_INTERFACE_REGISTRATION_PENDING_GPU'
    assert registry['resource_guard_revision'] == RESOURCE_GUARD_REVISION
    assert all(sha256(ROOT / p) == h for p, h in registry['source_hashes'].items())
    assert all(sha256(p) == h for p, h in registry['artifact_hashes'].items())
    if role == 'fit':
        proof = read(Path(work) / 'neck72_actual_GPU_engineering' / ('gpu' + physical) / 'complete.json')
        assert proof['status'] == 'COMPLETE_NECK72_GPU_INPUT_OPTIMIZER_AND_EXACT_ZERO_FALLBACK'
        assert proof['worker_interfaces_sha256'] == sha256(Path(work) / REGISTRY)
        assert proof['physical_gpu'] == physical and proof['fixed_seed'] == 0
        assert proof['Train_updates_each'] == 20 and proof['matched_conditions_checked'] == 8
        assert proof['RGB_core_and_encoder_BN_unchanged'] and not proof['Dev_or_Test_fitting']
    policy = read(ROOT / 'configs/perceptual_gpu_handoff_20261123.json')
    seed_policy = read(ROOT / 'configs/perceptual_single_seed_policy_20261123.json')
    assert seed_policy['fixed_seed'] == 0 and not seed_policy['additional_seed_inference']
    uuid = subprocess.check_output(['nvidia-smi', '-i', physical, '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    assert uuid == policy['future_gpu_uuids'][physical]
    if role == 'fit':
        assert proof['GPU_uuid'] == uuid
    import torch
    assert torch.cuda.device_count() == 1
    require_storage(work)
    return physical, uuid
