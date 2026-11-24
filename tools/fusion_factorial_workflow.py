"""Independent follow-up contracts; never modify a live parent registration."""
import copy
import hashlib
import json
from pathlib import Path
import sys

from tools.fusion_factorial_rules import CELLS, module_kwargs
from tools.native_pyramid_experiment import STUDY, context as parent_context, read
from tools.native_pyramid_train_data import schedule_proof
from tools.neck72_runtime import require_host_memory, require_storage
from tools.neck72_runtime_study import exited
from tools.run_pseudo_ir_comparison import sha256

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = ROOT/'configs/perceptual_native_contrast_quality_factorial_20261124.json'
REGISTRY = 'fusion_factorial_pipeline_registered.json'
AUTHORIZATION = 'authorization.json'
FULL_PIPELINE = ('tools/train_fusion_factorial.py', 'tools/evaluate_fusion_factorial.py',
                 'tools/launch_fusion_factorial.py', 'tools/benchmark_fusion_factorial.py',
                 'tools/export_fusion_factorial.py')


def context():
    plan, study, protocol, calibration, work, checkpoint = parent_context()
    science = read(SCIENCE)
    assert science['parent_study_sha256'] == sha256(STUDY)
    assert science['execution']['fixed_seed'] == 0
    assert science['execution']['physical_gpus'] == ['4', '5']
    return plan, study, protocol, calibration, work, checkpoint, work/'fusion_factorial'


def registered(work):
    value = read(Path(work)/REGISTRY)
    assert value['status'] == 'REGISTERED_COMPLETE_FACTORIAL_PIPELINE_PENDING_DEV_GATE_AND_GPU_ENGINEERING'
    assert value['complete_GPU_Dev_Test_pipeline_implemented']
    assert value['science_sha256'] == sha256(SCIENCE)
    assert set(FULL_PIPELINE).issubset(value['source_hashes'])
    assert all(sha256(ROOT/name) == h for name, h in value['source_hashes'].items())
    assert all(sha256(path) == h for path, h in value['artifact_hashes'].items())
    return value


def release_state_ready(state, identity, report, case_ids, process_exited=exited):
    """A state file or controller exit alone cannot release either GPU."""
    status = state.get('status', '')
    if 'FAIL' in status or 'MISMATCH' in status:
        raise RuntimeError('Preserve and resolve the unfinished fresh-runtime predecessor')
    if status != 'COMPLETE_NECK72_FULL_COMPONENT_PREFIX_RUNTIME_SOURCE_MATCHED':
        return False
    if (report.get('status') != 'COMPLETE_NECK72_FRESH_IMAGE_TEST_COST_MEASUREMENTS' or
            not report.get('all_frozen_cases_passed') or not case_ids or
            len(report.get('rows', [])) != len(case_ids) or
            {r['id'] for r in report.get('rows', [])} != set(case_ids) or
            not all(r.get('source_equivalence_passed') and r.get('valid_full_component_cost_for_preserved_accuracy')
                    for r in report.get('rows', []))):
        return False
    assert state['controller_pid'] == identity['pid']
    if not process_exited(state['controller_pid'], identity=identity): return False
    if set(state.get('phases', {})) != {'dev', 'test'}: return False
    for jobs in state['phases'].values():
        if set(jobs) != set(case_ids): return False
        for job in jobs.values():
            if (job.get('status') != 'COMPLETE' or job.get('exit_code') != 0 or
                    not job.get('source_equivalence_passed') or not job.get('start_ticks')):
                return False
            if not process_exited(job['pid'], command=job['command'], identity=job): return False
    return True


def predecessor_ready(work):
    work = Path(work)
    path = work/'neck72_fresh_runtime_state.json'
    if not path.exists(): return False
    state = read(path)
    if 'FAIL' in state['status'] or 'MISMATCH' in state['status']:
        raise RuntimeError('Preserve predecessor diagnostics; do not bypass its source gates')
    if state['status'] != 'COMPLETE_NECK72_FULL_COMPONENT_PREFIX_RUNTIME_SOURCE_MATCHED': return False
    report = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261124_NECK72_FRESH_RUNTIME_TEST.json'
    if not report.exists(): return False
    frozen = read(work/'neck72_Test_selection_frozen.json')  # configurations, not Test scores
    identity = read(work/'neck72_fresh_runtime_controller_identity.json')
    assert state['specification_sha256'] == identity['specification_sha256']
    return release_state_ready(state, identity, read(report), [r['id'] for r in frozen['cases']])


def make_joint_arm(study, mode):
    if mode not in ['quality', 'temporal']: raise ValueError('Use the frozen parent primary mode')
    value = copy.deepcopy(study)
    value.update(fit_id='FQ00_'+mode, mode=mode, cue='ir', removal='joint_no_contrast_quality',
        parent_study_sha256=sha256(STUDY), factorial_science_sha256=sha256(SCIENCE),
        module_kwargs=module_kwargs(study, mode, '00'))
    return value


def validate_history(saved, epochs, study):
    """Verify actual full-Train delivery at a checkpoint, not a sampled budget."""
    if saved['completed_epochs'] != epochs or len(saved['history']) != epochs:
        raise ValueError('A different checkpoint budget cannot supply a factorial cell')
    metadata = saved['metadata']
    if (metadata['seed'] != 0 or metadata['cue'] != 'ir' or
            metadata['study_sha256'] != sha256(STUDY) or
            metadata['source_RGB_sha256'] != study['source_RGB_sha256']):
        raise ValueError('Different teacher, cue, seed or parent study')
    for epoch, row in enumerate(saved['history'], 1):
        delivery = row['delivery']; expected = schedule_proof(study['Train_delivery']['lengths'], epoch)
        if (not row['frozen_RGB_exact_unchanged'] or not delivery['complete_all_Train_frames'] or
                delivery['actual_address_sha256'] != expected['address_sha256'] or
                delivery['address_sha256'] != expected['address_sha256'] or
                delivery['frames'] != 27251 or delivery['delivered_clips'] != 6814 or
                delivery['repeated_frames'] != 0 or delivery['per_video_frames'] != expected['per_video_frames']):
            raise ValueError('Actual chronological full-Train addresses or frozen RGB proof disagree')
    return metadata


def schedule_sha256(study, stage):
    if stage not in [6, 9, 12]: raise ValueError('Use the parent common executed Train budget')
    rows = [schedule_proof(study['Train_delivery']['lengths'], e) for e in range(1, stage+1)]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def authorized(work):
    work = Path(work); base = work/'fusion_factorial'
    reg = registered(work); value = read(base/AUTHORIZATION)
    assert value['status'] == 'AUTHORIZED_ONE_JOINT_DISABLE_FIT_FROM_COMPLETE_PARENT_DEV_ONLY'
    assert value['science_sha256'] == sha256(SCIENCE)
    assert value['registry_sha256'] == sha256(work/REGISTRY)
    assert value['selection_input_split'] == 'dev' and not value['Test_accuracy_used']
    assert value['stage'] in [6, 9, 12] and 1 <= value['primary_epoch'] <= value['stage']
    assert value['mode'] in ['quality', 'temporal']
    for path, h in value['artifacts'].items(): assert sha256(path) == h
    config = read(value['joint_arm_path'])
    assert config == make_joint_arm(read(STUDY), value['mode'])
    assert sha256(value['joint_arm_path']) == value['joint_arm_sha256']
    return value, config, reg


def require_gpu(work):
    """The missing registration/Dev authorization/release must fail before CUDA."""
    authorization, config, registration = authorized(work)
    if not predecessor_ready(work):
        raise RuntimeError('Wait for all existing queues, exports and actual child process exits')
    require_host_memory(8*2**30, maximum_workers=1)
    require_storage(work, growth_bytes=int(.5*2**30))
    from tools.native_pyramid_experiment import gpu_settings
    physical, uuid = gpu_settings()
    return authorization, config, registration, physical, uuid


def fit_metadata(authorization, arm, registration, provider, initial, physical, uuid, input_reference):
    return dict(fit_id=arm['fit_id'], mode=arm['mode'], cue='ir', removal=arm['removal'], seed=0,
        pipeline_revision=2, factorial_pipeline_revision=1,
        fit_config_sha256=authorization['joint_arm_sha256'], study_sha256=sha256(STUDY),
        factorial_science_sha256=sha256(SCIENCE), source_RGB_sha256=arm['source_RGB_sha256'],
        source_hashes=registration['source_hashes'], protocol_sha256=arm['protocol_sha256'],
        calibration_sha256=arm['calibration_sha256'], Train_GT_sha256=provider.gt_hashes,
        initial_native_sha256=initial, physical_gpu=physical, GPU_uuid=uuid,
        past_memory_stop_gradient=True, original_RGB_encoder_and_head_frozen=True,
        Dev_fit_rows=0, Test_fit_rows=0, input_content_reference=input_reference,
        input_content_guard_revision=1, encoded_image_RAM_limit_GiB=1)


def load_modules(rows, study, device, observer=None):
    """Explicit arm paths support old artifacts without modifying their files."""
    import torch
    from yolox.models.native_pyramid_fusion import NativePyramidFusion
    modules, information = {}, {}
    for row in rows:
        if row.get('baseline'):
            assert row['cue'] == 'rgb' and row['baseline'] == 'fresh_RGB_B1'
            modules[row['id']], information[row['id']] = None, dict(row)
            continue
        assert row['cue'] == 'ir' and row['seed'] == 0 and row['cell'] in CELLS
        assert sha256(row['arm_path']) == row['fit_config_sha256']
        arm = read(row['arm_path']); expected = module_kwargs(study, row['mode'], row['cell'])
        assert arm['module_kwargs'] == expected and arm['fit']['lambda_reliable_RGB'] == study['fit']['lambda_reliable_RGB']
        assert not arm['geometry_fixed'] and sha256(row['checkpoint']) == row['checkpoint_sha256']
        saved = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
        meta = validate_history(saved, row['epoch'], study)
        assert meta['fit_id'] == row['fit_id'] and meta['fit_config_sha256'] == row['fit_config_sha256']
        assert meta['mode'] == row['mode'] and row.get('geometry_fixed', False) == arm['geometry_fixed']
        module = NativePyramidFusion(**expected)
        module.load_state_dict(saved['module'], strict=True)
        modules[row['id']] = module.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if observer is not None: observer.attach(row['id'], modules[row['id']])
        information[row['id']] = dict(row, fit_metadata=meta,
            parameter_inventory=sum(p.numel() for p in module.parameters()))
    return modules, information
