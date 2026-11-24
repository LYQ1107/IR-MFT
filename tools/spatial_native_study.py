"""Independent spatial-study context, provenance and release gates.

No mutation of the live native-pyramid, structural, neck or factorial sources.
Formal CUDA requires the complete controller registration, not this file alone.
"""
import copy
import hashlib
import json
from pathlib import Path

from tools.native_pyramid_experiment import context as parent_context, frozen_json, read
from tools.neck72_runtime_study import exited
from tools.run_pseudo_ir_comparison import sha256

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'configs/perceptual_native_spatial_study_20261124.json'
REGISTRY = 'pipeline_registered.json'
PIPELINE = ['tools/train_spatial_native.py', 'tools/observe_spatial_native.py',
    'tools/benchmark_spatial_native.py', 'tools/launch_spatial_native.py',
    'tools/export_spatial_native.py', 'tools/probe_spatial_native_workflow.py']


def context():
    plan, parent, protocol, calibration, work, checkpoint = parent_context()
    study = read(STUDY)
    for field in ['main_plan_sha256', 'protocol_sha256', 'single_seed_policy_sha256',
                  'source_RGB_sha256', 'calibration_sha256', 'Train_length_source_sha256']:
        assert study[field] == parent[field]
    assert study['fixed_seed'] == 0 and not study['additional_seed_work']
    assert sha256(ROOT/'configs/perceptual_native_spatial_prototype_20261124.json') == study['prototype_sha256']
    assert sha256(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_SPATIAL_CPU_PROOF.json') == study['prototype_CPU_proof_sha256']
    assert study['Train_delivery']['lengths'] == parent['Train_delivery']['lengths']
    assert study['fit']['horizon_updates'] == 12*study['Train_delivery']['clips_per_epoch']
    assert sum(study['Train_delivery']['lengths'].values()) == 27251
    return plan, study, protocol, calibration, work, checkpoint, work/'native_spatial'


def registered(work):
    value = read(Path(work)/'native_spatial'/REGISTRY)
    assert value['status'] == 'REGISTERED_COMPLETE_NINE_ARM_SPATIAL_PIPELINE_PENDING_GPU_ENGINEERING'
    assert value['study_sha256'] == sha256(STUDY) and value['complete_Train_Dev_Test_cost_pipeline']
    assert set(PIPELINE).issubset(value['source_hashes'])
    assert all(sha256(ROOT/name) == digest for name, digest in value['source_hashes'].items())
    assert all(sha256(name) == digest for name, digest in value['artifact_hashes'].items())
    return value


def release_state_ready(state, identity, report, process_exited=exited):
    status = state.get('status', '')
    if 'FAIL' in status or 'MISMATCH' in status:
        raise RuntimeError('Preserve and resolve the predecessor failure before new spatial CUDA work')
    expected = {
        'COMPLETE_FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_DEV': 'COMPLETE_CONDITIONAL_FACTORIAL_NOT_TRIGGERED',
        'COMPLETE_FACTORIAL_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT': 'COMPLETE_FOUR_CELL_FACTORIAL_REQUIRES_SCIENTIFIC_ASSESSMENT',
    }
    if status not in expected or report.get('status') != expected[status]:
        return False
    if (not state.get('science_sha256') or not state.get('registry_sha256')
            or report.get('science_sha256') != state['science_sha256']
            or report.get('registry_sha256') != state['registry_sha256']):
        raise ValueError('Predecessor export is not from its registered scientific computation')
    if state.get('controller_pid') != identity.get('pid'):
        raise ValueError('Predecessor process identity disagrees')
    if not process_exited(identity['pid'], identity=identity):
        return False
    if status == 'COMPLETE_FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_DEV' and state.get('phases'):
        raise ValueError('An untriggered extension cannot have launched producers')
    if status != 'COMPLETE_FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_DEV' and not state.get('phases'):
        return False
    for entries in state.get('phases', {}).values():
        if not entries:
            return False
        for child in entries.values():
            if (child.get('status') != 'COMPLETE' or child.get('exit_code') != 0
                    or not child.get('start_ticks') or not child.get('command')):
                return False
            if not process_exited(child['pid'], command=child['command'], identity=child):
                return False
    return True


def predecessor_ready(work):
    base = Path(work)/'fusion_factorial'
    path = base/'state.json'
    if not path.exists():
        return False
    state = read(path)
    if 'FAIL' in state.get('status', ''):
        raise RuntimeError('Preserve the failed factorial predecessor')
    report = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_FACTORIAL_FORMAL.json'
    if not report.exists():
        return False
    assert state['science_sha256'] == sha256(ROOT/'configs/perceptual_native_contrast_quality_factorial_20261124.json')
    identity = read(base/'controller_identity.json')
    return release_state_ready(state, identity, read(report))


def make_arm(study, fit_id, variant, cue, removal=None):
    variants = {v['name']: v for v in study['variants']}
    if variant not in variants or cue not in study['matched_cues'] or (removal and cue != 'ir'):
        raise ValueError('A registered spatial variant and cue are required')
    config = copy.deepcopy(study)
    kwargs = {k: variants[variant][k] for k in ['use_quality', 'use_spatial', 'use_redundancy']}
    if removal:
        flag = {'no_spatial': 'use_spatial', 'no_frame_quality': 'use_quality',
                'no_redundancy': 'use_redundancy'}.get(removal)
        if flag is None or not kwargs[flag]:
            raise ValueError('Only an enabled registered mechanism can be removed')
        kwargs[flag] = False
    kwargs.update(channels=(320, 640, 1280), hidden=32, spectral=True, protect_rgb=True, bound=.5)
    # JSON canonicalization keeps future loaded arms identical to this recipe.
    kwargs = json.loads(json.dumps(kwargs))
    config.update(fit_id=fit_id, variant=variant, cue=cue, removal=removal, mode='quality',
        module_kwargs=kwargs, parent_study_sha256=sha256(STUDY))
    return config


def computation_key(arm):
    fields = ['cue', 'module_kwargs', 'geometry_fixed', 'fit', 'Train_delivery',
        'training_input_hw', 'inference_input_hw', 'fixed_seed', 'source_RGB_sha256',
        'calibration_sha256', 'protocol_sha256', 'main_plan_sha256', 'parent_study_sha256']
    value = {k: arm[k] for k in fields}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def arm_path(base, fit_id):
    if not fit_id or not fit_id.replace('_', '').isalnum():
        raise ValueError('Safe registered fit ID required')
    return Path(base)/'arms'/(fit_id+'.json')


def validate_history(saved, epoch, arm):
    from tools.native_pyramid_train_data import schedule_proof
    if saved['completed_epochs'] != epoch or len(saved['history']) != epoch:
        raise ValueError('Wrong complete-Train checkpoint budget')
    metadata = saved['metadata']
    if (metadata['seed'] != 0 or metadata['study_sha256'] != sha256(STUDY)
            or metadata['source_RGB_sha256'] != arm['source_RGB_sha256']
            or metadata['cue'] != arm['cue'] or metadata['fit_id'] != arm['fit_id']
            or metadata['computation_key'] != computation_key(arm)):
        raise ValueError('Checkpoint computation or source lineage disagrees')
    for index, row in enumerate(saved['history'], 1):
        delivery = row['delivery']
        expected = schedule_proof(arm['Train_delivery']['lengths'], index)
        if (row['epoch'] != index or not row['frozen_RGB_exact_unchanged']
                or not delivery['complete_all_Train_frames'] or delivery['frames'] != 27251
                or delivery['clips'] != 6814 or delivery['delivered_clips'] != 6814
                or delivery['repeated_frames'] != 0 or delivery['per_video_frames'] != expected['per_video_frames']
                or delivery['actual_address_sha256'] != expected['address_sha256']
                or delivery['address_sha256'] != expected['address_sha256']):
            raise ValueError('Every source frame and chronological clip must match the actual full-Train schedule')
    return metadata


def bounded_image_reader(work, plan, protocol, lengths, cue):
    from tools.native_pyramid_content_guard import make_image_reader
    reader, reference = make_image_reader(work, plan, protocol, lengths, cue)
    reader.max_bytes = 2*2**30
    assert not reader.cache and reader.bytes == 0
    return reader, dict(reference, maximum_encoded_image_RAM_bytes=reader.max_bytes)
