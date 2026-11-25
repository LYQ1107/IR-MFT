"""Source-bound task-supervision study and fail-closed formal worker gates.

This component alone is not a complete pipeline registration or CUDA release.
The existing native/spatial/F4 sources and registrations remain unchanged.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

from tools.native_pyramid_experiment import context as parent_context,read
from tools.neck72_runtime_study import exited
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_fit import LOSS_FIELDS,COUNT_FIELDS
from tools.task_utility_study_rules import VARIANTS,CUES,NORMAL

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'configs/perceptual_task_utility_study_20261125.json'
REGISTRY='pipeline_registered.json'
PIPELINE=('tools/train_task_utility.py','tools/observe_task_utility.py',
    'tools/benchmark_task_utility.py','tools/launch_task_utility.py',
    'tools/export_task_utility.py','tools/probe_task_utility_workflow.py')


def context():
    plan,parent,protocol,calibration,work,checkpoint=parent_context()
    study=read(STUDY)
    if sha256(ROOT/'configs/perceptual_native_pyramid_study_20261124.json')!=study['native_parent_study_sha256']:
        raise ValueError('Original native study changed')
    for key in ['main_plan_sha256','protocol_sha256','single_seed_policy_sha256',
                'source_RGB_sha256','calibration_sha256','Train_length_source_sha256']:
        if study[key]!=parent[key]:raise ValueError('Original task context changed: '+key)
    if (study['fixed_seed']!=0 or study['additional_seed_work'] or
            study['Train_delivery']['lengths']!=parent['Train_delivery']['lengths'] or
            study['Train_delivery']['complete_frames_per_epoch']!=27251 or
            study['Train_delivery']['clips_per_epoch']!=6814 or
            study['fit']['horizon_updates']!=12*6814 or
            study['architecture']['quality_parameters']!=553705):
        raise ValueError('Fixed complete Train, seed and equal-capacity budget required')
    for entry in study['component_proofs']:
        if sha256(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Original component proof changed')
        proof=read(ROOT/entry['path'])
        if any(sha256(ROOT/p)!=s for p,s in proof.get('source_hashes',{}).items()):
            raise ValueError('Proven component implementation changed')
    if tuple(v['id'] for v in study['variants'])!=VARIANTS or tuple(study['matched_cues'])!=CUES:
        raise ValueError('The complete four-supervision by three-cue factorial is mandatory')
    expected_supervision=((False,False),(True,False),(False,True),(True,True))
    if tuple((v['supervise_foreground'],v['supervise_utility']) for v in study['variants'])!=expected_supervision:
        raise ValueError('All four distinct auxiliary-supervision combinations are required')
    expected=[(f'TU_{v}_{c}',v,c) for v,c in NORMAL]
    if [(v['id'],v['variant'],v['cue']) for v in study['normal_fits']]!=expected:
        raise ValueError('Registered normal fit identities or order differ')
    return plan,study,protocol,calibration,work,checkpoint,work/'task_utility'


def make_arm(study,fit_id,variant,cue):
    if (variant,cue) not in NORMAL or fit_id!=f'TU_{variant}_{cue}':
        raise ValueError('A registered fixed-seed supervision/cue identity is required')
    variants={v['id']:v for v in study['variants']};definition=variants[variant]
    arm=copy.deepcopy(study)
    arm['task_supervision'].update(foreground=definition['supervise_foreground'],utility=definition['supervise_utility'])
    arm.update(fit_id=fit_id,variant=variant,cue=cue,family='task_utility',mode='quality',removal=None,
        module_kwargs=copy.deepcopy(study['architecture']['module_kwargs']),parent_study_sha256=sha256(STUDY))
    return arm


def computation_key(arm):
    keys=['family','cue','module_kwargs','task_supervision','geometry_fixed','fit','Train_delivery',
        'training_input_hw','inference_input_hw','fixed_seed','source_RGB_sha256','calibration_sha256',
        'protocol_sha256','main_plan_sha256','parent_study_sha256','input_content_reference_sha256']
    value={k:arm[k] for k in keys}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def arm_path(base,fit_id):
    if not fit_id or not fit_id.replace('_','').isalnum():raise ValueError('Safe registered fit ID required')
    return Path(base)/'arms'/(fit_id+'.json')


def registered(work):
    path=Path(work)/'task_utility'/REGISTRY
    value=read(path)
    if (value['status']!='REGISTERED_COMPLETE_TWELVE_ARM_TASK_UTILITY_PIPELINE_PENDING_GPU_ENGINEERING' or
            value['study_sha256']!=sha256(STUDY) or not value['complete_Train_Dev_Test_cost_pipeline'] or
            not set(PIPELINE).issubset(value['source_hashes']) or
            not all((ROOT/p).is_file() for p in PIPELINE)):
        raise ValueError('Complete task-utility pipeline registration required before formal worker CUDA')
    if (not all(sha256(ROOT/p)==s for p,s in value['source_hashes'].items()) or
            not all(sha256(p)==s for p,s in value['artifact_hashes'].items())):
        raise ValueError('Task-utility registration sources or original artifacts changed')
    return value


def release_state_ready(state,identity,report,export,process_exited=exited):
    status=state.get('status','')
    if 'FAIL' in status:raise RuntimeError('Preserve the failed F4 predecessor')
    expected={
        'COMPLETE_F4_STUDY_REQUIRES_GLOBAL_SCIENTIFIC_ASSESSMENT':'COMPLETE_F4_FORMAL_EXPORT_REQUIRES_GLOBAL_SCIENTIFIC_ASSESSMENT',
        'COMPLETE_F4_NOT_ELIGIBLE_NO_F4_EXPERIMENT':'F4_NOT_ELIGIBLE_FROM_ORIGINAL_COMPLETE_DEV_NOT_AN_F4_EXPERIMENT'}
    if status not in expected:return False
    if (report.get('status')!=expected[status] or report.get('study_sha256')!=state.get('study_sha256') or
            state.get('controller_pid')!=identity.get('pid') or state.get('controller_identity')!=identity):
        raise ValueError('F4 terminal report and original controller identity disagree')
    if export.get('exit_code')!=0 or not export.get('identity'):return False
    if not process_exited(identity['pid'],identity=identity):return False
    if not process_exited(export['identity']['pid'],identity=export['identity']):return False
    phases=state.get('phases',{})
    if status=='COMPLETE_F4_NOT_ELIGIBLE_NO_F4_EXPERIMENT':
        if phases:raise ValueError('Ineligible F4 cannot have launched experimental phases')
        return True
    required={'GPU_ENGINEERING','DEV_NORMAL_PRODUCE','DEV_NORMAL_EVALUATE','DEV_ADDITIONAL_PRODUCE',
        'DEV_ADDITIONAL_EVALUATE','TEST_PRODUCE','TEST_EVALUATE','RUNTIME'}
    if set(phases)!=required or report.get('pipeline_registration_sha256')!=state.get('pipeline_registration_sha256'):
        raise ValueError('All original F4 execution and provenance must finish')
    for jobs in phases.values():
        if not jobs:return False
        for child in jobs.values():
            if (child.get('status')!='COMPLETE' or child.get('exit_code')!=0 or
                    not child.get('pid') or not child.get('start_ticks') or not child.get('command_sha256')):
                return False
            if not process_exited(child['pid'],command=child['command'],identity=child):return False
    return True


def predecessor_ready(work):
    from tools.fusion_memory_workflow import FusionMemoryWorkflow
    workflow=FusionMemoryWorkflow();workflow.registered()
    base=Path(work)/'fusion_memory';path=base/'state.json'
    if not path.exists():return False
    state=read(path)
    if 'FAIL' in state.get('status',''):raise RuntimeError('Preserve failed F4 predecessor')
    if not state.get('status','').startswith('COMPLETE_F4_'):return False
    if not workflow.predecessor_ready():return False
    export_path=base/'final_export_execution.json'
    if not export_path.exists():return False
    export=read(export_path);report_path=Path(export['report'])
    if (sha256(report_path)!=export['report_sha256'] or sha256(export['log'])!=export['log_sha256'] or
            state['pipeline_registration_sha256']!=sha256(base/'pipeline_registered.json')):
        raise ValueError('Original completed F4 export files or registration changed')
    return release_state_ready(state,read(base/'controller_identity.json'),read(report_path),export)


def validate_history(saved,epoch,arm):
    from tools.native_pyramid_train_data import schedule_proof
    if (saved.get('loop_family')!='task_utility_v1' or saved.get('loss_fields')!=list(LOSS_FIELDS) or
            saved['completed_epochs']!=epoch or len(saved['history'])!=epoch or
            not saved.get('source_RGB_only_by_reference')):
        raise ValueError('Complete task-supervision snapshot envelope required')
    m=saved['metadata']
    if (m.get('family')!='task_utility' or type(m.get('seed')) is not int or m['seed']!=0 or m.get('study_sha256')!=sha256(STUDY) or
            m.get('source_RGB_sha256')!=arm['source_RGB_sha256'] or m.get('cue')!=arm['cue'] or
            m.get('fit_id')!=arm['fit_id'] or m.get('variant')!=arm['variant'] or
            m.get('task_supervision')!=arm['task_supervision'] or m.get('computation_key')!=computation_key(arm) or
            m.get('initial_native_sha256')!=arm['architecture']['initial_state_sha256'] or
            m.get('Dev_fit_rows')!=0 or m.get('Test_fit_rows')!=0):
        raise ValueError('Task-supervision checkpoint computation or source lineage differs')
    previous_updates=0
    for index,row in enumerate(saved['history'],1):
        expected=schedule_proof(arm['Train_delivery']['lengths'],index);delivery=row['delivery']
        if (row['epoch']!=index or not row['frozen_RGB_exact_unchanged'] or
                not delivery['complete_all_Train_frames'] or delivery['frames']!=27251 or
                delivery['clips']!=6814 or delivery['delivered_clips']!=6814 or delivery['repeated_frames']!=0 or
                delivery['per_video_frames']!=expected['per_video_frames'] or
                delivery['address_sha256']!=expected['address_sha256'] or
                delivery['actual_address_sha256']!=expected['address_sha256'] or
                row['delivered_clips_total']!=index*6814):
            raise ValueError('Every actual complete Train address and frame must match')
        losses=row['mean_clip_losses'];totals=row['task_epoch_totals'];updates=row['actual_optimizer_updates']
        if (set(losses)!=set(LOSS_FIELDS) or set(totals)!=set(COUNT_FIELDS) or
                any(not math.isfinite(v) or v<0 for v in (*losses.values(),*totals.values())) or
                totals['available_frames']+totals['missing_frames']!=27251 or
                not previous_updates<=updates<=previous_updates+6814):
            raise ValueError('All task losses, delivery counts and actual update counts required')
        previous_updates=updates
    return m
