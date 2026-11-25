"""TaskUtility registration and original-artifact validators; no CUDA launch."""
import os
from pathlib import Path
import sys

from tools.native_pyramid_experiment import frozen_json,read
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_design import snapshot
from tools.task_utility_model_loading import load_auxiliary_cpu
from tools.task_utility_study import ROOT,STUDY,REGISTRY,PIPELINE,arm_path,context,make_arm,registered

CPU_PROOF='task_utility_workflow_CPU_engineering_20261125/complete.json'
RUNTIME_CPU=ROOT/'configs/perceptual_task_utility_runtime_components_20261125.json'


def source_files():
    names=set(read(RUNTIME_CPU)['source_hashes'])|set(PIPELINE)|{
        'tools/task_utility_queue.py','tools/task_utility_pipeline.py','tools/task_utility_controller.py',
        'tools/export_task_utility_runtime.py','tests/test_task_utility_pipeline.py',
        str(RUNTIME_CPU.relative_to(ROOT))}
    return sorted(names)


def current_source_hashes():return {p:sha256(ROOT/p) for p in source_files()}


def task(identifier,script,*args,physical=None):
    value=dict(id=identifier,command=[sys.executable,'-u','tools/'+script,*map(str,args)])
    if physical is not None:
        if physical not in ['4','5']:raise ValueError('Only physical4/5')
        value['physical_gpu']=physical
    return value


def fit_tasks(study,mode,stage=None):
    if mode not in ['probe','fit'] or (mode=='fit' and stage not in [6,9,12]):raise ValueError('Original fit stage/mode required')
    return [task(f['id'],'train_task_utility.py',mode,'--fit-id',f['id'],
        *(['--through-epoch',stage] if stage is not None else []),physical=f['physical_gpu']) for f in study['normal_fits']]


def verify_probe(base,fit_id,study,registration):
    base=Path(base);path=base/'engineering'/fit_id/'proof.json';value=read(path);m=value['metadata'];arm=read(arm_path(base,fit_id))
    if (value['status']!='COMPLETE_TASK_UTILITY_GPU_TRAIN_B1_ENGINEERING' or value['GPU_work'] is not True or
            type(m['seed']) is not int or m['seed']!=0 or m['fit_id']!=fit_id or m['variant']!=arm['variant'] or m['cue']!=arm['cue'] or
            m['study_sha256']!=sha256(STUDY) or m['fit_config_sha256']!=sha256(arm_path(base,fit_id)) or
            m['registry_sha256']!=sha256(base/REGISTRY) or m['source_hashes']!=registration['source_hashes'] or
            m['source_RGB_sha256']!=study['source_RGB_sha256'] or m['protocol_sha256']!=study['protocol_sha256'] or
            m['calibration_sha256']!=study['calibration_sha256'] or
            m['initial_native_sha256']!=study['architecture']['initial_state_sha256'] or m['task_supervision']!=arm['task_supervision'] or
            m['physical_gpu']!=next(f['physical_gpu'] for f in study['normal_fits'] if f['id']==fit_id) or
            value['Dev_rows']!=0 or value['Test_rows']!=0 or value['peak_process_RSS_GiB']>8):
        raise ValueError('Actual original fixed-seed GPU engineering proof required')
    record=value['record'];online=record['inference'];checks=online['checks']
    if (record['actual_updates']!=3 or record['delivered_Train_frames']!=12 or record['complete_Train_epochs']!=0 or
            not record['existing_AdamW_moments_unchanged'] or record['all_missing_training']['optimizer_updated'] or
            online['actual_device']!='cuda:0' or online['raw_precision']!='torch.float16' or online['B1_frames']!=12 or len(checks)!=12 or
            online['GT_read'] or online['future_frame_input'] or not online['frozen_inference_states_exact']):
        raise ValueError('Twelve actual B1frames and three T4updates/skip checks required')
    expected_uuid=read(ROOT/'configs/perceptual_gpu_handoff_20261123.json')['future_gpu_uuids'][m['physical_gpu']]
    if str(m['GPU_uuid']).lower().replace('gpu-','')!=expected_uuid.lower().replace('gpu-',''):
        raise ValueError('Actual GPU UUID differs from the registered physical card')
    for index,row in enumerate(checks,1):
        if (row['frame']!=index or not row['direct_zero_missing_raw_head_and_NMS_exact'] or
                not row['direct_zero_missing_MOT_ID_KF_exact'] or row['GT_read'] or row['future_frame_input'] or
                not row['geometry_packet_matches_independent_coordinate_intervention']):
            raise ValueError('Actual direct/zero/missing/raw/NMS/MOT/geometry evidence required')
        if arm['cue']=='ir' and not row['IR_roll']['actual_input_and_latent_changed']:
            raise ValueError('Actual image-roll and learned latent path required')
    return dict(path=str(path),sha256=sha256(path),initial_state_sha256=m['initial_native_sha256'])


def verify_fit(base,fit_id,stage,study,registration):
    base=Path(base);progress=read(base/'fits'/fit_id/'progress.json')
    if progress['completed_full_Train_epochs']!=stage or progress['registered_through_epoch']!=stage:
        raise ValueError('All registered full-Train epochs must finish')
    row=snapshot(base,fit_id,stage);module,info=load_auxiliary_cpu(row,study,base,registration)
    if info['fit_metadata']!=progress['metadata']:raise ValueError('Actual fitted snapshot and progress lineage differ')
    from tools.fusion_memory_model_loading import load_pinned_torch
    saved=load_pinned_torch(row['checkpoint'],row['checkpoint_sha256'])
    if saved['history']!=progress['history']:raise ValueError('Actual fitted snapshot and progress histories differ')
    if not all((base/'fits'/fit_id/f'epoch_{e:02d}.pt').is_file() for e in range(1,stage+1)):
        raise ValueError('Retain every full-epoch snapshot for complete Dev curves')
    del module,saved
    return dict(progress_sha256=sha256(base/'fits'/fit_id/'progress.json'),checkpoint_sha256=row['checkpoint_sha256'])


def register(proof_path=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Registration is CPU only')
    plan,study,protocol,calibration,work,checkpoint,base=context();proof_path=Path(proof_path or work/CPU_PROOF)
    proof=read(proof_path);sources=current_source_hashes()
    execution_path=proof_path.parent.with_suffix('.execution.json');execution=read(execution_path)
    from tools.neck72_runtime_study import exited
    if (execution['exit_code']!=0 or execution['identity']!=proof['identity'] or
            not exited(execution['pid'],identity=execution['identity']) or
            sha256(execution['log'])!=execution['log_sha256'] or execution.get('RSS_limit_exceeded')):
        raise ValueError('The actual CPU workflow proof must finish and exit0before registration')
    if (proof['status']!='COMPLETE_TASK_UTILITY_CPU_WORKFLOW_GRAPH_NOT_GPU_EXECUTION' or
            proof['source_hashes']!=sources or proof['new_GPU_work'] or proof['formal_GPU_eligibility'] or
            proof['formal_fit_started'] or not proof['all_registered_budget_branches_exercised'] or
            not proof['real_CPU_subprocess_queue_lifecycle_verified']):
        raise ValueError('Complete source-bound CPU graph/lifecycle proof required before registration')
    evidence={str(proof_path):sha256(proof_path),str(execution_path):sha256(execution_path),
        str(execution['log']):execution['log_sha256']}
    for name,status in [
        ('docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_WORKER_CPU_ENGINEERING.json',
         'COMPLETE_ONE_IR_TASK_UTILITY_WORKER_TRAIN_B1_CPU_NOT_GPU_ELIGIBILITY'),
        ('docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_RUNTIME_COMPONENTS.json',
         'COMPLETE_TASK_UTILITY_ACTUAL_TRAIN_B4_CPU_RUNTIME_COMPONENT_NOT_GPU_COST')]:
        path=ROOT/name;value=read(path)
        if value['status']!=status or any(sources[p]!=s for p,s in value['source_hashes'].items()):
            raise ValueError('Original actual CPU component sources changed')
        evidence[str(path)]=sha256(path)
    prototype=read(ROOT/'configs/perceptual_task_utility_prototype_20261125.json');content=prototype['completed_IR_content_manifest']
    if sha256(content['path'])!=content['sha256']:raise ValueError('Original fixed RGB/IR Train byte manifest changed')
    evidence[content['path']]=content['sha256'];arms={}
    for fit in study['normal_fits']:
        path=arm_path(base,fit['id']);frozen_json(path,make_arm(study,fit['id'],fit['variant'],fit['cue']))
        arms[fit['id']]=sha256(path);evidence[str(path)]=sha256(path)
    if arms!=proof['normal_arm_hashes']:raise ValueError('Original twelve equal-capacity CPU arm recipes changed')
    for path,digest in proof['component_artifact_hashes'].items():
        if sha256(path)!=digest:raise ValueError('Original CPU workflow evidence changed')
        evidence[path]=digest
    from tools.neck72_runtime import require_storage
    require_storage(work,growth_bytes=int(1.5*2**30))
    value=dict(status='REGISTERED_COMPLETE_TWELVE_ARM_TASK_UTILITY_PIPELINE_PENDING_GPU_ENGINEERING',
        study_sha256=sha256(STUDY),source_hashes=sources,artifact_hashes=evidence,normal_arms=arms,
        complete_Train_Dev_Test_cost_pipeline=True,GPU_engineering_completed=False,formal_fits_started=False,
        CPU_workflow_graph_only_not_experiment=True,fixed_seed=0,physical_gpus=['4','5'],maximum_workers=2,
        goal_met=False,supported_innovations=0)
    frozen_json(base/REGISTRY,value);registered(work);return value
