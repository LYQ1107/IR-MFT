"""Strict auxiliary snapshot loading on CPU; no selection or CUDA authorization."""
import copy
from pathlib import Path

import torch

from tools.fusion_memory_model_loading import load_pinned_torch,strict_cpu_state
from tools.native_pyramid_experiment import read
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_fusion import TaskUtilityFusion
from tools.task_utility_study import STUDY,REGISTRY,arm_path,validate_history


def inference_kwargs(arm,row):
    removal=row.get('gate_removal');intervention=row.get('intervention','normal')
    geometry=row.get('geometry_fixed',False)
    if type(geometry) is not bool:raise ValueError('Explicit boolean geometry intervention required')
    if intervention not in ['normal','unavailable','IR_image_spatial_roll']:
        raise ValueError('Unregistered task-utility image intervention')
    if intervention!='normal' and arm['cue']!='ir':raise ValueError('Only original IR receives image interventions')
    if removal not in [None,'foreground','utility','foreground_and_utility']:
        raise ValueError('Only predeclared task-gate removals are supported')
    if removal and (arm['cue']!='ir' or intervention!='normal' or geometry):
        raise ValueError('Gate removals use the original IR checkpoint and no combined input/geometry intervention')
    if geometry and (type(geometry) is not bool or intervention!='normal' or arm['cue']!='ir'):
        raise ValueError('Fixed original geometry is a separate IR inference intervention')
    kwargs=copy.deepcopy(arm['module_kwargs'])
    if removal in ['foreground','foreground_and_utility']:kwargs['use_foreground']=False
    if removal in ['utility','foreground_and_utility']:kwargs['use_utility']=False
    return kwargs


def load_auxiliary_cpu(row,study,base,registration):
    """Caller must validate its complete original trial/Dev freeze before use."""
    if type(row.get('epoch')) is not int or not 1<=row['epoch']<=12 or type(row.get('seed')) is not int or row['seed']!=0:
        raise ValueError('An original fixed-seed complete-epoch snapshot required')
    base=Path(base);path=arm_path(base,row['fit_id']);arm=read(path)
    if (read(base/REGISTRY)!=registration or registration['study_sha256']!=sha256(STUDY) or
            registration['normal_arms'].get(row['fit_id'])!=sha256(path) or row['fit_config_sha256']!=sha256(path) or
            arm['parent_study_sha256']!=sha256(STUDY) or row['variant']!=arm['variant'] or
            row['cue']!=arm['cue'] or row.get('mode')!='quality'):
        raise ValueError('Original registered arm/source/cue/variant differs')
    expected=base/'fits'/row['fit_id']/f"epoch_{row['epoch']:02d}.pt"
    if Path(row['checkpoint'])!=expected:raise ValueError('Only the original complete-epoch checkpoint path is allowed')
    saved=load_pinned_torch(expected,row['checkpoint_sha256']);metadata=validate_history(saved,row['epoch'],arm)
    if (metadata['fit_config_sha256']!=sha256(path) or metadata['registry_sha256']!=sha256(base/REGISTRY) or
            metadata['source_hashes']!=registration['source_hashes'] or metadata['source_RGB_sha256']!=study['source_RGB_sha256']):
        raise ValueError('Snapshot is not from the complete registered computation')
    kwargs=inference_kwargs(arm,row)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0);module=TaskUtilityFusion(**kwargs)
    strict_cpu_state(module,saved['module'])
    if sum(p.numel() for p in module.parameters())!=553705:raise ValueError('Equal inference capacity must be retained')
    return module,dict(row,fit_metadata=metadata,training_module_kwargs=arm['module_kwargs'],
        inference_module_kwargs=kwargs,auxiliary_parameters=553705,CPU_tensors_only=True,
        scope='Strict snapshot provenance/tensors,not actual Dev qualification,selection,execution release or originality evidence.')
