"""Artificial tensor/file fixtures only; no actual fitted TaskUtility snapshot."""
import copy
import json

import pytest
import torch

from tools.native_pyramid_train_data import schedule_proof
from tools.task_utility_fit import LOSS_FIELDS,COUNT_FIELDS
from tools.task_utility_fusion import TaskUtilityFusion
from tools.task_utility_model_loading import inference_kwargs,load_auxiliary_cpu
from tools.task_utility_study import STUDY,REGISTRY,arm_path,make_arm,computation_key
from tools.run_pseudo_ir_comparison import sha256


def artifact(tmp_path):
    study=json.loads(STUDY.read_text());arm=make_arm(study,'TU_foreground_utility_labels_ir','foreground_utility_labels','ir')
    path=arm_path(tmp_path,arm['fit_id']);path.parent.mkdir();path.write_text(json.dumps(arm))
    registration=dict(study_sha256=sha256(STUDY),normal_arms={arm['fit_id']:sha256(path)},source_hashes={'explicitly_fake':'not_a_production_registry'})
    (tmp_path/REGISTRY).write_text(json.dumps(registration))
    torch.manual_seed(0);module=TaskUtilityFusion(**arm['module_kwargs'])
    delivery=schedule_proof(arm['Train_delivery']['lengths'],1)
    delivery.update(actual_address_sha256=delivery['address_sha256'],complete_all_Train_frames=True,delivered_clips=6814)
    totals={k:0. for k in COUNT_FIELDS};totals.update(available_frames=27251,valid_task_cells=2725100)
    history=[dict(epoch=1,delivery=delivery,frozen_RGB_exact_unchanged=True,delivered_clips_total=6814,
        actual_optimizer_updates=6814,mean_clip_losses={k:0. for k in LOSS_FIELDS},task_epoch_totals=totals)]
    metadata=dict(family='task_utility',seed=0,study_sha256=sha256(STUDY),source_RGB_sha256=study['source_RGB_sha256'],
        fit_id=arm['fit_id'],variant=arm['variant'],cue='ir',task_supervision=arm['task_supervision'],
        computation_key=computation_key(arm),initial_native_sha256=study['architecture']['initial_state_sha256'],
        Dev_fit_rows=0,Test_fit_rows=0,fit_config_sha256=sha256(path),registry_sha256=sha256(tmp_path/REGISTRY),source_hashes=registration['source_hashes'])
    saved=dict(loop_family='task_utility_v1',loss_fields=list(LOSS_FIELDS),completed_epochs=1,history=history,
        metadata=metadata,source_RGB_only_by_reference=True,module=module.state_dict(),artificial_fixture=True)
    checkpoint=tmp_path/'fits'/arm['fit_id']/'epoch_01.pt';checkpoint.parent.mkdir(parents=True);torch.save(saved,checkpoint)
    row=dict(id='fake_not_actual_Dev_choice',fit_id=arm['fit_id'],variant=arm['variant'],cue='ir',mode='quality',seed=0,
        epoch=1,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),fit_config_sha256=sha256(path),
        intervention='normal',geometry_fixed=False,gate_removal=None)
    return study,arm,registration,saved,row


def test_gate_removal_preserves_all_tensors_and_equal_capacity(tmp_path):
    study,arm,registration,saved,row=artifact(tmp_path)
    for removal,flags in [(None,(True,True)),('foreground',(False,True)),('utility',(True,False)),('foreground_and_utility',(False,False))]:
        case=dict(row,gate_removal=removal);module,information=load_auxiliary_cpu(case,study,tmp_path,registration)
        assert (module.use_foreground,module.use_utility)==flags
        assert information['auxiliary_parameters']==553705 and not module.training
        assert all(not p.requires_grad and p.device.type=='cpu' for p in module.parameters())
        assert all(torch.equal(v,saved['module'][k]) for k,v in module.state_dict().items())


def test_gate_input_and_geometry_interventions_cannot_be_silently_combined():
    study=json.loads(STUDY.read_text());arm=make_arm(study,'TU_foreground_utility_labels_ir','foreground_utility_labels','ir')
    assert inference_kwargs(arm,dict(geometry_fixed=True))==arm['module_kwargs']
    for row in [dict(gate_removal='unknown'),dict(gate_removal='utility',intervention='unavailable'),
            dict(gate_removal='foreground',geometry_fixed=True),dict(geometry_fixed=1),dict(geometry_fixed=True,intervention='IR_image_spatial_roll')]:
        with pytest.raises(ValueError):inference_kwargs(arm,row)
    gray=copy.deepcopy(arm);gray['cue']='gray'
    with pytest.raises(ValueError):inference_kwargs(gray,dict(intervention='IR_image_spatial_roll'))


def test_loader_rejects_wrong_seed_cue_path_or_digest(tmp_path):
    study,arm,registration,saved,row=artifact(tmp_path)
    for change in [dict(seed=1),dict(cue='gray'),dict(checkpoint=str(tmp_path/'other.pt')),dict(checkpoint_sha256='wrong')]:
        with pytest.raises(ValueError):load_auxiliary_cpu(dict(row,**change),study,tmp_path,registration)


def test_loader_rejects_wrong_auxiliary_dtype_and_supervision_lineage(tmp_path):
    study,arm,registration,saved,row=artifact(tmp_path);checkpoint=row['checkpoint']
    first=next(k for k,v in saved['module'].items() if v.is_floating_point())
    bad=copy.deepcopy(saved);bad['module'][first]=bad['module'][first].half();torch.save(bad,checkpoint)
    with pytest.raises(ValueError):load_auxiliary_cpu(dict(row,checkpoint_sha256=sha256(checkpoint)),study,tmp_path,registration)
    bad=copy.deepcopy(saved);bad['metadata']['task_supervision']['foreground']=False;torch.save(bad,checkpoint)
    with pytest.raises(ValueError):load_auxiliary_cpu(dict(row,checkpoint_sha256=sha256(checkpoint)),study,tmp_path,registration)
