"""Dev-only authorization, complete curves and frozen artifact reuse."""
from pathlib import Path

from tools.evaluate_fusion_factorial import FRESH
from tools.fusion_factorial_rules import CELLS,module_kwargs,select_complete_dev
from tools.fusion_factorial_workflow import (
    REGISTRY,SCIENCE,STUDY,context,make_joint_arm,read,schedule_sha256,validate_history,
)
from tools.native_pyramid_experiment import FIT_CONFIGS,frozen_json
from tools.native_pyramid_study_rules import select_models
from tools.run_pseudo_ir_comparison import sha256


def identity(row):
    """IDs/prose never determine scientific reuse; all model fields do."""
    return (row['fit_id'],row['checkpoint_sha256'],row['fit_config_sha256'],row['epoch'],
            row['mode'],row['cue'],row.get('seed',0),row.get('geometry_fixed',False),
            row.get('intervention','normal'))


def model_row(cell,epoch,fit,mode):
    checkpoint=Path(fit['fit_directory'])/f'epoch_{epoch:02d}.pt'
    return dict(id=f'FQ_{cell}_e{epoch:02d}',cell=cell,epoch=epoch,mode=mode,cue='ir',seed=0,
        fit_id=fit['fit_id'],arm_path=fit['arm_path'],fit_config_sha256=fit['fit_config_sha256'],
        checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        geometry_fixed=False,intervention='normal')


def prepare_authorization(registration):
    """Reads complete parent Dev only. No parent Test report is an input."""
    import torch
    plan,study,protocol,calibration,work,checkpoint,base=context()
    normal_path=work/'native_pyramid_normal_selection_frozen.json'
    dev_path=work/'native_pyramid_complete_Dev.json'
    normal,dev=read(normal_path),read(dev_path)
    assert normal['study_sha256']==dev['study_sha256']==sha256(STUDY)
    stage=normal['complete_epochs_each'];assert stage==dev['executed_complete_epochs_each']
    values=[r for r in dev['rows'] if not r.get('baseline') and not r.get('removal') and r.get('intervention','normal')=='normal']
    fresh=next(r for r in dev['rows'] if r['id']=='NP_source_fresh_RGB')
    for row in values+[fresh]:
        value=read(row['source_metrics'])
        assert sha256(row['source_metrics'])==row['source_metrics_sha256']
        assert value['status']=='COMPLETE' and value['trackeval']==row['metrics']
        assert value['manifest']['split']=='dev' and value['manifest']['frames']==4834
        assert value['manifest']['sequences']==protocol['dev']
    selected=select_models(values,stage,max(plan['reference']['rgb_dev_HOTA'],fresh['metrics']['HOTA']))
    assert selected==normal['selection']==dev['selection']
    if not selected['fit_registered_removals']:
        return dict(status='FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_COMPLETE_PARENT_DEV',
            selection=selected,stage=stage,parent_normal_selection_sha256=sha256(normal_path),
            parent_complete_Dev_sha256=sha256(dev_path),Test_accuracy_used=False,
            new_formal_fits=0,goal_met=False,supported_innovations=0)
    mode=selected['primary']['mode'];fits={};initials=set()
    artifacts={str(normal_path):sha256(normal_path),str(dev_path):sha256(dev_path)}
    for cell,fit_id in [('11',mode+'_ir'),('10',mode+'_no_fusion_quality_gate'),('01',mode+'_no_spectral_contrast')]:
        arm_path=FIT_CONFIGS/(fit_id+'.json');arm=read(arm_path)
        assert arm['module_kwargs']==module_kwargs(study,mode,cell)
        assert arm['fit']==study['fit'] and not arm['geometry_fixed']
        folder=work/'native_pyramid_fits'/fit_id;path=folder/f'epoch_{stage:02d}.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        metadata=validate_history(saved,stage,study)
        assert metadata['fit_config_sha256']==sha256(arm_path) and metadata['fit_id']==fit_id
        initials.add(metadata['initial_native_sha256'])
        fits[cell]=dict(fit_id=fit_id,arm_path=str(arm_path),fit_config_sha256=sha256(arm_path),
                        fit_directory=str(folder),initial_native_sha256=metadata['initial_native_sha256'])
        artifacts[str(arm_path)]=sha256(arm_path);artifacts[str(path)]=sha256(path)
        del saved
    assert len(initials)==1
    cpu_path=work/'fusion_factorial_training_worker_CPU_proof.json';cpu=read(cpu_path)
    assert cpu['status']=='COMPLETE_CPU_FACTORIAL_TRAIN_WORKER_AND_B1_SOURCE_CANARY'
    assert all(sha256(Path(__file__).resolve().parents[1]/n)==h for n,h in cpu['source_hashes'].items())
    assert next(r for r in cpu['records'] if r['mode']==mode)['initial_native_sha256']==next(iter(initials))
    artifacts[str(cpu_path)]=sha256(cpu_path)
    arm=make_joint_arm(study,mode);arm_path=base/'joint_arm.json';frozen_json(arm_path,arm)
    fits['00']=dict(fit_id=arm['fit_id'],arm_path=str(arm_path),fit_config_sha256=sha256(arm_path),
        fit_directory=str(base/'fit'),initial_native_sha256=next(iter(initials)))
    value=dict(status='AUTHORIZED_ONE_JOINT_DISABLE_FIT_FROM_COMPLETE_PARENT_DEV_ONLY',
        science_sha256=sha256(SCIENCE),registry_sha256=sha256(work/REGISTRY),
        selection_input_split='dev',Test_accuracy_used=False,mode=mode,stage=stage,
        primary_epoch=selected['primary']['epoch'],initial_native_sha256=next(iter(initials)),
        joint_arm_path=str(arm_path),joint_arm_sha256=sha256(arm_path),cell_fits=fits,
        Train_schedule_sha256=schedule_sha256(study,stage),artifacts=artifacts,
        parent_complete_Dev=str(dev_path),parent_normal_selection=str(normal_path))
    frozen_json(base/'authorization.json',value)
    return value


def complete_dev_and_freeze(authorization):
    """Use original39headline selection; retain full precision for analysis."""
    import torch
    plan,study,protocol,calibration,work,checkpoint,base=context()
    parent=read(authorization['parent_complete_Dev']);stage=authorization['stage']
    fits=authorization['cell_fits'];rows=[]
    saved=torch.load(base/'fit'/f'epoch_{stage:02d}.pt',map_location='cpu',weights_only=False)
    meta=validate_history(saved,stage,study)
    assert meta['initial_native_sha256']==authorization['initial_native_sha256'];del saved
    for cell in CELLS:
        for epoch in range(1,stage+1):
            model=model_row(cell,epoch,fits[cell],authorization['mode'])
            if cell=='00':path=base/'evaluation/dev'/model['id']/'metrics.json'
            else:
                matches=[r for r in parent['rows'] if r.get('fit_id')==model['fit_id'] and r.get('epoch')==epoch and r.get('intervention','normal')=='normal']
                assert len(matches)==1;path=Path(matches[0]['source_metrics'])
            value=read(path);manifest=value['manifest'];assert value['status']=='COMPLETE'
            assert manifest['split']=='dev' and manifest['frames']==4834 and manifest['sequences']==protocol['dev']
            source_model=manifest['config'].get('model',manifest['config'])
            assert identity(source_model)==identity(model)
            rows.append(dict(model,model=model,split='dev',sequences=protocol['dev'],frames=4834,
                executed_fit_budget=stage,complete_Train_frames_per_epoch=27251,
                source_RGB_sha256=study['source_RGB_sha256'],initial_native_sha256=authorization['initial_native_sha256'],
                Train_schedule_sha256=authorization['Train_schedule_sha256'],metrics=value['trackeval'],
                source_metrics=str(path),source_metrics_sha256=sha256(path)))
    selection=select_complete_dev(rows,stage,authorization['primary_epoch'],authorization['mode'])
    selected={};roles={}
    for policy in ['paired_parent_primary_epoch','own_Dev_best']:
        roles[policy]={}
        for cell,row in selection[policy].items():
            selected[row['model']['id']]=row['model'];roles[policy][cell]=row['model']['id']
    freeze=dict(status='FROZEN_FOUR_FACTORIAL_CELLS_FROM_COMPLETE_DEV',science_sha256=sha256(SCIENCE),
        authorization_sha256=sha256(base/'authorization.json'),complete_all_four_Dev=True,
        Test_accuracy_used_for_selection=False,stage=stage,mode=authorization['mode'],
        original_primary_epoch=authorization['primary_epoch'],Dev_rows=rows,selection=selection,
        roles=roles,selected_unique_models=[selected[k] for k in sorted(selected)],
        selection_precision='Unchanged parent39headline HOTA/AssA/IDF1 tie rule. Original CSV full precision used only for effect analysis.',
        historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0)
    frozen_json(base/'Test_selection_frozen.json',freeze)
    return freeze


def plan_test_reuse(freeze):
    """Run after our Dev freeze; match artifact identity, never Test scores."""
    plan,study,protocol,calibration,work,checkpoint,base=context()
    assert read(base/'Test_selection_frozen.json')==freeze
    parent_path=work/'native_pyramid_Test_selection_frozen.json';parent=read(parent_path)
    assert parent['complete_Dev_before_Test'] and parent['study_sha256']==sha256(STUDY)
    indexed={identity(r):r for r in parent['models'] if not r.get('baseline')}
    sources={FRESH['id']:dict(metrics_json=str(work/'native_pyramid_evaluation/test/NP_source_fresh_RGB/metrics.json'),source_case_id='NP_source_fresh_RGB',reused_parent=True)}
    missing=[]
    for row in freeze['selected_unique_models']:
        original=indexed.get(identity(row))
        if original is None:
            missing.append(row)
            sources[row['id']]=dict(metrics_json=str(base/'evaluation/test'/row['id']/'metrics.json'),source_case_id=row['id'],reused_parent=False)
        else:
            sources[row['id']]=dict(metrics_json=str(work/'native_pyramid_evaluation/test'/original['id']/'metrics.json'),source_case_id=original['id'],reused_parent=True)
    assert any(r['cell']=='00' for r in missing)
    value=dict(status='FROZEN_TEST_REUSE_AND_MISSING_INFERENCE_PLAN',
        Test_selection_sha256=sha256(base/'Test_selection_frozen.json'),
        parent_Test_selection_sha256=sha256(parent_path),sources=sources,missing_models=missing,
        artifact_identity_fields='fit,checkpoint SHA,arm SHA,epoch,mode,cue,seed,geometry,intervention',
        Test_scores_used_for_reuse=False)
    frozen_json(base/'Test_coverage_plan.json',value)
    return value
