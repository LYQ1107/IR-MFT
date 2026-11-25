"""Task-supervision recipes and original-file bindings; no CUDA or selection from Test."""
from pathlib import Path

from tools.native_pyramid_experiment import frozen_json,read
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_model_loading import inference_kwargs
from tools.task_utility_study import STUDY,REGISTRY,arm_path
from tools.task_utility_study_rules import NORMAL,test_plan

FRESH=dict(id='TU_source_RGB_fresh_B1',cue='rgb',seed=0,baseline='fresh_RGB_B1',kind='source_RGB_fresh_B1')
CANONICAL=dict(id='TU_source_RGB_canonical',cue='rgb',seed=0,baseline='historical_canonical_RGB_B4',kind='source_RGB_canonical')


def snapshot(base,fit_id,epoch,intervention='normal',gate_removal=None,geometry_fixed=False,identifier=None):
    if type(epoch) is not int or not 1<=epoch<=12:raise ValueError('A complete registered Train epoch required')
    config=arm_path(base,fit_id);arm=read(config)
    if (arm['variant'],arm['cue']) not in NORMAL or fit_id!=f"TU_{arm['variant']}_{arm['cue']}":
        raise ValueError('Original normal fit identity required')
    fields=dict(intervention=intervention,gate_removal=gate_removal,geometry_fixed=geometry_fixed)
    inference_kwargs(arm,fields)
    kind='normal'
    if intervention!='normal':kind='input_intervention'
    if gate_removal is not None:kind='inference_gate_removal'
    if geometry_fixed:kind='inference_only_fixed_geometry'
    suffix={'normal':'','unavailable':'_off','IR_image_spatial_roll':'_roll'}[intervention]
    if gate_removal:suffix+='_remove_'+gate_removal
    if geometry_fixed:suffix+='_original_geometry'
    identifier=identifier or f'{fit_id}_e{epoch:02d}'+suffix
    if not identifier.replace('_','').isalnum():raise ValueError('Safe distinct observation ID required')
    checkpoint=Path(base)/'fits'/fit_id/f'epoch_{epoch:02d}.pt'
    return dict(id=identifier,fit_id=fit_id,epoch=epoch,variant=arm['variant'],mode='quality',cue=arm['cue'],seed=0,
        kind=kind,**fields,fit_config_sha256=sha256(config),checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint))


def stage_models(study,base,first,last):
    if (first,last) not in [(1,6),(7,9),(10,12)]:raise ValueError('Complete common Dev stage spans only')
    if {(f['variant'],f['cue']) for f in study['normal_fits']}!=set(NORMAL) or len(study['normal_fits'])!=12:
        raise ValueError('All twelve normal fits are mandatory')
    models=[dict(FRESH)]
    for fit in study['normal_fits']:
        for epoch in range(first,last+1):
            models.append(snapshot(base,fit['id'],epoch))
            if fit['cue']=='ir':
                models.extend(snapshot(base,fit['id'],epoch,i) for i in ['unavailable','IR_image_spatial_roll'])
    if len(models)!=1+20*(last-first+1) or len({m['id'] for m in models})!=len(models):
        raise ValueError('Incomplete or repeated stage observations')
    return models


def frozen_test_models(base,selection):
    recipes=test_plan(selection);models=[dict(FRESH)]
    for case in recipes:
        if case['model'] is None:continue
        selected=case['model']
        models.append(snapshot(base,selected['fit_id'],selected['epoch'],case['intervention'],
            case['gate_removal'],case['kind']=='inference_only_fixed_geometry',case['id']))
    if len(models)!=25 or len({m['id'] for m in models})!=25:raise ValueError('All twenty-five online streams required')
    roles=dict(primary_IR=selection['primary']['fit_id'],matched_RGB=selection['matched_RGB']['fit_id'],
        matched_gray=selection['matched_gray']['fit_id'],strongest_Dev_RGB=selection['strongest_Dev_RGB']['fit_id'],
        strongest_Dev_gray=selection['strongest_Dev_gray']['fit_id'],source_RGB_fresh=FRESH['id'],source_RGB_canonical=CANONICAL['id'])
    return models,roles


def additional_Dev_models(base,selection):
    models,_=frozen_test_models(base,selection)
    return [dict(FRESH)]+[m for m in models if m.get('kind') in ['inference_gate_removal','inference_only_fixed_geometry']]


def validate_trial(value,base,registration):
    if (value.get('status')!='FROZEN_TASK_UTILITY_OBSERVATIONS_BEFORE_INFERENCE' or
            value.get('study_sha256')!=sha256(STUDY) or value.get('split') not in ['dev','test'] or
            value.get('pipeline_registration_sha256')!=sha256(Path(base)/REGISTRY) or
            read(Path(base)/REGISTRY)!=registration):
        raise ValueError('Original study and complete registration must bind every trial')
    phase=value['phase']
    if not phase or not phase.replace('_','').isalnum() or Path(value['stream_directory'])!=Path(base)/'streams'/phase:
        raise ValueError('Original safe observation destination required')
    models=value['models']
    if not models or models[0]!=FRESH or len({m['id'] for m in models})!=len(models):
        raise ValueError('Every unique stream and a fresh RGB reference required')
    if type(value.get('include_sources')) is not bool or any(not m['id'] or not m['id'].replace('_','').isalnum() for m in models):
        raise ValueError('Safe stream identities and explicit source scope required')
    for model in models:
        if model.get('baseline'):
            if model!=FRESH:raise ValueError('Canonical source is a separate CPU replay')
            continue
        config=arm_path(base,model['fit_id']);arm=read(config)
        if (type(model.get('seed')) is not int or model['seed']!=0 or type(model.get('epoch')) is not int or
                not 1<=model['epoch']<=12 or model.get('mode')!='quality' or
                model['variant']!=arm['variant'] or model['cue']!=arm['cue'] or
                model['fit_config_sha256']!=sha256(config) or
                registration['normal_arms'].get(model['fit_id'])!=sha256(config)):
            raise ValueError('Original fixed-seed supervision/cue/arm snapshot required')
        expected=Path(base)/'fits'/model['fit_id']/f"epoch_{model['epoch']:02d}.pt"
        if Path(model['checkpoint'])!=expected or sha256(expected)!=model['checkpoint_sha256']:
            raise ValueError('Original complete-epoch snapshot path/bytes changed')
        inference_kwargs(arm,model)
        kind=('inference_gate_removal' if model.get('gate_removal') else
              'inference_only_fixed_geometry' if model.get('geometry_fixed') else
              'input_intervention' if model.get('intervention')!='normal' else 'normal')
        if model.get('kind')!=kind:raise ValueError('Inference intervention kind differs from its actual flags')
    return value


def register_trial(base,phase,split,models,include_sources):
    if split not in ['dev','test'] or type(include_sources) is not bool:raise ValueError('Declared complete split/source scope required')
    registration=read(Path(base)/REGISTRY);path=Path(base)/'specifications'/(phase+'.json')
    value=dict(status='FROZEN_TASK_UTILITY_OBSERVATIONS_BEFORE_INFERENCE',phase=phase,split=split,models=models,
        study_sha256=sha256(STUDY),pipeline_registration_sha256=sha256(Path(base)/REGISTRY),
        stream_directory=str(Path(base)/'streams'/phase),include_sources=include_sources)
    validate_trial(value,base,registration);frozen_json(path,value);cases=[]
    for model in models+([dict(CANONICAL)] if include_sources else []):
        if model==FRESH and not include_sources:continue
        case=dict(id=model['id'],model=model,split=split,study_sha256=sha256(STUDY),canonical=model==CANONICAL,
            pipeline_registration_sha256=value['pipeline_registration_sha256'],specification=str(path),
            specification_sha256=sha256(path),stream_directory=value['stream_directory'])
        config=Path(base)/'cases'/split/(model['id']+'.json');frozen_json(config,case)
        cases.append(dict(id=model['id'],config=str(config),config_sha256=sha256(config)))
    return path,cases


def require_test_freeze(base,specification_path):
    frozen=read(Path(base)/'Test_selection_frozen.json')
    if (frozen.get('status')!='FROZEN_TASK_UTILITY_ALL_COMPLETE_DEV_BEFORE_TEST' or
            frozen.get('study_sha256')!=sha256(STUDY) or not frozen.get('complete_Dev_before_Test') or
            frozen.get('Test_accuracy_used_for_selection') is not False or
            frozen.get('specification_sha256')!=sha256(specification_path) or
            frozen.get('pipeline_registration_sha256')!=sha256(Path(base)/REGISTRY) or
            frozen.get('complete_Dev_sha256')!=sha256(Path(base)/'complete_Dev.json') or
            not frozen.get('Dev_file_hashes')):
        raise ValueError('All complete original Dev and exact Test recipes must be frozen first')
    if not all(sha256(p)==s for p,s in frozen['Dev_file_hashes'].items()):
        raise ValueError('Original frozen Dev files changed')
    return frozen


def metric_row(base,case,protocol):
    from tools.fusion_memory_evaluation import official_csv47
    data_path=Path(base)/'evaluation'/read(case['config'])['split']/case['id']/'metrics.json';data=read(data_path)
    config=read(case['config']);manifest=data['manifest'];split=config['split']
    if (sha256(case['config'])!=case['config_sha256'] or data['status']!='COMPLETE_TASK_UTILITY_OFFICIAL_METRICS' or
            manifest['config_sha256']!=case['config_sha256'] or manifest['config']!=config or
            manifest['study_sha256']!=sha256(STUDY) or manifest['split']!=split or
            manifest['frames']!={'dev':4834,'test':15985}[split] or
            manifest['sequences']!=protocol['dev' if split=='dev' else 'historical_test']):
        raise ValueError('Complete original official metric/config files required')
    csv_path=data_path.parent/'trackeval_output/pedestrian_detailed.csv'
    parsed=official_csv47(csv_path,data['official_schema'],split,data['metrics39'])
    if any(parsed[k]!=data[k] for k in ['combined47','per_sequence47','selection_metrics','official_CSV_sha256']):
        raise ValueError('Original official CSV and stored47 fields differ')
    return dict(config['model'],split=split,sequences=manifest['sequences'],frames=manifest['frames'],
        metrics39=data['metrics39'],selection_metrics=data['selection_metrics'],source_metrics=str(data_path),
        source_metrics_sha256=sha256(data_path),official_CSV_sha256=sha256(csv_path),
        config=str(case['config']),config_sha256=case['config_sha256'])
