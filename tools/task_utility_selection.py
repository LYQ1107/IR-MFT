"""Original-file Dev decisions and pre-Test freeze; no inference or CUDA release."""
from pathlib import Path

from tools.native_pyramid_experiment import frozen_json,read
from tools.observe_task_utility import verify_observations
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_design import (FRESH,CANONICAL,additional_Dev_models,frozen_test_models,
    metric_row,register_trial,require_test_freeze)
from tools.task_utility_study import STUDY,REGISTRY,registered
from tools.task_utility_study_rules import NORMAL,VARIANTS,INTERVENTIONS,budget_decision,select_models


def put_hash(files,path,expected=None):
    path=str(Path(path));actual=sha256(path)
    if (expected is not None and expected!=actual) or (path in files and files[path]!=actual):
        raise ValueError('Original completed Dev file changed: '+path)
    files[path]=actual


def original_Dev_rows(plan,protocol,base,cases):
    """Reparse official47 and reverify current observation/track files before choice."""
    base=Path(base);registration=registered(base.parent);files={};rows=[]
    put_hash(files,STUDY);put_hash(files,base/REGISTRY)
    if not cases or len({c['id'] for c in cases})!=len(cases):
        raise ValueError('Original distinct complete Dev cases required')
    for case in cases:
        row=metric_row(base,case,protocol)
        if row['split']!='dev':raise ValueError('Test files may not participate in Dev choices')
        config=read(case['config']);data_path=Path(row['source_metrics']);data=read(data_path)
        manifest=verify_observations(config['specification'],case['config'],base,registration,plan,protocol)
        if manifest!=data['manifest']:raise ValueError('Completed official metrics and current causal files differ')
        inference_path=data_path.parent/'inference_manifest.json';before=read(inference_path)
        if before!=dict(status='TASK_UTILITY_COMPLETE_INFERENCE_BEFORE_GT_EVALUATION',**manifest):
            raise ValueError('Original inference-before-GT record changed')
        for path,expected in [(case['config'],case['config_sha256']),
                (data_path,row['source_metrics_sha256']),
                (data_path.parent/'trackeval_output/pedestrian_detailed.csv',row['official_CSV_sha256']),
                (config['specification'],config['specification_sha256'])]:put_hash(files,path,expected)
        put_hash(files,inference_path)
        for source in manifest['sources'].values():put_hash(files,source['path'],source['sha256'])
        for stream in manifest['coverage'].values():put_hash(files,stream['track_path'],stream['track_sha256'])
        model=config['model']
        if not model.get('baseline'):
            put_hash(files,model['checkpoint'],model['checkpoint_sha256'])
            put_hash(files,base/'arms'/(model['fit_id']+'.json'),model['fit_config_sha256'])
        rows.append(row)
    return rows,files


def validate_curves(rows,through):
    if through not in [6,9,12]:raise ValueError('Common complete Train stage required')
    curves=[r for r in rows if r.get('kind') in ['normal','input_intervention']]
    sources=[r for r in rows if r.get('kind') in ['source_RGB_fresh_B1','source_RGB_canonical']]
    if len(curves)+len(sources)!=len(rows):raise ValueError('Gate/geometry interventions cannot enter normal Dev curves')
    expected={(v,c,e,'normal') for v,c in NORMAL for e in range(1,through+1)}
    expected|={(v,'ir',e,i) for v in VARIANTS for e in range(1,through+1) for i in INTERVENTIONS}
    actual=[(r['variant'],r['cue'],r['epoch'],r['intervention']) for r in curves]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Exact complete normal and IR-control epoch curves required')
    if sources and {r['id'] for r in sources}!={FRESH['id'],CANONICAL['id']}:
        raise ValueError('Keep both fresh B1 and historical canonical source controls')
    decision=budget_decision(curves,through)
    return curves,decision,select_models(curves,through)


def stage_path(base,through):return Path(base)/f'complete_Dev_stage_{through:02d}.json'


def load_stage(plan,protocol,base,through):
    path=stage_path(base,through);value=read(path)
    if (value.get('status')!='COMPLETE_TASK_UTILITY_COMMON_STAGE_ORIGINAL_DEV_FILES' or
            value['study_sha256']!=sha256(STUDY) or value['through_epoch']!=through or
            value['pipeline_registration_sha256']!=sha256(Path(base)/REGISTRY) or
            not all(sha256(p)==s for p,s in value['file_hashes'].items())):
        raise ValueError('Original complete common Dev stage changed')
    rows,files=original_Dev_rows(plan,protocol,base,value['cases'])
    curves,decision,selection=validate_curves(rows,through)
    if (rows!=value['rows'] or any(value['file_hashes'].get(p)!=s for p,s in files.items()) or
            value['budget_decision']!=decision or value['selection']!=selection):
        raise ValueError('Stored Dev choices differ from original official files')
    return value


def complete_stage(plan,protocol,base,cases,through):
    rows,files=original_Dev_rows(plan,protocol,base,cases);curves,decision,selection=validate_curves(rows,through)
    earlier=[s for s in [6,9] if s<through]
    for stage in earlier:
        old=load_stage(plan,protocol,base,stage)
        if not old['budget_decision']['extend_all'] or old['budget_decision']['through_epoch']!=stage+3:
            raise ValueError('All extra Train epochs must follow the original common Dev budget decision')
        for p,s in old['file_hashes'].items():put_hash(files,p,s)
        put_hash(files,stage_path(base,stage))
        current={c['id']:c for c in cases}
        if any(current.get(c['id'])!=c for c in old['cases']):raise ValueError('Earlier complete Dev case replaced')
    value=dict(status='COMPLETE_TASK_UTILITY_COMMON_STAGE_ORIGINAL_DEV_FILES',through_epoch=through,
        study_sha256=sha256(STUDY),pipeline_registration_sha256=sha256(Path(base)/REGISTRY),
        cases=cases,rows=rows,budget_decision=decision,selection=selection,file_hashes=files,
        Test_accuracy_used_for_selection=False,formal_GPU_eligibility_established_here=False)
    frozen_json(stage_path(base,through),value);return value


def freeze_Test(plan,protocol,base,through,additional_cases):
    """Freeze all required original Dev before creating any Test observation."""
    base=Path(base);stage=load_stage(plan,protocol,base,through)
    if stage['budget_decision']['extend_all']:raise ValueError('The common complete-Train budget must finish first')
    selection=stage['selection'];models,roles=frozen_test_models(base,selection)
    expected=additional_Dev_models(base,selection)[1:]
    observed=[read(c['config'])['model'] for c in additional_cases]
    if observed!=expected:raise ValueError('Exactly the three primary gate removals and geometry Dev interventions required')
    rows,files=original_Dev_rows(plan,protocol,base,additional_cases)
    for p,s in stage['file_hashes'].items():put_hash(files,p,s)
    put_hash(files,stage_path(base,through))
    complete=dict(status='COMPLETE_TASK_UTILITY_ALL_REQUIRED_DEV_BEFORE_TEST',study_sha256=sha256(STUDY),
        pipeline_registration_sha256=sha256(base/REGISTRY),through_epoch=through,
        normal_complete_stage=str(stage_path(base,through)),normal_complete_stage_sha256=sha256(stage_path(base,through)),
        selection=selection,additional_cases=additional_cases,additional_Dev_rows=rows,Dev_file_hashes=files,
        decision_split='dev',Test_accuracy_used_for_selection=False,goal_met=False,supported_innovations=0)
    frozen_json(base/'complete_Dev.json',complete)
    spec,cases=register_trial(base,'test_frozen','test',models,True)
    if len(cases)!=26:raise ValueError('All twenty-six full Test objects must be retained')
    frozen=dict(status='FROZEN_TASK_UTILITY_ALL_COMPLETE_DEV_BEFORE_TEST',study_sha256=sha256(STUDY),
        pipeline_registration_sha256=sha256(base/REGISTRY),specification_sha256=sha256(spec),
        complete_Dev_sha256=sha256(base/'complete_Dev.json'),Dev_file_hashes=files,
        complete_Dev_before_Test=True,Test_accuracy_used_for_selection=False,roles=roles,cases=cases,
        through_epoch=through,goal_met=False,supported_innovations=0)
    frozen_json(base/'Test_selection_frozen.json',frozen);require_test_freeze(base,spec)
    return spec,cases,frozen
