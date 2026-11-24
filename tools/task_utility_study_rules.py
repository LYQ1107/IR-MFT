"""Pure Dev decisions for the same-capacity task-supervision factorial.

No files, Test-based choices, execution authorization or GPU allocation here.
Real original-file and source validation belongs to a future complete workflow.
"""
from decimal import Decimal
import math

VARIANTS=('no_aux_labels','foreground_labels','utility_labels','foreground_utility_labels')
CUES=('rgb','gray','ir')
NORMAL=tuple((v,c) for v in VARIANTS for c in CUES)
INTERVENTIONS=('unavailable','IR_image_spatial_roll')


def valid_dev(row):
    if (row.get('split')!='dev' or row.get('sequences')!=['BT-005','SN-013'] or
            row.get('frames')!=4834 or type(row.get('seed')) is not int or row['seed']!=0 or
            type(row.get('epoch')) is not int or not 1<=row['epoch']<=12 or
            (row.get('variant'),row.get('cue')) not in NORMAL):
        raise ValueError('Complete fixed-seed two-video Dev for a registered condition required')
    intervention=row.get('intervention')
    if intervention not in ('normal',*INTERVENTIONS) or (intervention!='normal' and row['cue']!='ir'):
        raise ValueError('Only registered normal and actual IR interventions are allowed')
    metrics=row.get('selection_metrics',{})
    if any(k not in metrics or isinstance(metrics[k],bool) or
           not math.isfinite(float(metrics[k])) or not 0<=float(metrics[k])<=100
           for k in ['HOTA','AssA','IDF1']):
        raise ValueError('Finite full-precision official HOTA, AssA and IDF1 required')


def indexed_dev(rows):
    result={}
    for row in rows:
        valid_dev(row)
        key=(row['variant'],row['cue'],row['epoch'],row['intervention'])
        if key in result:raise ValueError('Duplicate or competing Dev observation')
        result[key]=row
    return result


def complete_curves(rows,through):
    if type(through) is not int or not 1<=through<=12:raise ValueError('Positive registered epoch required')
    index=indexed_dev(rows)
    expected={(v,c,e,'normal') for v,c in NORMAL for e in range(1,through+1)}
    if not expected.issubset(index):raise ValueError('All twelve complete normal Dev curves are mandatory')
    return {key:index[key] for key in sorted(expected)}


def complete_interventions(rows,through):
    if type(through) is not int or not 1<=through<=12:raise ValueError('Positive registered epoch required')
    index=indexed_dev(rows)
    expected={(v,'ir',e,i) for v in VARIANTS for e in range(1,through+1) for i in INTERVENTIONS}
    if not expected.issubset(index):raise ValueError('Every IR snapshot needs full missing and actual image-roll Dev')
    return {key:index[key] for key in sorted(expected)}


def best_dev(rows):
    rows=list(rows)
    if not rows:raise ValueError('No complete normal Dev candidate')
    indexed_dev(rows)
    if any(r['intervention']!='normal' for r in rows):raise ValueError('Interventions cannot choose a normal model')
    return max(rows,key=lambda r:(float(r['selection_metrics']['HOTA']),float(r['selection_metrics']['AssA']),
        float(r['selection_metrics']['IDF1']),-r['epoch'],-VARIANTS.index(r['variant']),-CUES.index(r['cue'])))


def own_best(rows,through):
    normal=complete_curves(rows,through)
    return {(v,c):best_dev(r for (variant,cue,_,_),r in normal.items() if (variant,cue)==(v,c)) for v,c in NORMAL}


def budget_decision(rows,stage):
    if stage not in (6,9,12):raise ValueError('Only common complete-Train stages 6,9,12')
    current=own_best(rows,stage);complete_interventions(rows,stage)
    if stage==12:return dict(stage=stage,extend_all=False,through_epoch=12,reason='fixed_maximum_complete_Train_budget')
    previous=own_best(rows,stage-3)
    improved=[]
    for key in NORMAL:
        new=Decimal(str(current[key]['selection_metrics']['HOTA']))
        old=Decimal(str(previous[key]['selection_metrics']['HOTA']))
        if new-old>Decimal('0.2'):improved.append(dict(variant=key[0],cue=key[1],absolute_HOTA_gain=str(new-old)))
    extend=bool(improved)
    return dict(stage=stage,extend_all=extend,through_epoch=stage+3 if extend else stage,
        improved_conditions=improved,reason='complete_Dev_common_budget_rule',learning_rate_horizon_complete_Train_epochs=12)


def select_models(rows,through):
    if through not in (6,9,12):raise ValueError('Normal selection requires an executed complete common stage')
    own=own_best(rows,through);complete_interventions(rows,through)
    primary=best_dev(own[v,'ir'] for v in VARIANTS)
    return dict(through_epoch=through,primary=primary,
        normal_models=[own[key] for key in NORMAL],
        matched_RGB=own[primary['variant'],'rgb'],matched_gray=own[primary['variant'],'gray'],
        strongest_Dev_RGB=best_dev(own[v,'rgb'] for v in VARIANTS),
        strongest_Dev_gray=best_dev(own[v,'gray'] for v in VARIANTS),
        decision_split='dev',all_twelve_formal_models_require_complete_Test=True)


def test_plan(selection):
    """Immutable recipes to freeze before Test, including negative Dev fits."""
    primary=selection['primary'];valid_dev(primary)
    models=selection['normal_models']
    if (len(models)!=12 or {(r['variant'],r['cue']) for r in models}!=set(NORMAL) or
            primary['cue']!='ir' or selection.get('decision_split')!='dev'):
        raise ValueError('Complete original Dev selection required before creating Test recipes')
    own={(r['variant'],r['cue']):r for r in models}
    for row in models:valid_dev(row)
    if any(r['intervention']!='normal' for r in models) or primary!=best_dev(own[v,'ir'] for v in VARIANTS):
        raise ValueError('Primary must agree with the complete normal IR Dev choice')
    if (selection.get('through_epoch') not in (6,9,12) or any(r['epoch']>selection['through_epoch'] for r in models) or
            selection.get('matched_RGB')!=own[primary['variant'],'rgb'] or
            selection.get('matched_gray')!=own[primary['variant'],'gray'] or
            selection.get('strongest_Dev_RGB')!=best_dev(own[v,'rgb'] for v in VARIANTS) or
            selection.get('strongest_Dev_gray')!=best_dev(own[v,'gray'] for v in VARIANTS)):
        raise ValueError('Matched and strongest controls must retain their original Dev choices')
    cases=[]
    def add(identifier,model,kind,intervention='normal',gate_removal=None):
        cases.append(dict(id=identifier,model=model,kind=kind,intervention=intervention,gate_removal=gate_removal,
            sequences=['BT-002','BT-004','MSK-003','PF-002','SN-009','SN-011','SN-014'],
            frames=15985,seed=0,complete_Test_required=True))
    for variant,cue in NORMAL:add('TU_'+variant+'_'+cue,own[variant,cue],'normal')
    for variant in VARIANTS:
        for intervention in INTERVENTIONS:
            add('TU_'+variant+'_ir_'+intervention,own[variant,'ir'],'input_intervention',intervention)
    for removal in ['foreground','utility','foreground_and_utility']:
        add('TU_primary_remove_'+removal,primary,'inference_gate_removal',gate_removal=removal)
    add('TU_primary_original_RGB_geometry',primary,'inference_only_fixed_geometry')
    add('TU_source_RGB_canonical',None,'source_RGB_canonical')
    add('TU_source_RGB_fresh_B1',None,'source_RGB_fresh_B1')
    assert len(cases)==26 and len({c['id'] for c in cases})==26
    return cases
