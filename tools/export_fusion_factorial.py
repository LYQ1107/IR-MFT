#!/usr/bin/env python3
"""Publish every factorial curve/cell, original CSV and all47 official fields."""
import csv
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.evaluate_fusion_factorial import FRESH
from tools.export_perceptual_metrics47 import official_schema,read_table
from tools.fusion_factorial_rules import necessary_numeric_evidence
from tools.fusion_factorial_selection import identity
from tools.fusion_factorial_workflow import REGISTRY,SCIENCE,context,read,registered
from tools.native_pyramid_experiment import frozen_json
from tools.run_pseudo_ir_comparison import sha256

DOCS=ROOT/'docs/fusion_research'


def small_json(path,value):
    assert len((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())<2**20
    frozen_json(path,value)


def publish_case(identifier,split,source_path,source_id,condition,plan,protocol,schema):
    source_path=Path(source_path);value=read(source_path);manifest=value['manifest']
    sequences=protocol['dev' if split=='dev' else 'historical_test']
    assert value['status']=='COMPLETE' and len(value['trackeval'])==39
    assert manifest['split']==split and manifest['sequences']==sequences
    assert manifest['frames']=={'dev':4834,'test':15985}[split]
    source_model=manifest['config'].get('model',manifest['config'])
    if condition is not None and not condition.get('baseline'):
        assert identity(source_model)==identity(condition)
        assert sha256(condition['checkpoint'])==condition['checkpoint_sha256']
    csv_path=source_path.parent/'trackeval_output/pedestrian_detailed.csv'
    parent_public=DOCS/'perceptual_20261122_official_tables/native_pyramid'/split/(source_id+'.csv')
    destination=parent_public if parent_public.exists() else DOCS/'perceptual_20261122_official_tables/native_factorial'/split/(identifier+'.csv')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():assert sha256(destination)==sha256(csv_path)
    else:shutil.copy2(csv_path,destination)
    assert destination.stat().st_size<2**20
    metric_rows,provenance=read_table(destination,schema,protocol)
    assert all(row['split']==split for row in metric_rows)
    combined=next(r for r in metric_rows if r['seq']=='COMBINED')
    assert abs(combined['HOTA']-value['trackeval']['HOTA'])<=.000501
    gates={}
    for seq in sequences:
        source=manifest['sources'][seq];assert sha256(source['manifest_path'])==source['manifest_sha256']
        producer=read(source['manifest_path']);stream=producer['streams'][source_id]
        assert stream['frames']==manifest['coverage'][seq]['frames']
        assert sha256(stream['track_path'])==stream['track_sha256']==manifest['coverage'][seq]['output_sha256']
        gate=producer.get('gate_statistics',{}).get(source_id)
        if not source_model.get('baseline'):
            assert gate is not None and gate['frames']==stream['frames']
        gates[seq]=gate
    payload=dict(id=identifier,source_condition_id=source_id,split=split,condition=condition,
        metrics39=value['trackeval'],combined47={s['metric']:combined[s['metric']] for s in schema},
        per_sequence47=metric_rows,frames=manifest['frames'],sequences=sequences,
        metrics_source=str(source_path),metrics_source_sha256=sha256(source_path),
        official_CSV_file=str(destination.relative_to(ROOT)),official_CSV_sha256=sha256(csv_path),
        coverage=manifest['coverage'],source_manifest_references=manifest['sources'],
        gate_statistics_by_sequence=gates,metric_schema_provenance=provenance)
    public=DOCS/'perceptual_native_factorial_20261124'/split/(identifier+'.json')
    small_json(public,payload)
    return payload,dict(id=identifier,file=str(public.relative_to(ROOT)),sha256=sha256(public))


def write47(path,rows,schema):
    fields=['candidate','split','seq']+[s['metric'] for s in schema]
    if path.exists():
        with path.open() as stream:assert list(csv.DictReader(stream))==[{k:str(r[k]) for k in fields} for r in rows]
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    assert path.stat().st_size<2**20


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Export is CPU-only'
    plan,study,protocol,calibration,work,checkpoint,base=context();registration=registered(work)
    state=read(base/'state.json');stem=DOCS/'PERCEPTUAL_MFT_20261124_NATIVE_FACTORIAL_FORMAL'
    if state['status']=='COMPLETE_FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_DEV':
        decision=read(base/'not_triggered.json')
        assert decision['status']=='FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_COMPLETE_PARENT_DEV'
        small_json(stem.with_suffix('.json'),dict(status='COMPLETE_CONDITIONAL_FACTORIAL_NOT_TRIGGERED',
            decision=decision,science_sha256=sha256(SCIENCE),registry_sha256=sha256(work/REGISTRY),
            no_extra_fit_or_Test_started=True,original_parent_results_retained=True,
            goal_met=False,supported_innovations=0))
        text='# 联合消融未触发\n\n原正常IR未通过预登记的完整Dev强RGB／灰度条件；不补训00，不据Test改写触发条件。原六组正式Test结果仍由父研究保留。整体Goal尚未完成。\n'
        if stem.with_suffix('.md').exists():assert stem.with_suffix('.md').read_text()==text
        else:stem.with_suffix('.md').write_text(text)
        return
    assert state['status']=='COMPLETE_FACTORIAL_EXECUTION_PENDING_SCIENTIFIC_EXPORT'
    freeze_path=base/'Test_selection_frozen.json';freeze=read(freeze_path)
    coverage_path=base/'Test_coverage_plan.json';coverage=read(coverage_path)
    assert freeze['complete_all_four_Dev'] and not freeze['Test_accuracy_used_for_selection']
    assert coverage['Test_selection_sha256']==sha256(freeze_path)
    runtime=read(base/'individual_runtime.json')
    assert runtime['status']=='COMPLETE_FROZEN_FACTORIAL_INDIVIDUAL_PREFIX_RUNTIME'
    assert runtime['Test_selection_sha256']==sha256(freeze_path)
    assert {r['id'] for r in runtime['rows']}=={FRESH['id']}|{r['id'] for r in freeze['selected_unique_models']}
    schema,_=official_schema(plan['evaluation']['TrackEval_root'])
    dev,test,refs,views={},{},{'dev':[],'test':[]},{'dev':[],'test':[]}
    for row in freeze['Dev_rows']:
        assert sha256(row['source_metrics'])==row['source_metrics_sha256']
        raw=read(row['source_metrics']);config=raw['manifest']['config'];source_id=config['id']
        payload,reference=publish_case(row['id'],'dev',row['source_metrics'],source_id,row['model'],plan,protocol,schema)
        dev[row['id']]=payload;refs['dev'].append(reference)
    for model in [dict(FRESH)]+freeze['selected_unique_models']:
        source=coverage['sources'][model['id']]
        payload,reference=publish_case(model['id'],'test',source['metrics_json'],source['source_case_id'],model,plan,protocol,schema)
        test[model['id']]=payload;refs['test'].append(reference)
    parent_freeze=read(work/'native_pyramid_Test_selection_frozen.json')
    assert coverage['parent_Test_selection_sha256']==sha256(work/'native_pyramid_Test_selection_frozen.json')
    parent_models={r['id']:r for r in parent_freeze['models']};controls={}
    for role in ['source_RGB_canonical','matched_RGB','matched_gray','primary_IR_unavailable','primary_IR_image_roll']:
        key=parent_freeze['roles'][role]
        payload,reference=publish_case('CONTROL_'+role,'test',work/'native_pyramid_evaluation/test'/key/'metrics.json',
            key,parent_models.get(key),plan,protocol,schema)
        controls[role]=payload;refs['test'].append(reference)
    assert controls['source_RGB_canonical']['metrics39']['HOTA']==29.529
    fresh=test[FRESH['id']];off=controls['primary_IR_unavailable']
    assert off['metrics39']==fresh['metrics39'] and off['official_CSV_sha256']==fresh['official_CSV_sha256']
    assert all(off['coverage'][s]['output_sha256']==fresh['coverage'][s]['output_sha256'] for s in protocol['historical_test'])
    parent_dev=read(work/'native_pyramid_complete_Dev.json');parent_rows={r['id']:r for r in parent_dev['rows']}
    parent_selection=parent_dev['selection'];control_dev={}
    assert parent_selection['primary']['epoch']==freeze['original_primary_epoch']
    dev_ids=dict(matched_RGB=parent_selection['primary_mode_RGB_control']['id'],
        matched_gray=parent_selection['primary_mode_gray_control']['id'],
        source_RGB_fresh='NP_source_fresh_RGB',source_RGB_canonical='NP_source_canonical_RGB',
        primary_IR_unavailable=parent_selection['primary']['id']+'_off',
        primary_IR_image_roll=parent_selection['primary']['id']+'_roll')
    for role,key in dev_ids.items():
        row=parent_rows[key];assert sha256(row['source_metrics'])==row['source_metrics_sha256']
        payload,reference=publish_case('CONTROL_'+role,'dev',row['source_metrics'],key,row,plan,protocol,schema)
        control_dev[role]=payload;refs['dev'].append(reference)
    dev_off,dev_fresh=control_dev['primary_IR_unavailable'],control_dev['source_RGB_fresh']
    assert dev_off['metrics39']==dev_fresh['metrics39'] and dev_off['official_CSV_sha256']==dev_fresh['official_CSV_sha256']
    assert all(dev_off['coverage'][s]['output_sha256']==dev_fresh['coverage'][s]['output_sha256'] for s in protocol['dev'])
    analyses={}
    for policy,roles in freeze['roles'].items():
        analyses[policy]=necessary_numeric_evidence(
            {cell:dev[key]['combined47']['HOTA'] for cell,key in roles.items()},
            {cell:test[key]['combined47']['HOTA'] for cell,key in roles.items()})
    primary_id=freeze['roles']['paired_parent_primary_epoch']['11'];primary=test[primary_id]
    required=max(34.529,fresh['combined47']['HOTA']+5)
    for split,items in [('dev',{**dev,**{'CONTROL_'+k:v for k,v in control_dev.items()}}),
                        ('test',{**test,**{'CONTROL_'+k:v for k,v in controls.items()}})]:
        for identifier,payload in items.items():
            for row in payload['per_sequence47']:views[split].append(dict(row,candidate=identifier))
        write47(DOCS/'perceptual_native_factorial_20261124'/(split+'_metrics47.csv'),views[split],schema)
    report=dict(status='COMPLETE_FOUR_CELL_FACTORIAL_REQUIRES_SCIENTIFIC_ASSESSMENT',
        science_sha256=sha256(SCIENCE),registry_sha256=sha256(work/REGISTRY),
        Test_selection_sha256=sha256(freeze_path),coverage_plan_sha256=sha256(coverage_path),
        Dev_cases=refs['dev'],Test_cases=refs['test'],roles=freeze['roles'],effect_analyses=analyses,
        original_primary_unchanged=True,original_primary_Test_HOTA=primary['combined47']['HOTA'],
        primary_minus_fixed_RGB_pp=primary['combined47']['HOTA']-29.529,
        primary_minus_fresh_RGB_pp=primary['combined47']['HOTA']-fresh['combined47']['HOTA'],
        fixed_target_HOTA=34.529,fair_required_HOTA=required,numeric_target_met=primary['combined47']['HOTA']>=required,
        primary_minus_matched_RGB_pp=primary['combined47']['HOTA']-controls['matched_RGB']['combined47']['HOTA'],
        primary_minus_matched_gray_pp=primary['combined47']['HOTA']-controls['matched_gray']['combined47']['HOTA'],
        primary_minus_wrong_IR_pp=primary['combined47']['HOTA']-controls['primary_IR_image_roll']['combined47']['HOTA'],
        primary_Dev_minus_matched_RGB_pp=dev[primary_id]['combined47']['HOTA']-control_dev['matched_RGB']['combined47']['HOTA'],
        primary_Dev_minus_matched_gray_pp=dev[primary_id]['combined47']['HOTA']-control_dev['matched_gray']['combined47']['HOTA'],
        primary_Dev_minus_wrong_IR_pp=dev[primary_id]['combined47']['HOTA']-control_dev['primary_IR_image_roll']['combined47']['HOTA'],
        full_missing_IR_metrics_CSV_tracks_exact_to_fresh_RGB=True,
        runtime=runtime,source_hashes=registration['source_hashes'],metric_schema=schema,
        complete_Train_epochs_each=freeze['stage'],fixed_seed=0,additional_seed_work=False,
        Test_based_selection=False,historical_Test_previously_exposed=True,
        supported_innovations=0,goal_met=False,
        remaining_acceptance='Positive single/conditional effects are necessary numerical evidence only. Audit strong Dev/Test controls, actual input interventions, novelty/prior art, all architecture/formal-candidate coverage, costs and complete goal before claiming2–3innovations or completion.',
        limitations=study['limitations'],contrast_scope=read(SCIENCE)['gate_scope'])
    parent_report=DOCS/'PERCEPTUAL_MFT_20261124_NATIVE_PYRAMID_FORMAL.json'
    report['all_six_parent_normal_fits_and_removals_report']=dict(file=str(parent_report.relative_to(ROOT)),sha256=sha256(parent_report))
    small_json(stem.with_suffix('.json'),report)
    lines=['# 原生融合四组归因的完整结果','','主方法及两个选点策略由完整Dev冻结；保留全部负差值，不按Test改选。',
        '','| 策略 | S单独 | Q单独 | S在Q上 | Q在S上 | 组合 | 交互 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for policy,analysis in analyses.items():
        v=analysis['Test'];lines.append('| '+policy+' | '+' | '.join(f'{v[k]:+.6f}' for k in
            ['contrast_alone','quality_alone','contrast_given_quality','quality_given_contrast','combined','interaction'])+' |')
    lines+=['',f"原冻结主模型HOTA：{report['original_primary_Test_HOTA']:.6f}；相对固定RGB：{report['primary_minus_fixed_RGB_pp']:+.6f}点。",
        '','原276列CSV、全部47字段、逐序列及COMBINED、门控、来源和成本均分文件保存。成本为七视频32帧前缀抽样，不是15985帧全程FPS；不含离线IR生成。',
        '','数值效果不自动等于创新性。局部对比输入关闭仍保留全局图像统计，质量门控关闭仍保留时序方差的统计通路。历史Test暴露及RGB预训练见过Dev的限制保留。Goal尚需完整科研验收。']
    text='\n'.join(lines)+'\n'
    if stem.with_suffix('.md').exists():assert stem.with_suffix('.md').read_text()==text
    else:stem.with_suffix('.md').write_text(text)
    print(report['status'],len(dev),'Dev',len(test),'selectedTest',flush=True)


if __name__=='__main__':main()
