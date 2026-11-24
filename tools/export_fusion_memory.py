#!/usr/bin/env python3
"""Publish all frozen F4 Dev/Test47/original CSV/cost files after actual execution.

Preflight is read-only. Test values never choose a model or comparator. This
substudy cannot declare the complete multi-direction research Goal achieved.
"""
import argparse
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_design import make_case
from tools.fusion_memory_evaluation import official_csv47, read, sha
from tools.fusion_memory_observation import LENGTHS, write_json
from tools.fusion_memory_parent_io import DevParentFiles, require
from tools.fusion_memory_results import (analysis_roles, checked_cost, checked_metric, exact_full_behavior,
    publish_case, scientific_effects, small_json, small_bytes)
from tools.fusion_memory_runtime_sources import original_source, runtime_references
from tools.fusion_memory_workflow import FusionMemoryWorkflow, STUDY
from tools.neck72_runtime_study import exited

DOCS = ROOT/'docs/fusion_research'
PUBLIC = DOCS/'perceptual_fusion_memory_20261125'
REPORT = DOCS/'PERCEPTUAL_MFT_20261125_FUSION_MEMORY_FORMAL'
EXECUTION_FINISHED = 'COMPLETE_F4_EXECUTION_PENDING_SCIENTIFIC_EXPORT'


def metric_context(workflow, specification):
    spec, registration = workflow.validate_specification(specification)
    split = spec['split']
    folder = 'test' if split == 'test' else 'train'
    hashes = {str(Path(workflow.plan['evaluation']['GT_root'])/folder/seq/'gt/gt.txt'):
        workflow.protocol['derived_gt_sha256'][folder+'/'+seq] for seq in LENGTHS[split]}
    proof = workflow.base/'gpu_engineering.json'
    value = read(proof)
    require(value['status'] == 'COMPLETE_REAL_GPU_F4_SOURCE_AND_MEMORY_ENGINEERING' and
            value['normal_specification_sha256'] == sha(workflow.base/'specifications/dev_normal.json') and
            value['pipeline_registration_sha256'] == sha(workflow.base/'pipeline_registered.json') and
            value['actual_GPU_tensors_verified'] is True, 'Real registered F4 GPU engineering required for publication')
    return dict(spec=spec, specification=str(specification), source_versions=registration['source_hashes'],
        registration_sha256=sha(workflow.base/'pipeline_registered.json'), GT_hashes=hashes,
        source_RGB_sha256=read(ROOT/'configs/perceptual_correlation_repair_20261123.json')['source_RGB_sha256'],
        GPU_engineering_sha256=sha(proof),
        GPU_uuids=read(ROOT/'configs/perceptual_gpu_handoff_20261123.json')['future_gpu_uuids'])


def execution_jobs(specifications, specs, python):
    result = {'GPU_ENGINEERING': {'actual': [python, '-u', 'tools/probe_f4_gpu_engineering.py',
        '--specification', str(specifications['dev_normal'])]}}
    for name, prefix in [('dev_normal','DEV_NORMAL'), ('dev_additional','DEV_ADDITIONAL'), ('test_frozen','TEST')]:
        path, spec = specifications[name], specs[name]
        result[prefix+'_PRODUCE'] = {seq: [python,'-u','tools/observe_fusion_memory.py','produce',
            '--specification',str(path),'--sequence',seq] for seq in LENGTHS[spec['split']]}
        result[prefix+'_EVALUATE'] = {c['id']: [python,'-u','tools/observe_fusion_memory.py','evaluate',
            '--specification',str(path),'--case',c['id']] for c in spec['cases']}
    result['RUNTIME'] = {'all_frozen': [python,'-u','tools/benchmark_fusion_memory.py',
        '--specification',str(specifications['test_frozen'])]}
    return result


def validate_execution_state(state, specifications, specs, process_exited=exited):
    require(state['status'] == EXECUTION_FINISHED and state['fixed_seed'] == 0 and
            state['physical_gpus'] == ['4','5'] and state['maximum_GPU_workers'] == 2,
            'All frozen F4 execution must complete before scientific export')
    jobs = execution_jobs(specifications, specs, state['worker_python'])
    require(set(state['phases']) == set(jobs), 'Missing or unregistered F4 execution phase')
    for phase, entries in jobs.items():
        actual = state['phases'][phase]
        require(set(actual) == set(entries), 'Every frozen case/sequence must execute, including negatives')
        for name, command in entries.items():
            job = actual[name]
            require(job['status'] == 'COMPLETE' and job['exit_code'] == 0 and job['command'] == command and
                    job['start_ticks'] > 0 and process_exited(job['pid'],command=command,identity=job),
                    'Prior F4 worker is incomplete, different, failed or still running')
    return jobs


def zero_Test_original_csv(workflow, case, metric, schema):
    _, _, pin = original_source(workflow, case, next(iter(LENGTHS['test'])))
    base = workflow.work/'native_pyramid_evaluation' if pin['family'] == 'native' else workflow.work/'native_spatial/evaluation'
    path = base/'test'/pin['model_id']/'trackeval_output/pedestrian_detailed.csv'
    parsed = official_csv47(path, schema, 'test')
    require(parsed['official_CSV_sha256'] == metric['official_CSV_sha256'],
            'Zero-memory Test must preserve the original visual all276-column CSV')
    return dict(path=str(path), sha256=sha(path))


def export_complete(workflow, public=PUBLIC, report=REPORT):
    require(workflow.predecessor_ready(), 'Wait for the complete previous GPU chain and actual exits')
    paths = {name: workflow.base/'specifications'/(name+'.json') for name in ['dev_normal','dev_additional','test_frozen']}
    contexts = {name: metric_context(workflow,path) for name,path in paths.items()}
    specs = {name: context['spec'] for name,context in contexts.items()}
    state_path = workflow.base/'state.json';state = read(state_path)
    require(state['pipeline_registration_sha256'] == sha(workflow.base/'pipeline_registered.json'), 'Execution registration changed')
    validate_execution_state(state, paths, specs)
    snapshot = workflow.base/'execution_finished_before_export.json'
    small_json(snapshot,state)
    frozen_path = workflow.base/'frozen_Test_case_plan.json';frozen = read(frozen_path)
    require(frozen['cases'] == specs['test_frozen']['cases'], 'Final Test set differs from complete Dev freeze')
    schema, families = official_schema(workflow.plan['evaluation']['TrackEval_root'])
    objects, references = {'dev':{},'test':{}}, {'dev':{},'test':{}}
    for name in ['dev_normal','dev_additional']:
        for case in specs[name]['cases']:
            path = workflow.base/'evaluation'/name/case['id']/'metrics.json'
            value,reference = checked_metric(path,case,contexts[name],schema)
            objects['dev'][case['id']],references['dev'][case['id']] = value,reference
    Dev_scores = {name: dict(id=name,split='dev',frames=4834,sequences=list(LENGTHS['dev']),seed=0,
        intervention=v['case']['intervention'],metrics=v['selection_metrics']) for name,v in objects['dev'].items()}
    roles = analysis_roles(workflow.parents.plan,frozen,Dev_scores)
    role_freeze = workflow.base/'export_Dev_analysis_roles_before_Test_read.json'
    role_value = dict(role_plan=roles,complete_Dev_metrics=references['dev'],original_Test_freeze_sha256=sha(frozen_path),
        pipeline_registration_sha256=sha(workflow.base/'pipeline_registered.json'), Test_accuracy_read=False)
    if role_freeze.exists():require(read(role_freeze)==role_value,'Preserve original Dev-derived analysis roles')
    else:write_json(role_freeze,role_value,exclusive=True)
    # Actual Test accuracy reads begin only after all Dev-only roles above exist.
    for case in frozen['cases']:
        path = workflow.base/'evaluation/test_frozen'/case['id']/'metrics.json'
        value,reference = checked_metric(path,case,contexts['test_frozen'],schema)
        objects['test'][case['id']],references['test'][case['id']] = value,reference
    require(set(objects['dev']) == set(objects['test']) == {c['id'] for c in frozen['cases']}, 'All normal/additional cases need full Dev and Test')
    fresh = workflow.parents.plan['role_visual_keys']['fresh_RGB']
    for case in frozen['cases']:
        if case['intervention'] != 'IR_unavailable':continue
        control = next(c for c in frozen['cases'] if c['visual_key']==fresh and c['intervention']=='normal' and
            c['appearance_weight']==case['appearance_weight'] and (c['profile']=='byte' or
            (c['profile']=='rgb_uniform' and c['minimum_reliability']==case['minimum_reliability'])))
        for split in ['dev','test']:exact_full_behavior(objects[split][case['id']],objects[split][control['id']],split)
    original_zero = {c['id']:zero_Test_original_csv(workflow,c,objects['test'][c['id']],schema)
        for c in frozen['cases'] if c['profile']=='byte'}
    runtime_path = workflow.base/'individual_runtime.json';runtime = read(runtime_path)
    require(runtime['status']=='COMPLETE_F4_INDIVIDUAL_SOURCE_MATCHED_PREFIX_RUNTIME' and
        runtime['specification_sha256']==sha(paths['test_frozen']) and
        runtime['pipeline_registration_sha256']==sha(workflow.base/'pipeline_registered.json') and
        [r['case'] for r in runtime['rows']]==frozen['cases'], 'Every frozen case requires its actual independent runtime')
    costs = {}
    for case,row in zip(frozen['cases'],runtime['rows']):
        source,evidence = runtime_references(workflow,paths['test_frozen'],case)
        costs[case['id']] = checked_cost(row,case,paths['test_frozen'],sha(workflow.base/'pipeline_registered.json'),
            objects['test'][case['id']],evidence,source,contexts['test_frozen']['GPU_uuids'])
    canonical_path = Path(workflow.plan['frozen_detection_sources']['rgb_test'])/'trackeval_output/pedestrian_detailed.csv'
    canonical = official_csv47(canonical_path,schema,'test')
    effects = scientific_effects(roles,objects['dev'],objects['test'])
    # Publish only after all cases, sources, original zeros, missing controls and costs pass.
    published = {'dev':[],'test':[]}
    for split in ['dev','test']:
        for case in frozen['cases']:
            published[split].append(publish_case(public,split,case,objects[split][case['id']],references[split][case['id']],schema))
    public_costs = {name:small_json(Path(public)/'costs'/(name+'.json'),row) for name,row in costs.items()}
    canonical_ref = small_bytes(Path(public)/'canonical_RGB_original276.csv',canonical_path.read_bytes())
    primary = objects['test'][roles['roles']['primary_IR']]['combined47']
    direct_byte = next(c for c in frozen['cases'] if c['visual_key']==fresh and c['profile']=='byte' and c['intervention']=='normal')
    direct = objects['test'][direct_byte['id']]['combined47']
    required = max(34.529,canonical['combined47']['HOTA']+5,direct['HOTA']+5)
    result = dict(status='COMPLETE_F4_FORMAL_EXPORT_REQUIRES_GLOBAL_SCIENTIFIC_ASSESSMENT',
        study_sha256=sha(STUDY),pipeline_registration_sha256=sha(workflow.base/'pipeline_registered.json'),
        complete_Dev_Test_freeze_sha256=sha(frozen_path),Dev_analysis_roles_sha256=sha(role_freeze),role_plan=roles,
        Dev_cases=published['dev'],Test_cases=published['test'],individual_costs=public_costs,
        official_schema=schema,official_family_field_counts=families,original_columns=276,
        canonical_RGB=canonical_ref,canonical_RGB_Test47=canonical['combined47'],fresh_B1_direct_RGB_Test47=direct,primary_Test47=primary,
        primary_minus_fixed_RGB_HOTA_pp=primary['HOTA']-29.529,fixed_RGB_HOTA=29.529,fixed_target_HOTA=34.529,
        fair_direct_RGB_plus5_threshold=required,numeric_fixed_target_met=primary['HOTA']>=34.529,
        numeric_fair_direct_RGB_plus5_met=primary['HOTA']>=required,effect_analyses=effects,
        original_zero_Test_CSV=original_zero,execution_state_source=dict(path=str(snapshot),sha256=sha(snapshot)),
        all_negative_cases_retained=True,Test_based_promotion=False,per_sequence_averaging_used=False,
        historical_Test_exposed=True,original_RGB_pretraining_saw_current_Dev=True,
        auxiliary_is_RGB_derived_not_independent_thermal=True,
        runtime_scope='Seven32-frame prefixes,8warmup24timed,168timed frames percase; not full15985-frameFPS',
        nonparameter_memory_scope='Observed prefix logical arrays/tensors, excludes NN/Python heap; not full-video peak',
        four_directions_complete=False,supported_innovations=0,goal_met=False)
    workflow.registered();workflow.parents.reader.verify_unchanged()
    text=(f'# F4 完整冻结结果\n\n主方法完整七段 Test HOTA {primary["HOTA"]:.6f}，相对固定 RGB29.529 为 '
        f'{primary["HOTA"]-29.529:+.6f} 点。主方法与所有比较对象来自完整 Dev 冻结，未据 Test 改选。\n\n'
        '全部负结果、47字段逐序列/官方汇总、原276列CSV与逐配置独立成本见同名JSON引用。'
        '普通RGB记忆与融合记忆分开说明；数值条件不自动证明创新或整体Goal完成。'
        '历史Test暴露、原RGB预训练见过Dev、辅助由RGB生成的限制保留。\n')
    small_bytes(Path(report).with_suffix('.md'),text.encode())
    small_json(Path(report).with_suffix('.json'),result)
    return result


def export_ineligible(workflow, report=REPORT):
    require(workflow.predecessor_ready(),'Wait for complete predecessor export and actual exits')
    workflow.registered()
    reader = DevParentFiles();evidence = reader.load()
    require(evidence['selection']['chosen_family'] is None,'Eligible parents must follow complete F4 execution')
    selection = small_json(PUBLIC/'ineligible_parent_selection.json',evidence['selection'])
    files = list(evidence['verified_files'].items())
    file_parts = [small_json(PUBLIC/'ineligible_parent_file_evidence'/(f'{start//100:03d}.json'),dict(files[start:start+100]))
                  for start in range(0,len(files),100)]
    result = dict(status='F4_NOT_ELIGIBLE_FROM_ORIGINAL_COMPLETE_DEV_NOT_AN_F4_EXPERIMENT',
        study_sha256=sha(STUDY),parent_selection=selection,verified_parent_files=file_parts,
        F4_fitted=False,F4_evaluated=False,Test_accuracy_input=False,
        four_directions_complete=False,supported_innovations=0,goal_met=False)
    small_bytes(Path(report).with_suffix('.md'),('# F4 尚无合格视觉父模型\n\n两个原视觉家族的完整Dev证据未满足已登记资格。'
        '本结果不表示F4已进行实验或已被证据排除，不能计为四方向完成。原负结果保留，整体Goal继续。\n').encode())
    reader.verify_unchanged();workflow.registered()
    small_json(Path(report).with_suffix('.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','complete','ineligible'])
    args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','F4 export and preflight are CPU-only')
    workflow=FusionMemoryWorkflow()
    if args.action=='preflight':
        import json
        print(json.dumps(workflow.preflight(),indent=2),flush=True)
    elif args.action=='complete':export_complete(workflow)
    else:export_ineligible(workflow)


if __name__=='__main__':main()
