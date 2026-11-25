#!/usr/bin/env python3
"""Export all real fitted conditions, official47, rawCSV and independent costs."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.native_pyramid_experiment import read
from tools.neck72_runtime_study import exited
from tools.observe_task_utility import verify_observations
from tools.fusion_memory_observation import LENGTHS
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_controller import verify_runtime
from tools.task_utility_design import CANONICAL,FRESH,metric_row,require_test_freeze
from tools.task_utility_pipeline import current_source_hashes,verify_fit,verify_probe
from tools.task_utility_selection import load_stage
from tools.task_utility_study import STUDY,REGISTRY,context,registered


def validate_execution_state(base,state):
    if state['study_sha256']!=sha256(STUDY) or state['pipeline_registration_sha256']!=sha256(Path(base)/REGISTRY):
        raise ValueError('Original execution state and source registration required')
    required={'NORMAL_GPU_ENGINEERING','TRAIN_06','dev_stage_06_NN','dev_stage_06_CANONICAL',
        'dev_stage_06_SOURCE_OFFICIAL','dev_stage_06_OFFICIAL','dev_primary_interventions_NN',
        'dev_primary_interventions_OFFICIAL','test_frozen_NN','test_frozen_CANONICAL',
        'INDIVIDUAL_RUNTIME','test_frozen_SOURCE_OFFICIAL','test_frozen_OFFICIAL'}
    for stage in [9,12]:
        if state['through_epoch']>=stage:required|={f'TRAIN_{stage:02d}',f'dev_stage_{stage:02d}_NN',f'dev_stage_{stage:02d}_OFFICIAL'}
    phases=state['phases']
    if set(phases)-{'FINAL_EXPORT'}!=required:raise ValueError('All original phase barriers must finish')
    normal=set(read(Path(base)/REGISTRY)['normal_arms'])
    if len(normal)!=12:raise ValueError('Exactly twelve registered normal fit identities required')
    expected_jobs={'NORMAL_GPU_ENGINEERING':normal,'INDIVIDUAL_RUNTIME':{c['id'] for c in read(Path(base)/'Test_selection_frozen.json')['cases']}}
    for stage in [6,9,12]:
        if stage<=state['through_epoch']:expected_jobs[f'TRAIN_{stage:02d}']=normal
    trials=[('dev_stage_06','dev'),('dev_primary_interventions','dev'),('test_frozen','test')]
    trials += [(f'dev_stage_{stage:02d}','dev') for stage in [9,12] if stage<=state['through_epoch']]
    for phase,split in trials:
        specification=read(Path(base)/'specifications'/(phase+'.json'))
        expected_jobs[phase+'_NN']=set(LENGTHS[split])
        expected_jobs[phase+'_OFFICIAL']={m['id'] for m in specification['models'] if m['id']!=FRESH['id']}
        if specification['include_sources']:
            expected_jobs[phase+'_CANONICAL']={'source'}
            expected_jobs[phase+'_SOURCE_OFFICIAL']={FRESH['id'],CANONICAL['id']}
    if set(expected_jobs)!=required or any(set(phases[k])!=v for k,v in expected_jobs.items()):
        raise ValueError('Every actual fit,video,official object and cost job must finish;no partial phase accepted')
    for phase in required:
        if not phases[phase]:raise ValueError('No empty executed phases')
        for item in phases[phase].values():
            if (item['status']!='COMPLETE' or item['exit_code']!=0 or not item['argv_identity_observed'] or
                    not all(item.get(k) for k in ['pid','start_ticks','command_sha256']) or
                    not exited(item['pid'],command=item['command'],identity=item)):
                raise ValueError('Every original worker must exit0and retain verified process identity')
    return True


def write_exact(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Preserve previous differing export: '+str(path))
    else:
        with path.open('xb') as stream:stream.write(data)


def encoded(value):return (json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Final export is separate CPU work')
    plan,study,protocol,calibration,work,checkpoint,base=context();registration=registered(work)
    if registration['source_hashes']!=current_source_hashes():raise ValueError('Original full workflow source closure changed')
    state=read(base/'state.json');validate_execution_state(base,state);through=state['through_epoch']
    specification=base/'specifications/test_frozen.json';frozen=require_test_freeze(base,specification)
    complete=read(base/'complete_Dev.json');stage=load_stage(plan,protocol,base,through)
    if complete['selection']!=stage['selection']:raise ValueError('Actual complete Dev choice changed')
    selection=complete['selection'];cases=frozen['cases'];rows=[];published={}
    directory=ROOT/'docs/fusion_research/perceptual_20261125_task_utility_formal'
    dev_artifacts=[];dev_cases=stage['cases']+complete['additional_cases']
    if len(dev_cases)!=20*through+6:raise ValueError('Every normal/input epoch and required additional Dev object must remain')
    for item in dev_cases:
        row=metric_row(base,item,protocol);config=read(item['config']);data=read(row['source_metrics'])
        if row['split']!='dev' or row['frames']!=4834:raise ValueError('Complete original Dev required')
        if verify_observations(config['specification'],item['config'],base,registration,plan,protocol)!=data['manifest']:
            raise ValueError('Actual Dev observations changed')
        case=dict(id=item['id'],model=config['model'],metrics39=data['metrics39'],combined47=data['combined47'],
            per_sequence47=data['per_sequence47'],official_schema=data['official_schema'],
            original_metrics_sha256=sha256(row['source_metrics']),official_CSV_sha256=row['official_CSV_sha256'])
        outputs=[(directory/'Dev/official_CSV'/(item['id']+'.csv'),
            (Path(row['source_metrics']).parent/'trackeval_output/pedestrian_detailed.csv').read_bytes()),
            (directory/'Dev/metrics47'/(item['id']+'.json'),encoded(case))]
        for path,content in outputs:
            if len(content)>2**20:raise ValueError('Split oversized Dev artifacts;do not discard negative conditions')
            write_exact(path,content);published[str(path.relative_to(ROOT))]=sha256(path)
        dev_artifacts.append(dict(id=item['id'],metrics_artifact=str(outputs[1][0].relative_to(ROOT)),
            original_metrics_sha256=case['original_metrics_sha256'],official_CSV_sha256=row['official_CSV_sha256']))
    for item in cases:
        row=metric_row(base,item,protocol);config=read(item['config']);data=read(row['source_metrics'])
        if row['split']!='test' or row['frames']!=15985:raise ValueError('Every formal object needs all7Test')
        if verify_observations(specification,item['config'],base,registration,plan,protocol)!=data['manifest']:
            raise ValueError('Actual Test source observations changed')
        runtime=verify_runtime(base,item,registration)
        export_case=dict(id=item['id'],model=config['model'],metrics39=data['metrics39'],combined47=data['combined47'],
            per_sequence47=data['per_sequence47'],official_schema=data['official_schema'],
            original_metrics_sha256=sha256(row['source_metrics']),original_CSV_sha256=row['official_CSV_sha256'])
        destinations=[(directory/'official_CSV'/f"{item['id']}.csv",
            (Path(row['source_metrics']).parent/'trackeval_output/pedestrian_detailed.csv').read_bytes()),
            (directory/'metrics47'/f"{item['id']}.json",encoded(export_case)),
            (directory/'runtime'/f"{item['id']}.json",encoded(runtime))]
        for path,content in destinations:
            if len(content)>2**20:raise ValueError('Split oversized result artifacts;never omit required data')
            write_exact(path,content);published[str(path.relative_to(ROOT))]=sha256(path)
        rows.append(dict(config['model'],metrics39=data['metrics39'],combined47=data['combined47'],
            selection_metrics=data['selection_metrics'],cost=runtime['costs'],
            metrics47_artifact=str(destinations[1][0].relative_to(ROOT)),runtime_artifact=str(destinations[2][0].relative_to(ROOT)),
            original_metrics_sha256=sha256(row['source_metrics']),official_CSV_sha256=row['official_CSV_sha256']))
    training=[]
    for fit in study['normal_fits']:
        proof=verify_probe(base,fit['id'],study,registration);verified=verify_fit(base,fit['id'],through,study,registration)
        path=base/'fits'/fit['id']/'progress.json';value=read(path)
        destination=directory/'training'/(fit['id']+'.json');content=path.read_bytes()
        if len(content)>2**20:raise ValueError('Split oversized training history;do not discard it')
        write_exact(destination,content);published[str(destination.relative_to(ROOT))]=sha256(destination)
        costs=[]
        for budget in [6,9,12]:
            if budget>through:continue
            job=state['phases'][f'TRAIN_{budget:02d}'][fit['id']]
            costs.append({k:job[k] for k in ['pid','start_ticks','command_sha256','slot','exit_code',
                'worker_wall_lower_seconds','worker_wall_upper_seconds','observed_peak_RSS_bytes'] if k in job})
        training.append(dict(fit,complete_Train_epochs=through,unique_frames_each_epoch=27251,
            actual_optimizer_updates=value['actual_optimizer_updates'],progress_artifact=str(destination.relative_to(ROOT)),
            source_progress_sha256=sha256(path),proof=proof,verified=verified,actual_process_costs=costs))
    by_id={r['id']:r for r in rows};roles=frozen['roles'];primary=by_id[roles['primary_IR']]
    hota=lambda name:by_id[name]['combined47']['HOTA']
    quad={v:hota('TU_'+v+'_ir') for v in ['no_aux_labels','foreground_labels','utility_labels','foreground_utility_labels']}
    assessment=dict(primary_Test_HOTA=primary['combined47']['HOTA'],original_baseline_Test_HOTA=29.529,
        primary_minus_original_Test_HOTA=primary['combined47']['HOTA']-29.529,
        target_Test_HOTA=34.529,target_numeric_threshold_met=primary['combined47']['HOTA']>=34.529,
        primary_minus_matched_RGB_HOTA=hota(roles['primary_IR'])-hota(roles['matched_RGB']),
        primary_minus_frozen_strongest_Dev_RGB_HOTA=hota(roles['primary_IR'])-hota(roles['strongest_Dev_RGB']),
        primary_minus_frozen_strongest_Dev_gray_HOTA=hota(roles['primary_IR'])-hota(roles['strongest_Dev_gray']),
        IR_supervision_factorial_own_Dev_best_Test_HOTA=quad,
        foreground_marginal_given_utility=quad['foreground_utility_labels']-quad['utility_labels'],
        utility_marginal_given_foreground=quad['foreground_utility_labels']-quad['foreground_labels'],
        factorial_interaction=quad['foreground_utility_labels']-quad['foreground_labels']-quad['utility_labels']+quad['no_aux_labels'],
        scope='Frozen own-Dev-best supervised fits;inference gate removals remain separate,not retrained ablations. No Test reselection or originality claim.')
    result=dict(status='COMPLETE_TASK_UTILITY_FORMAL_EXPORT_REQUIRES_GLOBAL_SCIENTIFIC_ASSESSMENT',
        study_sha256=sha256(STUDY),pipeline_registration_sha256=sha256(base/REGISTRY),
        complete_Test_objects=26,complete_Test_frames_each=15985,official_fields=47,raw_CSV_columns=276,
        all_original_choices_preserved=True,decision_split='dev',Test_accuracy_used_for_selection=False,
        through_epoch=through,selection=selection,roles=roles,rows=rows,training=training,
        complete_Dev_objects=len(dev_cases),all_Dev_epoch_and_intervention_artifacts=dev_artifacts,
        complete_Dev_sha256=sha256(base/'complete_Dev.json'),Test_selection_sha256=sha256(base/'Test_selection_frozen.json'),
        published_artifact_hashes=published,assessment=assessment,
        historical_Test_exposed=True,original_RGB_pretraining_saw_present_Dev=True,auxiliary_RGB_derived=True,
        goal_met=False,supported_innovations=0,
        scope='Complete12formal fits/allnormal,inputintervention,inferencegate,geometryandtwoRGBreferenceTestobjects. '
            'All47pervideo/combined,original276CSV,epochhistoriesandactualindependent25B1plus1canonicalB4costspreserved. '
            'Numericthresholdalonecannotcompleteglobalgoal;independentcontributions,priorartandfour-directionauditremainroot-owned.')
    output=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_FORMAL_RESULTS.json'
    content=encoded(result)
    if len(content)>2**20:raise ValueError('Split aggregate report while retaining all required results')
    registered(work);require_test_freeze(base,specification);write_exact(output,content)
    text=['# TaskUtility: complete frozen Test results','',result['scope'],'',
        f"Primary Test HOTA: {assessment['primary_Test_HOTA']:.6f}; versus original RGB: {assessment['primary_minus_original_Test_HOTA']:+.6f}.",
        '', '| Case | HOTA | AssA | IDF1 | NN batch | Prefix FPS |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:text.append('| '+row['id']+' | '+' | '.join(f"{row['combined47'][k]:.6f}" for k in ['HOTA','AssA','IDF1'])+
        f" | {row['cost']['NN_batch_size']} | {row['cost']['FPS']:.3f} |")
    text+=['','Costs are168timedprefixframes percase,notfullTestFPS;B4batchlatency differsfromB1framelatency. '
        'All12ownDevbestconditionsincludingnegativeDevretained. No Test-based choice change,extra seedsorindependentinnovationclaim.']
    write_exact(output.with_suffix('.md'),('\n'.join(text)+'\n').encode())
    print(json.dumps(dict(status=result['status'],assessment=assessment)),flush=True)


if __name__=='__main__':main()
