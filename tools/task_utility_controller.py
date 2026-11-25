"""The complete same-budget TaskUtility experiment graph; no implicit GPU release."""
import math
from pathlib import Path

from tools.fusion_memory_evaluation import validate_mot
from tools.fusion_memory_observation import LENGTHS
from tools.native_pyramid_experiment import frozen_json,read
from tools.observe_task_utility import verify_observations
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.task_utility_design import FRESH,CANONICAL,additional_Dev_models,metric_row,register_trial,stage_models
from tools.task_utility_pipeline import task,fit_tasks,verify_probe,verify_fit
from tools.task_utility_runtime_sources import RUNTIME,runtime_routes
from tools.task_utility_selection import complete_stage,freeze_Test,load_stage,stage_path
from tools.task_utility_study import ROOT,STUDY,REGISTRY,arm_path


def authorize(plan,protocol,base,study,stage):
    prior={}
    if stage not in [6,9,12]:raise ValueError('Original common6/9/12budget only')
    if stage>6:
        old=load_stage(plan,protocol,base,stage-3)
        if not old['budget_decision']['extend_all'] or old['budget_decision']['through_epoch']!=stage:
            raise ValueError('No extra Train without the actual complete common Dev decision')
        prior[str(stage_path(base,stage-3))]=sha256(stage_path(base,stage-3))
    fits=study['normal_fits']
    if len(fits)!=12 or len({f['id'] for f in fits})!=12:raise ValueError('All twelve same-budget fits required')
    dump(Path(base)/'stage_authorized.json',dict(study_sha256=sha256(STUDY),through_epoch=stage,
        fit_ids=[f['id'] for f in fits],arm_hashes={f['id']:sha256(arm_path(base,f['id'])) for f in fits},
        prior_complete_Dev_files=prior,Test_accuracy_used=False))


def verify_sequence(specification,entry,models,registration,split):
    spec=read(specification);directory=Path(spec['stream_directory']);sequence=entry['id'];count=LENGTHS[split][sequence]
    value=read(directory/'sequence_manifests'/(sequence+'.json'));p=value['provenance']
    if (value['status']!='COMPLETE_TASK_UTILITY_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT' or
            value['split']!=split or value['sequence']!=sequence or value['frames']!=count or
            value['complete_source_sequence_frames']!=count or value['formal_Dev_Test_sequence_coverage'] is not True or
            value['models']!=models or value['GT_read'] or value['future_frame_input'] or value['previous_outputs_rewritten'] or
            value['independent_input_batch_size']!=1 or value['feature_or_detection_disk_cache'] or
            p['source_versions']!=registration['source_hashes'] or p['specification_sha256']!=sha256(specification) or
            p['pipeline_registration_sha256']!=spec['pipeline_registration_sha256'] or p['study_sha256']!=sha256(STUDY) or
            p['physical_gpu']!=entry['slot'] or p['parent_fresh_packets_and_tracks_exact'] is not True or
            set(value['streams'])!={m['id'] for m in models}):
        raise ValueError('Actual complete registered source-matched B1sequence required')
    for name,stream in value['streams'].items():
        path=directory/name/'track_results'/(sequence+'.txt')
        if Path(stream['track_path'])!=path or sha256(path)!=stream['track_sha256'] or stream['frames']!=count:
            raise ValueError('Original complete MOT file changed')
        validate_mot(path,count,stream['track_rows'])
    expected={m['id'] for m in models if not m.get('baseline')}
    if set(value['gate_statistics'])!=expected:raise ValueError('Every learned stream needs its actual scalar diagnostics')
    for gates in value['gate_statistics'].values():
        if gates['frames']!=count or gates['available_frames']+gates['missing_frames']!=count or len(gates['levels'])!=3:
            raise ValueError('Full actual gate/missing coverage required')
    return value


def observe_trial(queue,plan,protocol,phase,split,models,include_sources,registration):
    specification,cases=register_trial(queue.base,phase,split,models,include_sources)
    jobs=[task(s,'observe_task_utility.py','produce','--split',split,'--sequence',s,'--specification',specification)
        for s in sorted(LENGTHS[split],key=lambda s:-LENGTHS[split][s])]
    queue.jobs(phase+'_NN',jobs,True,lambda e:verify_sequence(specification,e,models,registration,split))
    if include_sources:
        def canonical(entry):
            for sequence,count in LENGTHS[split].items():
                value=read(Path(read(specification)['stream_directory'])/'sequence_manifests'/(sequence+'.canonical.json'))
                if (value['status']!='COMPLETE_TASK_UTILITY_CANONICAL_REPLAY_WITHOUT_GT' or value['frames']!=count or
                        value['specification_sha256']!=sha256(specification) or value['GT_read'] or
                        value['future_frame_input'] or value['previous_outputs_rewritten']):
                    raise ValueError('Actual full canonical replay without GT required')
        queue.jobs(phase+'_CANONICAL',[task('source','observe_task_utility.py','canonical','--split',split,
            '--specification',specification)],False,canonical)
    # Reverify complete original files before any evaluator is launched.
    for case in cases:verify_observations(specification,case['config'],queue.base,registration,plan,protocol)
    return specification,cases


def evaluate_cases(queue,plan,protocol,phase,cases):
    source=[c for c in cases if c['id'] in [FRESH['id'],CANONICAL['id']]]
    ordinary=[c for c in cases if c not in source]
    for suffix,group in [('SOURCE_OFFICIAL',source),('OFFICIAL',ordinary)]:
        if not group:continue
        jobs=[dict(task(c['id'],'observe_task_utility.py','evaluate','--split',read(c['config'])['split'],
            '--config',c['config']),**c) for c in group]
        queue.jobs(phase+'_'+suffix,jobs,False,lambda e:metric_row(queue.base,e,protocol))
    rows=[metric_row(queue.base,c,protocol) for c in cases]
    for row in rows:
        if row.get('intervention')!='unavailable':continue
        split=row['split'];fresh=read(queue.base/'evaluation'/split/FRESH['id']/'metrics.json');data=read(row['source_metrics'])
        if any(data[k]!=fresh[k] for k in ['metrics39','combined47','per_sequence47','official_CSV_sha256']):
            raise ValueError('Missing input must reproduce all original fresh RGB official metrics and CSV')
    return rows


def verify_runtime(base,case,registration):
    value=read(Path(base)/'individual_runtime'/(case['id']+'.json'));configuration=read(case['config'])
    canonical=configuration['model']==CANONICAL;cost=value['costs'];batch=4 if canonical else 1
    if (value['status']!='COMPLETE_TASK_UTILITY_INDIVIDUAL_SOURCE_MATCHED_PREFIX_COST' or
            value['case']!=configuration['model'] or value['config_sha256']!=case['config_sha256'] or
            value['specification_sha256']!=configuration['specification_sha256'] or
            value['pipeline_registration_sha256']!=sha256(Path(base)/REGISTRY) or
            value['study_sha256']!=sha256(STUDY) or value['runtime_clarification_sha256']!=sha256(RUNTIME) or
            cost['NN_batch_size']!=batch or cost['measured_frames']!=168 or cost['measured_batches']!=168//batch or
            value['physical_gpu'] not in ['4','5'] or value['peak_process_RSS_GiB']>8 or
            value['peak_allocated_GPU_GiB']<=0 or value['methods_measured_simultaneously']!=1 or
            value['GT_read'] or value['Test_accuracy_read'] or value['image_or_FPN_disk_cache'] or
            not value['source_and_output_equivalence_before_and_during_timing'] or
            value['gate_observer_attached'] or value['original_RGB_backbone_and_head_precision']!='FP16'):
        raise ValueError('Actual source-matched individual GPU prefix cost required')
    expected=read(ROOT/'configs/perceptual_gpu_handoff_20261123.json')['future_gpu_uuids'][value['physical_gpu']]
    if str(value['GPU_uuid']).lower().replace('gpu-','')!=expected.lower().replace('gpu-',''):
        raise ValueError('Actual runtime GPU UUID differs')
    if not all(sha256(p)==s for p,s in value['original_file_hashes'].items()):raise ValueError('Original runtime source changed')
    from tools.task_utility_runtime import aggregate_runtime
    if aggregate_runtime(value['sequences'])!=cost:raise ValueError('Actual runtime samples and stored costs differ')
    if [s['timing']['sequence'] for s in value['sequences']]!=list(LENGTHS['test']):
        raise ValueError('All seven original cost prefixes in order required')
    if not math.isfinite(cost['FPS']) or cost['FPS']<=0:raise ValueError('Actual finite cost required')
    return value


def execute(queue,plan,study,protocol,registration):
    """All12fits,complete stages,all26Testandcost; negatives never remove jobs."""
    base=queue.base
    queue.jobs('NORMAL_GPU_ENGINEERING',fit_tasks(study,'probe'),True,
        lambda e:verify_probe(base,e['id'],study,registration))
    all_dev=[];stage=6;first=1
    while True:
        authorize(plan,protocol,base,study,stage)
        queue.jobs(f'TRAIN_{stage:02d}',fit_tasks(study,'fit',stage),True,
            lambda e:verify_fit(base,e['id'],stage,study,registration))
        phase=f'dev_stage_{stage:02d}'
        _,cases=observe_trial(queue,plan,protocol,phase,'dev',stage_models(study,base,first,stage),not all_dev,registration)
        evaluate_cases(queue,plan,protocol,phase,cases);all_dev.extend(cases)
        completed=complete_stage(plan,protocol,base,all_dev,stage)
        if not completed['budget_decision']['extend_all']:break
        first,stage=stage+1,completed['budget_decision']['through_epoch']
    frozen_json(base/'normal_selection_frozen.json',dict(status='FROZEN_TASK_UTILITY_NORMAL_DEV_CHOICE',
        study_sha256=sha256(STUDY),through_epoch=stage,selection=completed['selection'],
        complete_stage_sha256=sha256(stage_path(base,stage)),Test_accuracy_used=False))
    _,additional=observe_trial(queue,plan,protocol,'dev_primary_interventions','dev',
        additional_Dev_models(base,completed['selection']),False,registration)
    evaluate_cases(queue,plan,protocol,'dev_primary_interventions',additional)
    specification,cases,frozen=freeze_Test(plan,protocol,base,stage,additional)
    observe_trial(queue,plan,protocol,'test_frozen','test',read(specification)['models'],True,registration)
    routes=runtime_routes(base,specification)
    jobs=[dict(task(c['id'],'benchmark_task_utility.py','--specification',specification,'--config',c['config'],
        physical=['4','5'][index%2]),**c) for index,c in enumerate(routes)]
    queue.jobs('INDIVIDUAL_RUNTIME',jobs,True,lambda e:verify_runtime(base,e,registration))
    evaluate_cases(queue,plan,protocol,'test_frozen',cases)
    queue.state['through_epoch']=stage;queue.state['status']='COMPLETE_TASK_UTILITY_EXECUTION_PENDING_EXPORT';queue.save()
