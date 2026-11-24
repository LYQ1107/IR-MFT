"""Synthetic complete-length files/costs; none are actual F4 experiment results."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from test_fusion_memory_design import plan, normal_scores, additional_scores
from test_fusion_memory_evaluation import official
from test_fusion_memory_runtime import source_fixture
from tools.fusion_memory_design import freeze_case_plan
from tools.fusion_memory_evaluation import official_csv47, read, sha, verify_observations
from tools.fusion_memory_observation import LENGTHS, write_json
from tools.fusion_memory_online import visual_routes
from tools.fusion_memory_results import (analysis_roles, checked_cost, checked_metric, exact_full_behavior,
    publish_case, scientific_effects, small_bytes)
from tools.fusion_memory_runtime import aggregate_cost
from tools.fusion_memory_runtime_sources import runtime_references
from tools.export_fusion_memory import execution_jobs, validate_execution_state, EXECUTION_FINISHED

UUIDS={'4':'GPU-daa9b388-4540-242c-83d7-3261bb232a7c','5':'GPU-f07a0798-7ddc-ec7c-a27c-d47124bbfe81'}


def choices(p=None, scores=None):
    p=plan() if p is None else p
    normal=normal_scores(p) if scores is None else scores
    extra=additional_scores(p,normal)
    frozen=freeze_case_plan(p,normal,extra)
    return p,frozen,{**normal,**extra}


def metric_objects(scores):
    return {name:dict(combined47=dict(row['metrics'],IDSW=20)) for name,row in scores.items()}


def test_all78_cases_keep_Dev_frozen_roles_and_negative_ablation_without_novelty_count():
    p,frozen,scores=choices()
    roles=analysis_roles(p,frozen,scores)
    assert len(frozen['cases'])==78 and not roles['Test_input']
    Dev=metric_objects(scores);Test=copy.deepcopy(Dev)
    effect=scientific_effects(roles,Dev,Test)
    assert effect['necessary_fused_memory_numeric_conditions']
    assert not effect['reliability_floor']['necessary_positive_HOTA_both_Dev_Test']
    assert effect['independently_validated_innovations']==0
    frozen_primary=roles['roles']['primary_IR']
    unrelated=next(c['id'] for c in frozen['cases'] if c['visual_key']!=p['role_visual_keys']['primary_IR'])
    Test[unrelated]['combined47']['HOTA']=99.
    changed=scientific_effects(roles,Dev,Test)
    assert roles['roles']['primary_IR']==frozen_primary and changed['independently_validated_innovations']==0


def test_RGB_memory_winner_does_not_become_a_fused_memory_contribution():
    p=plan();normal=normal_scores(p)
    rgb=next(c for c in p['normal_cases'] if c['visual_key']==p['role_visual_keys']['primary_IR'] and
        c['profile']=='rgb_attention_proxy' and c['appearance_weight']==.1)
    normal[rgb['id']]['metrics']={k:45. for k in ['HOTA','AssA','IDF1']}
    p,frozen,scores=choices(p,normal);roles=analysis_roles(p,frozen,scores)
    effect=scientific_effects(roles,metric_objects(scores),metric_objects(scores))
    assert roles['roles']['primary_IR']==rgb['id'] and not roles['primary_uses_fused_memory']
    assert not effect['necessary_fused_memory_numeric_conditions'] and effect['independently_validated_innovations']==0


@pytest.mark.parametrize('change',['Test_scores','tampered_choice'])
def test_Test_scores_or_posthoc_choice_cannot_enter_analysis_roles(change):
    p,frozen,scores=choices()
    if change=='Test_scores':
        for value in scores.values():value['split']='test'
    else:frozen['selection']['primary_uses_fused_memory']=False
    with pytest.raises(ValueError):analysis_roles(p,frozen,scores)


def table(path,official,split='test',value=0.):
    schema,columns=official;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for sequence in [*LENGTHS[split],'COMBINED']:
            row={name:0 for name in columns}
            row.update(seq=sequence,CLR_Frames=sum(LENGTHS[split].values()) if sequence=='COMBINED' else LENGTHS[split][sequence])
            for name in columns:
                if name.startswith('HOTA___') or name.startswith('AssA___') or name=='IDF1':row[name]=value/100
            writer.writerow(row)
    return path


def receipt_fixture(root,official):
    workflow,specification,cases,_,directory=source_fixture(root)
    spec=read(specification);spec['parent_Dev_evidence_sha256']='b'*64;write_json(specification,spec)
    context=dict(spec=spec,specification=str(specification),registration_sha256=sha(workflow.base/'pipeline_registered.json'),
        source_versions=workflow.validate_specification(specification)[1]['source_hashes'],GT_hashes={
            str(root/'ARTIFICIAL_GT/test'/seq/'gt/gt.txt'):'e'*64 for seq in LENGTHS['test']},
        source_RGB_sha256='a'*64,GPU_engineering_sha256='c'*64,GPU_uuids=UUIDS)
    routes,_=visual_routes(cases)
    information={key:dict(route,fusion_parameters=0,checkpoint_tensor_load=False) for key,route in routes.items()}
    for seq in LENGTHS['test']:
        p=directory/'sequence_manifests'/(seq+'.json');v=read(p);v['provenance'].update(
            specification_sha256=sha(specification),source_RGB_sha256=context['source_RGB_sha256'],
            parent_Dev_evidence_sha256=spec['parent_Dev_evidence_sha256'],actual_GPU_engineering_sha256=context['GPU_engineering_sha256'],
            strict_CPU_tensor_load_before_CUDA=True,original_RGB_parameters_frozen=True,GT_input=False,future_input=False,
            GPU_uuid=UUIDS['4'],actual_producer_identity=dict(pid=123456,start_ticks=1,command_sha256='f'*64),model_information=information)
        write_json(p,v)
    case=cases[1]
    manifest=verify_observations(specification,case,directory,context['registration_sha256'],context['source_versions'],spec['image_source_anchors'])
    path=root/'evaluation/test_frozen'/case['id']/'metrics.json'
    raw=table(path.parent/'trackeval_output/pedestrian_detailed.csv',official)
    parsed=official_csv47(raw,official[0],'test')
    headline={s['metric']:round(parsed['combined47'][s['metric']],3) for s in official[0] if s['in_official_headline']}
    value=dict(status='COMPLETE_F4_OFFICIAL_METRICS',case=case,manifest=manifest,metrics39=headline,
        official_schema=official[0],official_family_field_counts=dict(HOTA=15,CLEAR=22,Identity=6,Count=4),**parsed)
    write_json(path,value)
    driver=dict(status='COMPLETE_REGISTERED_F4_METRIC_DRIVER_VALIDATION',metrics_sha256=sha(path),
        specification_sha256=sha(specification),parent_Dev_evidence_sha256=spec['parent_Dev_evidence_sha256'],
        pipeline_registration_sha256=context['registration_sha256'],protocol_GT_file_sha256=context['GT_hashes'],
        actual_producer_specification_revalidated=True,GT_inference_input=False,no_Test_accuracy_input_for_Dev_selection=True)
    write_json(path.parent/'driver_validation.json',driver)
    return workflow,path,case,context,value


def test_checked_full7_metric_receipts_and_raw_CSV_publication_preserve_all_fields(tmp_path,official):
    _,path,case,context,_=receipt_fixture(tmp_path,official)
    value,reference=checked_metric(path,case,context,official[0])
    result=publish_case(tmp_path/'public','test',case,value,reference,official[0])
    raw=Path(result['official_original276']['path'])
    assert raw.read_bytes()==Path(value['official_CSV']).read_bytes() and b'\r\n' in raw.read_bytes()
    with Path(result['metrics47']['path']).open() as stream:
        reader=csv.DictReader(stream);rows=list(reader);fields=reader.fieldnames
    assert len(fields)==50 and len(rows)==8 and rows[-1]['seq']=='COMBINED'
    assert [r['seq'] for r in rows[:-1]]==list(LENGTHS['test'])
    assert float(rows[-1]['CLR_Frames'])==15985
    assert publish_case(tmp_path/'public','test',case,value,reference,official[0])==result
    raw.write_bytes(b'CORRUPT')
    with pytest.raises(ValueError,match='Preserve previously'):
        publish_case(tmp_path/'public','test',case,value,reference,official[0])


@pytest.mark.parametrize('change',['GT_receipt','UUID','source_RGB','parent','GPU_proof','driver_hash','non_AUC_nan','frames'])
def test_forged_or_partial_result_sources_are_rejected(tmp_path,official,change):
    _,path,case,context,value=receipt_fixture(tmp_path,official)
    driver=path.parent/'driver_validation.json'
    if change in ['GT_receipt','driver_hash']:
        v=read(driver)
        if change=='GT_receipt':v['protocol_GT_file_sha256']={}
        else:v['metrics_sha256']='0'*64
        write_json(driver,v)
    elif change=='non_AUC_nan':
        p=Path(value['official_CSV'])
        with p.open() as stream:r=csv.DictReader(stream);rows=list(r);columns=r.fieldnames
        rows[0][next(k for k in columns if k.startswith('HOTA___') and not k.endswith('AUC'))]='nan'
        with p.open('w',newline='') as stream:w=csv.DictWriter(stream,fieldnames=columns);w.writeheader();w.writerows(rows)
    else:
        p=Path(context['spec']['stream_directory'])/'sequence_manifests'/(next(iter(LENGTHS['test']))+'.json');v=read(p)
        if change=='frames':v['frames']=32
        else:
            name={'UUID':'GPU_uuid','source_RGB':'source_RGB_sha256','parent':'parent_Dev_evidence_sha256','GPU_proof':'actual_GPU_engineering_sha256'}[change]
            v['provenance'][name]='0'*64
        write_json(p,v)
    with pytest.raises(ValueError):checked_metric(path,case,context,official[0])


def cost_fixture(root,official):
    workflow,path,case,context,value=receipt_fixture(root,official)
    checked_metric(path,case,context,official[0])
    refs,evidence=runtime_references(workflow,context['specification'],case)
    measurements=[]
    for seq in LENGTHS['test']:
        common=dict(sequence=seq,prefix_frames=32,packet_sha256=refs[seq]['prefix_packet_sha256'][case['id']],
            literal_MOT_sha256=refs[seq]['prefix_literal_MOT_sha256'][case['id']],
            sampled_retained_state_logical_bytes_max=dict(total=0),IR_byte_frames_hashed=32,IR_frames_decoded=0,
            source_image_bytes_exact=True,complete_production_packets_and_MOT_exact=True,simultaneous_conditions=1)
        measurements.append(dict(verification=dict(common,warmup_frames=0,measured_frames=0,durations_seconds=[]),
            timing=dict(common,warmup_frames=8,measured_frames=24,durations_seconds=[.01]*24,warmup_to_timing_state_preserved=True),
            exact_source_and_output_checked_before_and_during_timing=True))
    routes,_=visual_routes([case])
    row=dict(status='COMPLETE_ACTUAL_F4_INDIVIDUAL_PREFIX_COST',case=case,specification_sha256=sha(context['specification']),
        pipeline_registration_sha256=context['registration_sha256'],source_evidence=evidence,measurements=measurements,
        **aggregate_cost(measurements),physical_gpu='4',GPU_uuid=UUIDS['4'],independent_input_batch_size=1,simultaneous_configurations=1,
        actual_prefix_matches_full_production_before_and_during_timing=True,no_GT_or_Test_accuracy_input=True,
        learned_memory_parameters=0,additional_fit_or_seed_work=False,feature_or_detection_disk_cache=False,
        unused_IR_bytes_hashed_in_all_controls_for_equal_source_provenance=True,
        original_RGB_parameters=98995378,fusion_parameters=0,total_visual_parameters=98995378,
        model_information={key:dict(route,fusion_parameters=0,checkpoint_tensor_load=False) for key,route in routes.items()},
        peak_allocated_GPU_GiB=1.,peak_reserved_GPU_GiB=1.,process_lifetime_peak_host_RSS_GiB=2.)
    return row,case,context,value,evidence,refs


def test_all7_independent_costs_are_recomputed_from_168_latencies(tmp_path,official):
    row,case,context,value,evidence,refs=cost_fixture(tmp_path,official)
    assert checked_cost(row,case,context['specification'],context['registration_sha256'],value,evidence,refs,UUIDS)==row
    assert row['measured_frames']==168 and row['FPS']==pytest.approx(100.)


@pytest.mark.parametrize('change',['bundled','warmup_reset','missing_sequence','FPS','source_packet','wrong_UUID','host_memory','wrong_model'])
def test_invalid_runtime_or_different_sources_cannot_be_published(tmp_path,official,change):
    row,case,context,value,evidence,refs=cost_fixture(tmp_path,official)
    if change=='bundled':row['simultaneous_configurations']=48
    elif change=='warmup_reset':row['measurements'][0]['timing']['warmup_to_timing_state_preserved']=False
    elif change=='missing_sequence':row['measurements'].pop()
    elif change=='FPS':row['FPS']+=1
    elif change=='source_packet':row['measurements'][0]['timing']['packet_sha256']='0'*64
    elif change=='wrong_UUID':row['GPU_uuid']=UUIDS['5']
    elif change=='host_memory':row['process_lifetime_peak_host_RSS_GiB']=8.01
    else:next(iter(row['model_information'].values()))['visual_key']='wrong'
    with pytest.raises(ValueError):checked_cost(row,case,context['specification'],context['registration_sha256'],value,evidence,refs,UUIDS)


def test_every_frozen_case_and_sequence_needs_a_completed_exited_worker():
    p,frozen,_=choices();specs={name:dict(split='test' if name=='test_frozen' else 'dev',cases=cases) for name,cases in
        [('dev_normal',p['normal_cases']),('dev_additional',frozen['cases'][48:]),('test_frozen',frozen['cases'])]}
    paths={name:Path('/ARTIFICIAL/specifications')/(name+'.json') for name in specs}
    jobs=execution_jobs(paths,specs,'/ARTIFICIAL/python')
    assert sum(map(len,jobs.values()))==169 and len(jobs['TEST_EVALUATE'])==78
    phases={phase:{name:dict(command=command,status='COMPLETE',exit_code=0,pid=123,start_ticks=1) for name,command in entries.items()}
        for phase,entries in jobs.items()}
    state=dict(status=EXECUTION_FINISHED,fixed_seed=0,physical_gpus=['4','5'],maximum_GPU_workers=2,worker_python='/ARTIFICIAL/python',phases=phases)
    assert validate_execution_state(state,paths,specs,lambda *a,**k:True)==jobs
    with pytest.raises(ValueError):validate_execution_state(state,paths,specs,lambda *a,**k:False)
    state['phases']['TEST_EVALUATE'].pop(next(iter(state['phases']['TEST_EVALUATE'])))
    with pytest.raises(ValueError,match='Every frozen'):validate_execution_state(state,paths,specs,lambda *a,**k:True)


def test_publisher_keeps_all78_negative_and_positive_case_rows_under_size_cap(tmp_path,official):
    p,frozen,scores=choices();raw=table(tmp_path/'actual_artificial_original.csv',official,value=0.)
    parsed=official_csv47(raw,official[0],'test');summary={s['metric']:round(parsed['combined47'][s['metric']],3) for s in official[0] if s['in_official_headline']}
    value=dict(parsed,metrics39=summary,manifest=dict(frames=15985,image_source_anchors={},coverage={},source_versions={}))
    outputs=[publish_case(tmp_path/'public','test',case,value,dict(artificial=True),official[0]) for case in frozen['cases']]
    assert len(outputs)==78 and {o['id'] for o in outputs}=={c['id'] for c in frozen['cases']}
    assert all(len(list(csv.DictReader(Path(o['metrics47']['path']).open())))==8 for o in outputs)
    with pytest.raises(ValueError,match='below1MiB'):small_bytes(tmp_path/'too_large.json',b'x'*2**20)


def test_full_missing_equivalence_requires_CSV_and_every_Test_packet():
    base=dict(official_CSV_sha256='a'*64,manifest=dict(coverage={s:dict(track_sha256='b'*64,detection_packet_sha256='c'*64) for s in LENGTHS['test']}))
    other=copy.deepcopy(base);exact_full_behavior(base,other,'test')
    other['manifest']['coverage']['SN-014']['detection_packet_sha256']='d'*64
    with pytest.raises(ValueError,match='full per-sequence'):exact_full_behavior(base,other,'test')
