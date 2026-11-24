"""F4 result receipts, Dev-derived analysis roles and bounded publication files.

Contexts are caller-owned. This component cannot qualify a parent/model, start
CUDA, evaluate GT, choose from Test, or establish a scientific innovation.
"""
import csv
import io
import json
import math
from pathlib import Path
import re

from tools.fusion_memory_design import best_case, make_case, freeze_case_plan
from tools.fusion_memory_evaluation import official_csv47, read, sha, verify_observations
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_online import visual_routes
from tools.fusion_memory_parent_io import require
from tools.fusion_memory_runtime import aggregate_cost
from tools.fusion_memory_runtime_sources import unchanged_runtime_sources


def driver_receipt(path, value, context):
    receipt_path = Path(path).parent/'driver_validation.json'
    receipt = read(receipt_path)
    require(receipt['status'] == 'COMPLETE_REGISTERED_F4_METRIC_DRIVER_VALIDATION' and
            receipt['metrics_sha256'] == sha(path) and receipt['specification_sha256'] == sha(context['specification']) and
            receipt['parent_Dev_evidence_sha256'] == context['spec']['parent_Dev_evidence_sha256'] and
            receipt['pipeline_registration_sha256'] == context['registration_sha256'] and
            receipt['protocol_GT_file_sha256'] == context['GT_hashes'] and
            receipt['actual_producer_specification_revalidated'] is True and receipt['GT_inference_input'] is False and
            receipt['no_Test_accuracy_input_for_Dev_selection'] is True, 'Registered metric driver/GT receipt differs')
    require(value['manifest']['specification_sha256'] == receipt['specification_sha256'], 'Driver and metric specifications differ')
    return dict(path=str(receipt_path), sha256=sha(receipt_path))


def checked_metric(path, case, context, schema):
    path = Path(path)
    value = read(path)
    require(value['status'] == 'COMPLETE_F4_OFFICIAL_METRICS' and value['case'] == case and case in context['spec']['cases'],
            'Exact frozen complete F4 metric case required')
    actual = verify_observations(context['specification'], case, context['spec']['stream_directory'],
        context['registration_sha256'], context['source_versions'], context['spec']['image_source_anchors'])
    require(value['manifest'] == actual, 'Official metric observations changed')
    csv_path = path.parent/'trackeval_output/pedestrian_detailed.csv'
    require(Path(value['official_CSV']).resolve() == csv_path.resolve(), 'Original own-case CSV required')
    parsed = official_csv47(csv_path, schema, actual['split'], value['metrics39'])
    require(all(value[k] == parsed[k] for k in parsed), 'Original CSV/all47 values changed')
    require(value['official_schema'] == schema and value['official_family_field_counts'] == {
        name:sum(s['family']==name for s in schema) for name in ['HOTA','CLEAR','Identity','Count']},
        'Original official metric schema/families changed')
    routes, case_routes = visual_routes(context['spec']['cases'])
    route = routes[case_routes[case['id']]]
    for source in actual['sources'].values():
        producer = read(source['path'])
        provenance = producer['provenance']
        require(provenance['source_RGB_sha256'] == context['source_RGB_sha256'] and
                provenance['parent_Dev_evidence_sha256'] == context['spec']['parent_Dev_evidence_sha256'] and
                provenance['actual_GPU_engineering_sha256'] == context['GPU_engineering_sha256'] and
                provenance['strict_CPU_tensor_load_before_CUDA'] is True and provenance['original_RGB_parameters_frozen'] is True and
                provenance['GT_input'] is False and provenance['future_input'] is False,
                'Producer source/parent/GPU engineering provenance differs')
        physical = provenance['physical_gpu']
        require(str(provenance['GPU_uuid']).lower().removeprefix('gpu-') ==
                context['GPU_uuids'][physical].lower().removeprefix('gpu-'), 'Producer physical GPU UUID differs')
        identity = provenance['actual_producer_identity']
        require(identity['pid'] > 0 and identity['start_ticks'] > 0 and
                re.fullmatch('[0-9a-f]{64}', identity['command_sha256']) is not None,
                'Actual producer process identity missing')
        loaded = provenance['model_information'][route['id']]
        require(all(loaded[k] == v for k, v in route.items()) and
                loaded['checkpoint_tensor_load'] is (not case['visual']['baseline']), 'Producer loaded a different visual route')
    receipt = driver_receipt(path, value, context)
    return value, dict(path=str(path), sha256=sha(path), official_CSV_sha256=parsed['official_CSV_sha256'], driver_validation=receipt)


def exact_full_behavior(a, b, split):
    require(a['official_CSV_sha256'] == b['official_CSV_sha256'] and all(
        all(a['manifest']['coverage'][seq][key] == b['manifest']['coverage'][seq][key]
            for key in ['track_sha256', 'detection_packet_sha256']) for seq in LENGTHS[split]),
        'Missing/zero behavior must preserve all276 CSV columns and full per-sequence packets/MOT')


def analysis_roles(plan, frozen, Dev_scores):
    """All comparator choices here use Dev only; Test values are not accepted."""
    cases = {c['id']: c for c in frozen['cases']}
    require(set(Dev_scores) == set(cases) and all(r['split'] == 'dev' and r['frames'] == 4834 for r in Dev_scores.values()),
            'Analysis role choices require all frozen complete Dev scores only')
    normal = {c['id'] for c in plan['normal_cases']}
    rebuilt = freeze_case_plan(plan,{k:v for k,v in Dev_scores.items() if k in normal},
                              {k:v for k,v in Dev_scores.items() if k not in normal})
    require(all(frozen[k]==v for k,v in rebuilt.items() if k!='status'),'Analysis plan differs from actual complete Dev scores')
    selected = rebuilt['selection']
    primary, fused = selected['primary_IR'], selected['best_fused_IR']
    key = plan['role_visual_keys']['primary_IR']
    normal = [c for c in plan['normal_cases'] if c['visual_key'] == key]
    byte = next(c for c in normal if c['profile'] == 'byte')
    uniform_fused = best_case([c for c in normal if c['profile'] in ['byte', 'fused_uniform']], Dev_scores)
    roles = dict(primary_IR=primary['id'], best_fused_IR=fused['id'], own_visual_byte=byte['id'],
        own_fused_uniform_Dev_best=uniform_fused['id'],
        own_visual_RGB_uniform_best=selected['own_visual_RGB_uniform_best']['id'],
        own_visual_same_q_RGB_best=selected['own_visual_same_q_RGB_best']['id'],
        RGB_uniform_at_best_fused_weight=selected['RGB_uniform_at_best_fused_weight']['id'],
        RGB_same_q_at_best_fused_weight=selected['RGB_same_q_at_best_fused_weight']['id'],
        RGB_uniform_at_primary_IR_weight=selected['RGB_uniform_at_primary_IR_weight']['id'],
        RGB_same_q_at_primary_IR_weight=selected['RGB_same_q_at_primary_IR_weight']['id'],
        **frozen['strong_control_roles_including_registered_floor_removals'])
    for label, source in [('primary', primary), ('best_fused', fused)]:
        for intervention in ['IR_unavailable', 'IR_image_spatial_roll']:
            case = make_case(source['visual'], source['profile'], source['appearance_weight'], intervention=intervention)
            roles[label+'_'+intervention] = case['id']
    if fused['appearance_weight']:
        roles['fused_uniform_at_best_fused_weight'] = make_case(fused['visual'], 'fused_uniform', fused['appearance_weight'])['id']
    else:
        roles['fused_uniform_at_best_fused_weight'] = byte['id']
    policies = frozen['reliability_floor_removal_policies']
    if policies is not None:
        roles.update(floor_removal_own_Dev_best=policies['own_Dev_best'],
                     floor_removal_paired_fused_weight=policies['paired_best_fused_weight'])
        for profile, label in [('rgb_uniform', 'uniform'),
            ('rgb_attention_proxy' if fused['quality_policy'] == 'attention_proxy' else 'rgb_uniform', 'same_q')]:
            choices = [byte]+[c for c in cases.values() if c['visual_key'] == key and c['intervention'] == 'normal' and
                c['minimum_reliability'] == 0 and c['profile'] == profile]
            roles['floor_removal_RGB_'+label+'_own_Dev_best'] = best_case(choices, Dev_scores)['id']
            roles['floor_removal_RGB_'+label+'_paired_fused_weight'] = make_case(fused['visual'], profile,
                fused['appearance_weight'], floor=0.)['id']
    require(all(name in cases for name in roles.values()), 'Every analysis comparator must already belong to the frozen case set')
    return dict(roles=roles, primary_uses_fused_memory=selected['primary_uses_fused_memory'],
        best_fused_is_nonzero=fused['appearance_weight'] > 0, selected_fused_profile=fused['profile'],
        reliability_floor_removal_triggered=policies is not None,
        comparator_selection_input='Complete original frozen Dev only', Test_input=False)


def scientific_effects(role_plan, Dev, Test):
    """Report necessary numeric effects, never auto-count novelty or select Test."""
    roles = role_plan['roles']
    require(all(n in Dev and n in Test for n in roles.values()), 'All frozen role metrics must be retained')
    def comparison(left, right):
        return {split: {metric: data[roles[left]]['combined47'][metric]-data[roles[right]]['combined47'][metric]
            for metric in ['HOTA', 'AssA', 'IDF1', 'IDSW']} for split, data in [('Dev', Dev), ('Test', Test)]}
    cue_controls = ['matched_RGB', 'matched_gray', 'global_RGB', 'global_gray', 'fresh_RGB',
                    'primary_IR_unavailable', 'primary_IR_image_spatial_roll']
    cues = {name: comparison('primary_IR', name) for name in cue_controls}
    memory_controls = ['matched_RGB', 'matched_gray', 'global_RGB', 'global_gray', 'fresh_RGB',
        'own_visual_byte', 'own_visual_RGB_uniform_best', 'own_visual_same_q_RGB_best',
        'RGB_uniform_at_best_fused_weight', 'RGB_same_q_at_best_fused_weight',
        'best_fused_IR_unavailable', 'best_fused_IR_image_spatial_roll']
    memory = {name: comparison('best_fused_IR', name) for name in memory_controls}
    positive = lambda rows: all(v[split]['HOTA'] > 0 for v in rows.values() for split in ['Dev', 'Test'])
    quality = {name: comparison('best_fused_IR', name) for name in
        ['own_fused_uniform_Dev_best', 'fused_uniform_at_best_fused_weight']}
    floor = None
    if role_plan['reliability_floor_removal_triggered']:
        effects = {name: comparison('best_fused_IR', name) for name in
            ['floor_removal_own_Dev_best', 'floor_removal_paired_fused_weight']}
        interactions = {split: {metric: effects['floor_removal_paired_fused_weight'][split][metric] -
            (data[roles['RGB_same_q_at_best_fused_weight']]['combined47'][metric] -
             data[roles['floor_removal_RGB_same_q_paired_fused_weight']]['combined47'][metric])
            for metric in ['HOTA', 'AssA', 'IDF1', 'IDSW']} for split, data in [('Dev', Dev), ('Test', Test)]}
        floor = dict(effects=effects, fused_minus_same_q_RGB_difference_in_differences=interactions,
            necessary_positive_HOTA_both_Dev_Test=positive(effects) and all(interactions[s]['HOTA'] > 0 for s in ['Dev', 'Test']))
    return dict(primary_cue_controls=cues, fused_memory_controls=memory, proxy_quality_controls=quality,
        reliability_floor=floor,
        necessary_cue_numeric_conditions=positive(cues),
        necessary_fused_memory_numeric_conditions=role_plan['primary_uses_fused_memory'] and
            role_plan['best_fused_is_nonzero'] and positive(memory),
        secondary_best_fused_necessary_numeric_conditions=role_plan['best_fused_is_nonzero'] and positive(memory),
        necessary_attention_proxy_numeric_conditions=role_plan['selected_fused_profile'] == 'fused_attention_proxy' and
            role_plan['primary_uses_fused_memory'] and positive(memory) and positive(quality),
        primary_uses_fused_memory=role_plan['primary_uses_fused_memory'],
        attention_is_calibrated_reliability=False, independently_validated_innovations=0,
        scope='Full-precision official combined percentage-point differences; IDSW signed raw counts. Necessary numeric controls are not novelty proof.')


def checked_cost(row, case, specification, registration_sha256, metric, source_evidence, references, GPU_uuids):
    require(row['status'] == 'COMPLETE_ACTUAL_F4_INDIVIDUAL_PREFIX_COST' and row['case'] == case and
            row['specification_sha256'] == sha(specification) and row['pipeline_registration_sha256'] == registration_sha256 and
            row['source_evidence'] == source_evidence and
            source_evidence['complete_observation_manifest'] == metric['manifest'], 'Independent cost belongs to a different case/source')
    unchanged_runtime_sources(source_evidence)
    measurements = row['measurements']
    require([r['timing']['sequence'] for r in measurements] == list(LENGTHS['test']), 'Every original Test prefix needs its independent cost')
    needs_IR = case['visual']['cue'] == 'ir' and case['intervention'] != 'IR_unavailable' and not case['visual']['baseline']
    for result in measurements:
        timing, verification = result['timing'], result['verification']
        sequence = timing['sequence']
        require(verification['sequence'] == sequence and result['exact_source_and_output_checked_before_and_during_timing'] is True and
                timing['prefix_frames'] == verification['prefix_frames'] == 32 and timing['warmup_frames'] == 8 and
                timing['measured_frames'] == len(timing['durations_seconds']) == 24 and verification['measured_frames'] == 0 and
                verification['durations_seconds'] == [] and verification['warmup_frames'] == 0 and
                timing['warmup_to_timing_state_preserved'] is True and
                timing['simultaneous_conditions'] == verification['simultaneous_conditions'] == 1,
                'Individual32/8/24 timing coverage/state differs')
        for value in [verification, timing]:
            require(value['source_image_bytes_exact'] is True and value['complete_production_packets_and_MOT_exact'] is True and
                    value['IR_byte_frames_hashed'] == 32 and value['IR_frames_decoded'] == (32 if needs_IR else 0) and
                    value['packet_sha256'] == references[sequence]['prefix_packet_sha256'][case['id']] and
                    value['literal_MOT_sha256'] == references[sequence]['prefix_literal_MOT_sha256'][case['id']],
                    'Timed packet/MOT/source prefix differs from complete production')
    computed = aggregate_cost(measurements)
    require(computed['measured_frames'] == row['measured_frames'] == 168 and all(
        math.isclose(row[k], computed[k], rel_tol=1e-12, abs_tol=1e-12) for k in
        ['seconds', 'FPS', 'mean_latency_ms', 'median_latency_ms']) and
        row['sampled_retained_state_logical_bytes_max'] == computed['sampled_retained_state_logical_bytes_max'],
        'Published cost does not equal actual per-frame measurements')
    require(row['physical_gpu'] in ['4', '5'] and row['independent_input_batch_size'] == row['simultaneous_configurations'] == 1 and
            row['actual_prefix_matches_full_production_before_and_during_timing'] is True and row['no_GT_or_Test_accuracy_input'] is True and
            row['learned_memory_parameters'] == 0 and row['additional_fit_or_seed_work'] is False and
            row['feature_or_detection_disk_cache'] is False and row['unused_IR_bytes_hashed_in_all_controls_for_equal_source_provenance'] is True,
            'Cost scope is not the registered independent computation')
    require(str(row['GPU_uuid']).lower().removeprefix('gpu-') == GPU_uuids[row['physical_gpu']].lower().removeprefix('gpu-'),
            'Independent cost physical GPU UUID differs')
    require(all(isinstance(row[k], int) and row[k] >= 0 for k in ['original_RGB_parameters', 'fusion_parameters', 'total_visual_parameters']) and
            row['original_RGB_parameters'] > 0 and row['total_visual_parameters'] == row['original_RGB_parameters']+row['fusion_parameters'] and
            (row['fusion_parameters'] == 0) is case['visual']['baseline'], 'Actual visual parameter counts differ')
    routes,_=visual_routes([case])
    require(set(row['model_information'])==set(routes) and all(all(row['model_information'][key][k]==v for k,v in route.items()) and
        row['model_information'][key]['fusion_parameters']==row['fusion_parameters'] and
        row['model_information'][key]['checkpoint_tensor_load'] is (not case['visual']['baseline'])
        for key,route in routes.items()),'Independent runtime loaded a different visual route')
    for key in ['peak_allocated_GPU_GiB', 'peak_reserved_GPU_GiB', 'process_lifetime_peak_host_RSS_GiB']:
        require(math.isfinite(row[key]) and row[key] > 0, 'Actual resource footprint missing')
    require(row['process_lifetime_peak_host_RSS_GiB'] <= 8, 'Cost worker exceeded registered host bound')
    return row


def small_bytes(path, data):
    path = Path(path)
    require(len(data) < 2**20, 'Each public result file must stay below1MiB')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, 'Preserve previously published result bytes')
    else:
        with path.open('xb') as stream: stream.write(data)
    return dict(path=str(path), sha256=sha(path), bytes=len(data))


def small_json(path, value):
    return small_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())


def publish_case(destination, split, case, value, source_reference, schema):
    destination = Path(destination)/split/case['id']
    raw = Path(value['official_CSV'])
    require(sha(raw) == value['official_CSV_sha256'], 'Original CSV changed before publication')
    original = small_bytes(destination/'official_original276.csv', raw.read_bytes())
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=['candidate','split','seq']+[s['metric'] for s in schema], lineterminator='\n')
    writer.writeheader()
    writer.writerows(dict(candidate=case['id'], **row) for row in value['per_sequence47'])
    metrics = small_bytes(destination/'metrics47.csv', buffer.getvalue().encode())
    summary = small_json(destination/'summary.json', dict(case=case, metrics39=value['metrics39'],
        combined47=value['combined47'], selection_metrics=value['selection_metrics'],
        metric_source=source_reference, official_original276=original, metrics47=metrics,
        complete_frames=value['manifest']['frames'], full_source_image_anchors=value['manifest']['image_source_anchors'],
        track_and_packet_coverage=value['manifest']['coverage'], original_source_versions=value['manifest']['source_versions']))
    return dict(id=case['id'], split=split, summary=summary, official_original276=original, metrics47=metrics)
