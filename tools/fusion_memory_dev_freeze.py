"""Freeze F4 choices from complete original Dev metric files; no Test input.

The caller must separately establish actual visual-parent eligibility, complete
pipeline registration and GPU execution. These file checks do not authorize Test.
"""
from pathlib import Path

from tools.fusion_memory_design import extra_cases, freeze_case_plan, select_memory
from tools.fusion_memory_evaluation import official_csv47, read, sha, verify_observations
from tools.fusion_memory_observation import write_json
from tools.fusion_memory_parent_io import require


def dev_path(path):
    path = Path(path).resolve()
    require(not any(part.lower() in ['test', 'test_frozen'] for part in path.parts),
            'Test accuracy paths cannot enter F4 Dev selection')
    return path


def reference_for_byte(plan, case):
    if case['intervention'] == 'normal':
        return case['visual']
    parent = next(r for r in plan['parent']['details'] if r['family'] == plan['parent']['chosen_family'])
    return parent['controls'][case['intervention']]


def exact_behavior(a, b):
    require(a['official_CSV_sha256'] == b['official_CSV_sha256'], 'Exact behavior requires the whole original276-column CSV')
    for seq in ['BT-005', 'SN-013']:
        require(all(a['manifest']['coverage'][seq][key] == b['manifest']['coverage'][seq][key]
            for key in ['track_sha256', 'detection_packet_sha256']), 'Exact behavior requires every Dev track/packet byte')


def load_dev_files(plan, cases, files, schema):
    require(set(files) == {c['id'] for c in cases}, 'Every registered Dev case must be retained, including negatives')
    scores, values, references = {}, {}, {}
    for case in cases:
        name, path = case['id'], dev_path(files[case['id']])
        value = read(path)
        require(value['status'] == 'COMPLETE_F4_OFFICIAL_METRICS' and value['case'] == case, 'Exact complete F4 case metric file required')
        manifest = value['manifest']
        require(manifest['split'] == 'dev' and manifest['frames'] == 4834 and manifest['sequences'] == ['BT-005', 'SN-013'],
                'Only complete original Dev can choose F4')
        specification = dev_path(manifest['specification'])
        spec = read(specification)
        for reference in [*manifest['sources'].values(), *manifest['image_source_anchors'].values()]:
            dev_path(reference['path'])
        for stream in manifest['coverage'].values():
            dev_path(stream['track_path'])
        actual = verify_observations(specification, case, dev_path(spec['stream_directory']),
            manifest['pipeline_registration_sha256'], manifest['source_versions'], manifest['image_source_anchors'])
        require(actual == manifest, 'Dev observation receipts changed since evaluation')
        csv_path = dev_path(value['official_CSV'])
        parsed = official_csv47(csv_path, schema, 'dev', value['metrics39'])
        require(all(parsed[k] == value[k] for k in ['combined47', 'per_sequence47', 'selection_metrics', 'official_CSV_sha256']),
                'Dev official values/CSV changed')
        if case['appearance_weight'] == 0:
            original = reference_for_byte(plan, case)
            original_csv = dev_path(original['official_CSV'])
            require(sha(original_csv) == original['official_CSV_sha256'] == value['official_CSV_sha256'],
                    'ByteTrack case does not reproduce original visual full Dev CSV')
            for seq, source in original['source_receipts'].items():
                require(all(source[k] == manifest['coverage'][seq][k] for k in ['track_sha256', 'detection_packet_sha256']),
                        'ByteTrack case does not reproduce original visual packets/MOT')
        scores[name] = dict(id=name, split='dev', frames=4834, sequences=['BT-005', 'SN-013'], seed=0,
            intervention=case['intervention'], metrics=parsed['selection_metrics'])
        values[name], references[name] = value, dict(path=str(path), sha256=sha(path), official_CSV_sha256=parsed['official_CSV_sha256'])
    return scores, values, references


def normal_dev_freeze(plan, files, schema, output):
    require(not Path(output).exists(), 'Preserve the originally frozen normal Dev choice')
    scores, _, references = load_dev_files(plan, plan['normal_cases'], files, schema)
    selection = select_memory(plan, scores)
    result = dict(status='FROZEN_F4_NORMAL_DEV_FILES_BEFORE_INTERVENTIONS',
        normal_source_metrics=references, normal_scores=scores, selection=selection,
        additional_cases=extra_cases(plan, selection), Test_accuracy_read=False,
        actual_parent_eligibility_must_be_established_separately=True, formal_launch_ready=False,
        goal_met=False, supported_innovations=0)
    write_json(output, result, exclusive=True)
    return result


def final_dev_freeze(plan, normal_choice, normal_files, additional_files, schema, output):
    require(not Path(output).exists(), 'Preserve the original complete Dev case freeze')
    normal_choice = dev_path(normal_choice)
    original = read(normal_choice)
    normal, values, references = load_dev_files(plan, plan['normal_cases'], normal_files, schema)
    selection = select_memory(plan, normal)
    additions = extra_cases(plan, selection)
    require(original['status'] == 'FROZEN_F4_NORMAL_DEV_FILES_BEFORE_INTERVENTIONS' and
            original['normal_scores'] == normal and original['normal_source_metrics'] == references and
            original['selection'] == selection and original['additional_cases'] == additions and original['Test_accuracy_read'] is False,
            'Original normal choice changed after intervention production')
    additional, extra_values, extra_references = load_dev_files(plan, additions, additional_files, schema)
    for value in extra_values.values():
        spec = read(dev_path(value['manifest']['specification']))
        require(spec['normal_Dev_choice_sha256'] == sha(normal_choice), 'Interventions did not bind the original normal Dev choice')
    frozen = freeze_case_plan(plan, normal, additional)
    all_values = {**values, **extra_values}
    fresh = plan['role_visual_keys']['fresh_RGB']
    for case in additions:
        if case['intervention'] != 'IR_unavailable':
            continue
        control = next(c for c in frozen['cases'] if c['visual_key'] == fresh and c['intervention'] == 'normal' and
            c['appearance_weight'] == case['appearance_weight'] and (c['profile'] == 'byte' or
                (c['profile'] == 'rgb_uniform' and c['minimum_reliability'] == case['minimum_reliability'])))
        exact_behavior(all_values[case['id']], all_values[control['id']])
    result = dict(frozen, status='FROZEN_F4_ALL_COMPLETE_DEV_METRIC_FILES_PENDING_EXECUTION_GATES',
        original_normal_choice=str(normal_choice), original_normal_choice_sha256=sha(normal_choice),
        normal_source_metrics=references, additional_source_metrics=extra_references,
        full_original_CSV_MOT_and_packet_missing_equivalence=True, Test_accuracy_read=False,
        actual_parent_eligibility_must_be_established_separately=True, formal_launch_ready=False)
    write_json(output, result, exclusive=True)
    return result
