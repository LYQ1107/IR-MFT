"""Artificial full-length metadata/CSV fixtures; no trained models or HOTA evidence."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from test_fusion_memory_design import additional_scores, normal_scores, parent
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_design import normal_case_plan, select_visual_parent, extra_cases, select_memory
from tools.fusion_memory_dev_freeze import dev_path, final_dev_freeze, normal_dev_freeze
from tools.fusion_memory_evaluation import (evaluate_observations, official_csv47, read, sha, verify_observations)
from tools.fusion_memory_observation import LENGTHS, write_json

ROOT = Path(__file__).resolve().parents[1]
PACKET = 'c'*64
EMPTY = __import__('hashlib').sha256(b'').hexdigest()


@pytest.fixture(scope='module')
def official():
    plan = read(ROOT/'configs/perceptual_mft_20261122.json')
    schema, _ = official_schema(plan['evaluation']['TrackEval_root'])
    with (Path(plan['frozen_detection_sources']['rgb_dev'])/'trackeval_output/pedestrian_detailed.csv').open() as stream:
        columns = csv.DictReader(stream).fieldnames
    return schema, columns


def table(path, value, official):
    schema, columns = official
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for seq in [*LENGTHS['dev'], 'COMBINED']:
            row = {key: 0 for key in columns}
            row.update(seq=seq, CLR_Frames=sum(LENGTHS['dev'].values()) if seq == 'COMBINED' else LENGTHS['dev'][seq])
            for key in columns:
                if key.startswith('HOTA___') or key.startswith('AssA___') or key == 'IDF1':
                    row[key] = value/100
            writer.writerow(row)
    return path


def original_objects(root, official):
    records = [parent('native'), parent('spatial')]
    for record in records:
        for row in [*record['normal_best'], record['fresh_RGB'], record['canonical_RGB'], *record['interventions'].values()]:
            path = table(root/'original_dev'/(row['id']+'.csv'), row['metrics']['HOTA'], official)
            row.update(official_CSV=str(path), official_CSV_sha256=sha(path),
                source_receipts={seq: dict(track_sha256=EMPTY, detection_packet_sha256=PACKET) for seq in LENGTHS['dev']})
    return normal_case_plan(select_visual_parent(records))


def phase(root, name, cases, scores, official, normal_choice=None):
    directory = root/name
    spec_path = root/(name+'.json')
    spec = dict(split='dev', cases=cases, stream_directory=str(directory), pipeline_registration_sha256='a'*64)
    if normal_choice is not None:
        spec['normal_Dev_choice_sha256'] = sha(normal_choice)
    write_json(spec_path, spec, exclusive=True)
    versions = {'tools/fusion_memory_design.py': sha(ROOT/'tools/fusion_memory_design.py')}
    anchors = {}
    for seq, frames in LENGTHS['dev'].items():
        streams = {}
        for case in cases:
            track = directory/case['id']/'track_results'/(seq+'.txt')
            track.parent.mkdir(parents=True, exist_ok=True)
            track.write_text('')
            streams[case['id']] = dict(sequence=seq, frames=frames, track_path=str(track),
                track_sha256=EMPTY, detection_packet_sha256=PACKET, detections=0, track_rows=0)
        anchor = root/'anchors'/(seq+'.json')
        write_json(anchor, dict(sequence=seq, split='dev', frames=frames, GT_read=False,
            original_image_bytes_delivery_sha256='b'*64))
        anchors[seq] = dict(path=str(anchor), sha256=sha(anchor))
        write_json(directory/'sequence_manifests'/(seq+'.json'), dict(status='COMPLETE_F4_CAUSAL_SEQUENCE_OBSERVATIONS',
            formal_Dev_Test_sequence_coverage=True, sequence=seq, split='dev', frames=frames,
            complete_source_sequence_frames=frames, cases=cases, streams=streams,
            original_image_bytes_delivery_sha256='b'*64, GT_read=False, future_frame_input=False,
            previous_outputs_rewritten=False, independent_input_batch_size=1,
            provenance=dict(specification_sha256=sha(spec_path), pipeline_registration_sha256='a'*64,
                            source_versions=versions, physical_gpu='4')))
    files = {}
    schema = official[0]
    for case in cases:
        name = case['id']
        manifest = verify_observations(spec_path, case, directory, 'a'*64, versions, anchors)
        csv_path = table(root/'metrics'/name/'pedestrian_detailed.csv', scores[name]['metrics']['HOTA'], official)
        parsed = official_csv47(csv_path, schema, 'dev')
        headline = {s['metric']: round(parsed['combined47'][s['metric']], 3) for s in schema if s['in_official_headline']}
        path = csv_path.parent/'metrics.json'
        write_json(path, dict(status='COMPLETE_F4_OFFICIAL_METRICS', case=case, manifest=manifest, metrics39=headline, **parsed))
        files[name] = str(path)
    return files, spec_path


def complete_fixture(root, official):
    plan = original_objects(root, official)
    scores = normal_scores(plan)
    files, _ = phase(root, 'dev_normal', plan['normal_cases'], scores, official)
    normal = root/'normal_choice.json'
    normal_dev_freeze(plan, files, official[0], normal)
    additions = extra_cases(plan, select_memory(plan, scores))
    extra_files, _ = phase(root, 'dev_additional', additions, additional_scores(plan, scores), official, normal)
    return plan, files, extra_files, normal


def test_complete_files_freeze_all78_without_authorizing_test(tmp_path, official):
    plan, normal_files, more, normal = complete_fixture(tmp_path, official)
    result = final_dev_freeze(plan, normal, normal_files, more, official[0], tmp_path/'frozen.json')
    assert len(result['cases']) == 78 and result['all_normal_negative_cases_retained']
    assert result['full_original_CSV_MOT_and_packet_missing_equivalence']
    assert not result['formal_launch_ready'] and not result['Test_accuracy_read']
    assert result['supported_innovations'] == 0 and not result['goal_met']


@pytest.mark.parametrize('change', ['frames', 'nonfinite_non_AUC', 'missing_sequence', 'headline'])
def test_all276_columns_and_full_coverage_checked(tmp_path, official, change):
    path = table(tmp_path/'table.csv', 40, official)
    schema = official[0]
    original = official_csv47(path, schema, 'dev')
    with path.open() as stream:
        reader = csv.DictReader(stream)
        rows, columns = list(reader), reader.fieldnames
    if change == 'frames':
        rows[-1]['CLR_Frames'] = '4833'
    elif change == 'nonfinite_non_AUC':
        key = next(k for k in columns if k.startswith('HOTA___') and not k.endswith('AUC'))
        rows[0][key] = 'nan'
    elif change == 'missing_sequence':
        rows.pop(0)
    else:
        headline = {s['metric']: original['combined47'][s['metric']] for s in schema if s['in_official_headline']}
        headline['HOTA'] += .01
        with pytest.raises(ValueError):
            official_csv47(path, schema, 'dev', headline)
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError):
        official_csv47(path, schema, 'dev')


def test_partial_observations_fail_before_any_gt_evaluation(tmp_path, official):
    plan = original_objects(tmp_path, official)
    scores = normal_scores(plan)
    cases = [plan['normal_cases'][0]]
    files, spec = phase(tmp_path, 'dev_normal', cases, scores, official)
    value = read(files[cases[0]['id']])
    manifest = value['manifest']
    path = Path(manifest['sources']['BT-005']['path'])
    source = read(path);source['frames'] = 1833;write_json(path, source)
    with patch('tools.fusion_memory_evaluation.run_official_tracks', side_effect=AssertionError('GT evaluation attempted')) as run:
        with pytest.raises(ValueError, match='Incomplete'):
            evaluate_observations(spec, cases[0], tmp_path/'dev_normal', 'a'*64, manifest['source_versions'],
                manifest['image_source_anchors'], tmp_path/'evaluation', {})
    run.assert_not_called()
    assert not (tmp_path/'evaluation').exists()


def test_all_negative_registered_cases_required_before_choice(tmp_path, official):
    plan = original_objects(tmp_path, official)
    files, _ = phase(tmp_path, 'dev_normal', plan['normal_cases'], normal_scores(plan), official)
    files.pop(next(iter(files)))
    with pytest.raises(ValueError, match='including negatives'):
        normal_dev_freeze(plan, files, official[0], tmp_path/'choice.json')


def test_normal_choice_cannot_change_after_interventions(tmp_path, official):
    plan, files, extra, normal = complete_fixture(tmp_path, official)
    value = read(normal)
    value['selection']['primary_IR']['appearance_weight'] = .1
    write_json(normal, value)
    with pytest.raises(ValueError, match='Original normal choice changed'):
        final_dev_freeze(plan, normal, files, extra, official[0], tmp_path/'frozen.json')


def test_missing_equivalence_compares_extra_metrics_not_only_hota(tmp_path, official):
    plan, files, more, normal = complete_fixture(tmp_path, official)
    name = next(k for k, p in more.items() if read(p)['case']['intervention'] == 'IR_unavailable'
                and read(p)['case']['appearance_weight'] == .05)
    value = read(more[name])
    path = Path(value['official_CSV'])
    with path.open() as stream:
        reader = csv.DictReader(stream);rows, columns = list(reader), reader.fieldnames
    for i, row in enumerate(rows):row['MOTP_sum'] = str(2 if i == 2 else 1)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns);writer.writeheader();writer.writerows(rows)
    value.update(official_csv47(path, official[0], 'dev', value['metrics39']))
    write_json(Path(more[name]), value)
    with pytest.raises(ValueError, match='whole original276-column CSV'):
        final_dev_freeze(plan, normal, files, more, official[0], tmp_path/'frozen.json')


def test_test_accuracy_path_rejected_before_open(tmp_path):
    with pytest.raises(ValueError, match='Test accuracy paths'):
        dev_path(tmp_path/'test/case/metrics.json')
