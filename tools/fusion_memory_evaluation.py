"""Official F4 metric/file components; registration and execution remain caller-owned.

No model loading, CUDA or Test selection. A complete receipt is not by itself
proof of parent qualification; the formal driver must verify that separately.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from tools.fusion_memory_observation import LENGTHS, write_json
from tools.fusion_memory_parent_io import require, ROOT


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def official_csv47(path, schema, split, headline=None):
    require(split in ['dev', 'test'], 'Only complete formal Dev/Test tables supported')
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns, raw = reader.fieldnames, list(reader)
    require(columns is not None and len(columns) == len(set(columns)) == 276, 'Original276-column official CSV required')
    require(len(schema) == 47 and len({s['metric'] for s in schema}) == 47, 'All47 official fields required')
    aggregate = {k for k in columns if k != 'seq' and ('___' not in k or k.endswith('___AUC'))}
    require(aggregate == {s['source_column'] for s in schema}, 'Official CSV aggregate schema differs')
    require([r['seq'] for r in raw] == [*LENGTHS[split], 'COMBINED'], 'Original full per-sequence and combined rows required')
    result = []
    for row in raw:
        require(set(row) == set(columns) and all(math.isfinite(float(row[k])) for k in columns if k != 'seq'),
                'Nonfinite or incomplete official CSV')
        values = {}
        for item in schema:
            value = float(row[item['source_column']])
            if item['unit'] == 'count':
                require(value.is_integer(), 'Official integer count differs')
                values[item['metric']] = int(value)
            else:
                values[item['metric']] = value*item['scale_from_raw_detailed']
        count = sum(LENGTHS[split].values()) if row['seq'] == 'COMBINED' else LENGTHS[split][row['seq']]
        require(values['CLR_Frames'] == count, 'Official table does not cover the complete sequence')
        require(values['CLR_TP']+values['CLR_FN'] == values['GT_Dets'] and
                values['IDTP']+values['IDFN'] == values['GT_Dets'] and
                values['IDTP']+values['IDFP'] == values['Dets'], 'Official count conservation differs')
        require(math.isclose(values['HOTA_TP']+values['HOTA_FN'], values['GT_Dets'], rel_tol=1e-12, abs_tol=1e-8) and
                math.isclose(values['HOTA_TP']+values['HOTA_FP'], values['Dets'], rel_tol=1e-12, abs_tol=1e-8) and
                math.isclose(values['FP_per_frame'], values['CLR_FP']/count, rel_tol=1e-12, abs_tol=1e-12),
                'Official array/count units differ')
        require(all(0 <= values[k] <= 100 for k in ['HOTA', 'AssA', 'IDF1']), 'Invalid official selection metric')
        result.append(dict(seq=row['seq'], split=split, **values))
    combined = {s['metric']: result[-1][s['metric']] for s in schema}
    if headline is not None:
        expected = {s['metric'] for s in schema if s['in_official_headline']}
        require(set(headline) == expected and len(headline) == 39 and all(
            math.isfinite(float(v)) and abs(combined[k]-float(v)) <= .000501 for k, v in headline.items()),
            'Official39 summary and original full-precision CSV differ')
    return dict(combined47=combined, per_sequence47=result,
        selection_metrics={k: combined[k] for k in ['HOTA', 'AssA', 'IDF1']},
        official_CSV=str(Path(path).resolve()), official_CSV_sha256=sha(path), original_columns=276)


def validate_mot(path, frames, rows):
    previous, seen, count = 0, set(), 0
    with Path(path).open() as stream:
        for line in stream:
            cells = line.strip().split(',')
            require(len(cells) == 10, 'Original10-column MOT required')
            frame, identity = int(cells[0]), int(cells[1])
            values = [float(v) for v in cells[2:7]]
            require(1 <= frame <= frames and frame >= previous and identity > 0 and
                    all(math.isfinite(v) for v in values) and values[2] > 0 and values[3] > 0 and
                    0 <= values[4] <= 1, 'Invalid causal MOT output')
            if frame != previous:
                seen.clear()
            require(identity not in seen, 'Duplicate emitted identity')
            seen.add(identity)
            previous, count = frame, count+1
    require(count == rows, 'Declared MOT row coverage differs')


def verify_observations(specification, case, directory, registration_sha256, source_versions, anchors):
    """Before GT: actual files, whole sequences and fixed input-image anchors."""
    path, directory = Path(specification), Path(directory)
    spec = read(path)
    split = spec['split']
    require(split in ['dev', 'test'] and case in spec['cases'], 'Exact frozen formal case required')
    require(spec['pipeline_registration_sha256'] == registration_sha256 and
            Path(spec['stream_directory']).resolve() == directory.resolve(), 'Frozen observation destination/registration differs')
    require(set(anchors) == set(LENGTHS[split]), 'Every original source image anchor required')
    for name, digest in source_versions.items():
        require(sha(ROOT/name) == digest, 'Frozen inference source changed: '+name)
    coverage, sources = {}, {}
    for seq, frames in LENGTHS[split].items():
        source = directory/'sequence_manifests'/(seq+'.json')
        value = read(source)
        require(value['status'] == 'COMPLETE_F4_CAUSAL_SEQUENCE_OBSERVATIONS' and value['formal_Dev_Test_sequence_coverage'] is True and
                value['split'] == split and value['sequence'] == seq and value['frames'] == frames == value['complete_source_sequence_frames'] and
                value['cases'] == spec['cases'] and value['GT_read'] is False and value['future_frame_input'] is False and
                value['previous_outputs_rewritten'] is False and value['independent_input_batch_size'] == 1,
                'Incomplete or noncausal original sequence observations')
        provenance = value['provenance']
        require(provenance['specification_sha256'] == sha(path) and
                provenance['pipeline_registration_sha256'] == registration_sha256 and
                provenance['source_versions'] == source_versions and provenance['physical_gpu'] in ['4', '5'],
                'Observation execution provenance differs')
        anchor = anchors[seq]
        require(sha(anchor['path']) == anchor['sha256'], 'Original input-image anchor changed')
        original = read(anchor['path'])
        require(original['sequence'] == seq and original['split'] == split and original['frames'] == frames and
                original['GT_read'] is False and original['original_image_bytes_delivery_sha256'] == value['original_image_bytes_delivery_sha256'],
                'Different fixed November image bytes')
        stream = value['streams'][case['id']]
        track = directory/case['id']/'track_results'/(seq+'.txt')
        require(Path(stream['track_path']).resolve() == track.resolve() and sha(track) == stream['track_sha256'] and
                stream['frames'] == frames and stream['sequence'] == seq, 'Track file path/bytes/coverage differ')
        require(len(stream['detection_packet_sha256']) == 64, 'Missing observation packet digest')
        validate_mot(track, frames, stream['track_rows'])
        coverage[seq], sources[seq] = stream, dict(path=str(source), sha256=sha(source))
    return dict(split=split, sequences=list(LENGTHS[split]), frames=sum(LENGTHS[split].values()),
        case=case, specification=str(path), specification_sha256=sha(path), coverage=coverage, sources=sources,
        pipeline_registration_sha256=registration_sha256, source_versions=source_versions, image_source_anchors=anchors,
        GT_inference_input=False, future_frame_input=False, previous_outputs_rewritten=False)


def run_official_tracks(result_directory, output, plan, split):
    """Separate CPU evaluation only; no inference, weights or model selection."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Official metrics run on CPU only')
    require(split in ['dev', 'test'], 'Official evaluation requires full original Dev/Test')
    from tools.eval_mft26ir import run_trackeval
    from tools.export_perceptual_metrics47 import official_schema
    metrics = run_trackeval(str(result_directory), plan['evaluation']['GT_root'], str(output),
        list(LENGTHS[split]), plan['evaluation']['TrackEval_root'], data_split='train' if split == 'dev' else 'test')
    schema, counts = official_schema(plan['evaluation']['TrackEval_root'])
    parsed = official_csv47(Path(output)/'trackeval_output/pedestrian_detailed.csv', schema, split, metrics)
    return dict(metrics39=metrics, official_schema=schema, official_family_field_counts=counts, **parsed)


def evaluate_observations(specification, case, directory, registration_sha256, source_versions, anchors, output, plan):
    """Caller must establish actual parent eligibility and frozen Test choices."""
    output = Path(output)
    require(not output.exists(), 'Preserve prior official evaluation outputs')
    manifest = verify_observations(specification, case, directory, registration_sha256, source_versions, anchors)
    output.mkdir(parents=True)
    write_json(output/'inference_manifest.json', dict(status='F4_COMPLETE_INFERENCE_BEFORE_GT_EVALUATION', **manifest), exclusive=True)
    values = run_official_tracks(Path(directory)/case['id']/'track_results', output, plan, manifest['split'])
    require(verify_observations(specification, case, directory, registration_sha256, source_versions, anchors) == manifest,
            'Inference files changed during official evaluation')
    result = dict(status='COMPLETE_F4_OFFICIAL_METRICS', case=case, manifest=manifest, **values,
        complete_parent_eligibility_and_Test_freeze_must_be_verified_by_driver=True,
        formal_launch_ready=False, goal_met=False, supported_innovations=0)
    write_json(output/'metrics.json', result, exclusive=True)
    return result
