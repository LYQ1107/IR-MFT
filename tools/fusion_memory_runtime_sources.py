"""Cost references from complete F4/original Test observations, never accuracy.

Only use after the actual complete Dev-derived case freeze. All original parent
campaigns must already have completed; this helper grants no GPU authorization.
"""
import hashlib
from pathlib import Path

from tools.fusion_memory_evaluation import read, sha, verify_observations
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_parent_io import require


def original_source(workflow, case, sequence):
    visual = case['visual']
    family = 'native' if visual['baseline'] else visual['family']
    require(family in ['native', 'spatial'], 'Only originally qualified visual families supported')
    base = workflow.work if family == 'native' else workflow.work/'native_spatial'
    path = base/('native_pyramid_specs' if family == 'native' else 'specifications')/'test_frozen.json'
    freeze_path = base/('native_pyramid_Test_selection_frozen.json' if family == 'native' else 'Test_selection_frozen.json')
    spec, frozen = read(path), read(freeze_path)
    require(spec['split'] == 'test' and frozen['complete_Dev_before_Test'] is True and
            frozen['specification_sha256'] == sha(path) and
            spec['study_sha256'] == workflow.study['parent_studies'][family]['sha256'],
            'Original source must retain its original complete Dev freeze')
    if visual['baseline'] or case['intervention'] == 'IR_unavailable':
        matches = [m for m in spec['models'] if m.get('baseline') == 'fresh_RGB_B1']
    elif case['intervention'] == 'normal':
        matches = [m for m in spec['models'] if m == visual['model']]
    else:
        require(case['intervention'] == 'IR_image_spatial_roll', 'Unknown original source intervention')
        matches = [m for m in spec['models'] if m.get('intervention') == 'IR_image_spatial_roll' and all(
            m.get(k) == v for k, v in visual['model'].items() if k not in ['id', 'intervention'])]
    require(len(matches) == 1, 'Find exactly the identical original frozen visual/intervention')
    model = matches[0]
    source_path = Path(spec['stream_directory'])/'sequence_manifests'/(sequence+'.json')
    value = read(source_path)
    expected_status = ('COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT' if family == 'native' else
                       'COMPLETE_SPATIAL_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT')
    require(value['status'] == expected_status and value['split'] == 'test' and value['sequence'] == sequence and
            value['frames'] == LENGTHS['test'][sequence] and value['GT_read'] is False and
            value['future_frame_input'] is False and value['previous_outputs_rewritten'] is False and
            value['specification_sha256'] == sha(path) and value['study_sha256'] == spec['study_sha256'] and
            value['physical_gpu'] in ['4', '5'], 'Original complete Test producer identity differs')
    require(all(value['models'][model['id']][k] == v for k, v in model.items()),
            'Original Test producer loaded a different visual')
    stream = value['streams'][model['id']]
    require(sha(stream['track_path']) == stream['track_sha256'], 'Original source MOT bytes changed')
    return value, stream, dict(path=str(source_path), sha256=sha(source_path), family=family,
        specification=str(path), specification_sha256=sha(path), model_id=model['id'],
        track_path=stream['track_path'], track_sha256=stream['track_sha256'])


def literal_MOT_prefix(path, frames=32):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for row in stream:
            if int(row.split(b',', 1)[0]) > frames:
                break
            digest.update(row)
    return digest.hexdigest()


def runtime_references(workflow, specification, case):
    spec, registration = workflow.validate_specification(specification)
    require(spec['phase'] == 'test_frozen' and spec['split'] == 'test' and case in spec['cases'],
            'Runtime requires an actual complete Dev-derived frozen Test case')
    manifest = verify_observations(specification, case, spec['stream_directory'],
        sha(workflow.base/'pipeline_registered.json'), registration['source_hashes'], spec['image_source_anchors'])
    zeros = [c for c in spec['cases'] if c['visual_key'] == case['visual_key'] and
             c['intervention'] == case['intervention'] and c['profile'] == 'byte']
    require(len(zeros) == 1, 'Exactly one original zero-memory computation required')
    zero = zeros[0]
    result, pins = {}, {}
    for sequence in LENGTHS['test']:
        path = Path(spec['stream_directory'])/'sequence_manifests'/(sequence+'.json')
        value = read(path)
        require(value['prefix_frames'] == 32 and len(value['prefix_source_rows']) == 32,
                'Full production must retain the original 32-frame prefix')
        original, source_stream, original_pin = original_source(workflow, case, sequence)
        zero_stream, current = value['streams'][zero['id']], value['streams'][case['id']]
        require(sha(zero_stream['track_path']) == zero_stream['track_sha256'] and all(
            zero_stream[k] == source_stream[k] for k in ['track_sha256', 'detection_packet_sha256']),
            'Zero-memory full packets and MOT must equal the original visual source')
        require(current['detection_packet_sha256'] == zero_stream['detection_packet_sha256'] and
                value['original_image_bytes_delivery_sha256'] == original['original_image_bytes_delivery_sha256'],
                'Memory altered detector packets or original source-image delivery')
        require(value['prefix_packet_sha256'][case['id']] == value['prefix_packet_sha256'][zero['id']] and
                value['prefix_literal_MOT_sha256'][zero['id']] == literal_MOT_prefix(source_stream['track_path']),
                'Production detection/zero-MOT prefix differs from the original source')
        if original_pin['family'] == 'spatial':
            require(value['prefix_source_rows'] == original['prefix32_source_rows'] and
                    value['prefix_packet_sha256'][case['id']] == original['prefix32_packet_sha256'][original_pin['model_id']] and
                    value['prefix_literal_MOT_sha256'][zero['id']] == original['prefix32_literal_MOT_sha256'][original_pin['model_id']],
                    'Spatial stored source prefix differs')
        # Original native production has no stored packet/image prefix. Its full
        # image/packet digest and exact literal MOT bytes were checked above;
        # both independent passes check actual current bytes against F4's prefix.
        if case['intervention'] == 'IR_unavailable':
            fresh = workflow.parents.plan['role_visual_keys']['fresh_RGB']
            controls = [c for c in spec['cases'] if c['visual_key'] == fresh and c['intervention'] == 'normal' and
                c['appearance_weight'] == case['appearance_weight'] and (c['profile'] == 'byte' or
                (c['profile'] == 'rgb_uniform' and c['minimum_reliability'] == case['minimum_reliability']))]
            require(len(controls) == 1 and all(current[k] == value['streams'][controls[0]['id']][k] for k in
                    ['track_sha256', 'detection_packet_sha256']), 'Missing IR must equal its same-floor/weight fresh RGB memory control')
        result[sequence] = value
        pins[sequence] = dict(path=str(path), sha256=sha(path), original_visual_source=original_pin,
            original_visual_packets_exact=True, zero_memory_full_MOT_exact=True,
            original_native_packet_prefix_saved=False if original_pin['family'] == 'native' else True,
            current_track=dict(path=current['track_path'], sha256=current['track_sha256']),
            zero_track=dict(path=zero_stream['track_path'], sha256=zero_stream['track_sha256']),
            full_sequence_frames=LENGTHS['test'][sequence])
    return result, dict(source_pins=pins, complete_observation_manifest=manifest,
                        no_Test_accuracy_or_GT_read=True)


def unchanged_runtime_sources(evidence):
    for pin in evidence['source_pins'].values():
        require(sha(pin['path']) == pin['sha256'], 'Complete F4 runtime source changed')
        source = pin['original_visual_source']
        require(sha(source['path']) == source['sha256'] and sha(source['specification']) == source['specification_sha256'],
                'Original visual runtime source changed')
        require(sha(source['track_path']) == source['track_sha256'] and all(
            sha(pin[name]['path']) == pin[name]['sha256'] for name in ['current_track', 'zero_track']),
            'Runtime source MOT bytes changed')
