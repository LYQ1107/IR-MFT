"""Artificial images/models and clocks: timing/state/source checks, no real FPS."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_fusion_memory_observation import artificial_files
from test_fusion_memory_online import engine, visual
from tools.fusion_memory_design import make_case
from tools.fusion_memory_evaluation import read, sha
from tools.fusion_memory_observation import FixedSequenceImages, LENGTHS, observe_sequence, write_json
from tools.fusion_memory_runtime import aggregate_cost, measure_sequence
from tools.fusion_memory_runtime_sources import runtime_references, unchanged_runtime_sources


def fixture(tmp_path, kind='rgb'):
    plan, digest = artificial_files(tmp_path, 4)
    source = visual(baseline=True) if kind == 'rgb' else visual(kind)
    case = make_case(source, 'rgb_uniform' if kind == 'rgb' else 'fused_uniform', .05)
    reference_engine, _, _ = engine([case], kind)
    reference = observe_sequence(reference_engine, FixedSequenceImages(plan, 'train', 'BT-003', 4),
        [case], tmp_path/'original', digest, dict(artificial=True))
    return plan, case, reference


@pytest.mark.parametrize('kind', ['rgb', 'quality', 'temporal', 'spatial'])
def test_one_configuration_two_fresh_passes_preserve_warmup_state_and_exact_production(tmp_path, kind):
    plan, case, reference = fixture(tmp_path, kind)
    engines, readers, ticks, sync = [], [], [], []
    def make_engine():
        value, _, _ = engine([case], kind)
        engines.append(value)
        return value
    def make_images():
        reader = FixedSequenceImages(plan, 'train', 'BT-003', 4)
        readers.append(reader)
        return reader
    def clock():
        value = len(ticks)*.005
        ticks.append(value)
        return value
    result = measure_sequence(make_engine, make_images, case, 'BT-003', reference,
        prefix_frames=4, warmup_frames=1, clock=clock, synchronize=lambda: sync.append(1))
    assert len(engines) == len(readers) == 2 and all(v.closed for v in engines)
    assert all(v.last_frame == 4 and v.detector.model.backbone.calls == 4 for v in engines)
    assert len(ticks) == 6 and len(sync) == 7
    assert result['verification']['measured_frames'] == 0
    assert result['timing']['durations_seconds'] == pytest.approx([.005]*3)
    assert all(v.hashed_IR_frames == 4 and v.decoded_IR_frames == (0 if kind == 'rgb' else 4) for v in readers)
    memory = result['timing']['sampled_retained_state_logical_bytes_max']
    assert memory['track_appearance'] > 0 and memory['track_KF_and_boxes'] > 0
    assert (memory['visual_temporal'] > 0) == (kind == 'temporal')
    assert result['timing']['warmup_to_timing_state_preserved']
    assert all(not v.detector.model.backbone._forward_hooks for v in engines)


def test_bundled_engine_rejected_and_closed_before_images_are_read(tmp_path):
    _, case, reference = fixture(tmp_path)
    second = make_case(case['visual'], 'rgb_uniform', .1)
    value, _, _ = engine([case, second])
    def forbidden():
        pytest.fail('Bundled engine must be rejected before image loading')
    with pytest.raises(ValueError, match='exactly one frozen'):
        measure_sequence(lambda: value, forbidden, case, 'BT-003', reference, prefix_frames=4, warmup_frames=1)
    assert value.closed


@pytest.mark.parametrize('pass_number', [1, 2])
def test_source_change_in_either_pass_is_rejected_and_closes_hooks(tmp_path, pass_number):
    plan, case, reference = fixture(tmp_path)
    made, clocks = [], []
    def make_engine():
        value, _, _ = engine([case])
        made.append(value)
        return value
    def make_images():
        reader = FixedSequenceImages(plan, 'train', 'BT-003', 4)
        if len(made) == pass_number:
            original = reader.read_frame
            def changed(*args):
                rgb, ir, row = original(*args)
                row[-1] = '0'*64
                return rgb, ir, row
            reader.read_frame = changed
        return reader
    with pytest.raises(ValueError, match='source image bytes changed'):
        measure_sequence(make_engine, make_images, case, 'BT-003', reference, prefix_frames=4,
            warmup_frames=1, clock=lambda: clocks.append(1))
    assert len(made) == pass_number and all(v.closed for v in made)
    assert not clocks  # The changed first frame is still pre-timing warmup.


@pytest.mark.parametrize('key', ['prefix_packet_sha256', 'prefix_literal_MOT_sha256'])
def test_output_mismatch_stops_before_any_timing_pass(tmp_path, key):
    plan, case, reference = fixture(tmp_path)
    reference[key][case['id']] = '0'*64
    made = []
    def make_engine():
        value, _, _ = engine([case]); made.append(value); return value
    with pytest.raises(ValueError, match='prefix differs'):
        measure_sequence(make_engine, lambda: FixedSequenceImages(plan, 'train', 'BT-003', 4), case,
            'BT-003', reference, prefix_frames=4, warmup_frames=1, clock=lambda: pytest.fail('No timing allowed'))
    assert len(made) == 1 and made[0].closed


def test_failed_forward_closes_independent_engine(tmp_path):
    plan, case, reference = fixture(tmp_path)
    value, teacher, _ = engine([case]); teacher.head.fail_at = 3
    with pytest.raises(RuntimeError, match='Intentional'):
        measure_sequence(lambda: value, lambda: FixedSequenceImages(plan, 'train', 'BT-003', 4),
            case, 'BT-003', reference, prefix_frames=4, warmup_frames=1)
    assert value.closed and not teacher.backbone._forward_hooks


def test_nonpositive_timing_cannot_be_published_as_cost(tmp_path):
    plan, case, reference = fixture(tmp_path)
    with pytest.raises(ValueError, match='positive and finite'):
        measure_sequence(lambda: engine([case])[0], lambda: FixedSequenceImages(plan, 'train', 'BT-003', 4),
            case, 'BT-003', reference, prefix_frames=4, warmup_frames=1, clock=lambda: 1.)


def test_aggregate_uses_all_frame_latencies_and_actual_even_median():
    rows = [dict(timing=dict(sequence='a', durations_seconds=[.01, .02],
        sampled_retained_state_logical_bytes_max=dict(total=10))),
        dict(timing=dict(sequence='b', durations_seconds=[.03, .04],
        sampled_retained_state_logical_bytes_max=dict(total=20)))]
    result = aggregate_cost(rows)
    assert result['measured_frames'] == 4 and result['seconds'] == pytest.approx(.1)
    assert result['median_latency_ms'] == pytest.approx(25) and result['FPS'] == pytest.approx(40)
    assert result['sampled_retained_state_logical_bytes_max']['total'] == 20
    with pytest.raises(ValueError, match='Unique'):
        aggregate_cost([rows[0], rows[0]])


def source_fixture(root, kind='rgb', intervention='normal', fresh_floor=.2):
    """Complete-length synthetic metadata with empty MOT; no actual Test inference."""
    source = visual(baseline=True) if kind == 'rgb' else visual(kind)
    cases = [make_case(source, 'byte', 0, intervention=intervention),
             make_case(source, 'rgb_uniform' if kind == 'rgb' else 'fused_uniform', .05, intervention=intervention)]
    original_models = [source['model']]
    fresh = visual(baseline=True)
    if intervention != 'normal':
        if intervention == 'IR_image_spatial_roll':
            original_models.append(dict(source['model'], id=source['id']+'_rolled', intervention=intervention))
        else:
            original_models.append(fresh['model'])
            cases.extend([make_case(fresh, 'byte', 0), make_case(fresh, 'rgb_uniform', .05, floor=fresh_floor)])
    family = 'native' if kind != 'spatial' else 'spatial'
    base = root/'native_spatial' if family == 'spatial' else root
    parent_spec = base/('specifications' if family == 'spatial' else 'native_pyramid_specs')/'test_frozen.json'
    parent_streams = root/'original_streams'
    parent = dict(split='test', models=original_models, study_sha256='a'*64, stream_directory=str(parent_streams))
    write_json(parent_spec, parent)
    write_json(base/('Test_selection_frozen.json' if family == 'spatial' else 'native_pyramid_Test_selection_frozen.json'),
        dict(complete_Dev_before_Test=True, specification_sha256=sha(parent_spec)))
    spec_path = root/'f4.json'; directory = root/'f4_streams'
    spec = dict(phase='test_frozen', split='test', cases=cases, stream_directory=str(directory))
    write_json(spec_path, spec)
    write_json(root/'fusion_memory/pipeline_registered.json', dict(artificial=True))
    registration_sha = sha(root/'fusion_memory/pipeline_registered.json')
    spec['pipeline_registration_sha256'] = registration_sha
    anchors = {}; versions = {'tools/fusion_memory_runtime.py': sha(Path(__file__).resolve().parents[1]/'tools/fusion_memory_runtime.py')}
    for seq, frames in LENGTHS['test'].items():
        prefix = [[seq, f, 'b'*64, 'c'*64] for f in range(1, 33)]
        packet = hashlib.sha256()
        for f in range(1, frames+1): packet.update(np.asarray([f, 0], dtype='<i8').tobytes())
        prefix_packet = hashlib.sha256()
        for f in range(1, 33): prefix_packet.update(np.asarray([f, 0], dtype='<i8').tobytes())
        track = parent_streams/'tracks'/(seq+'.txt');track.parent.mkdir(parents=True, exist_ok=True);track.write_text('')
        original = dict(status='COMPLETE_SPATIAL_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT' if family == 'spatial' else
            'COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT', split='test', sequence=seq, frames=frames,
            study_sha256='a'*64, specification_sha256=sha(parent_spec), GT_read=False, future_frame_input=False,
            previous_outputs_rewritten=False, physical_gpu='4', models={m['id']: m for m in original_models},
            original_image_bytes_delivery_sha256='d'*64, streams={m['id']: dict(track_path=str(track),
                track_sha256=sha(track), detection_packet_sha256=packet.hexdigest()) for m in original_models},
            prefix32_source_rows=prefix, prefix32_packet_sha256={m['id']: prefix_packet.hexdigest() for m in original_models},
            prefix32_literal_MOT_sha256={m['id']: sha(track) for m in original_models})
        if family == 'native':
            # Match actual original native producer: it never stored prefixes.
            for key in ['prefix32_source_rows', 'prefix32_packet_sha256', 'prefix32_literal_MOT_sha256']:
                del original[key]
        original_path=parent_streams/'sequence_manifests'/(seq+'.json');write_json(original_path, original)
        streams={}
        for case in cases:
            path=directory/case['id']/'track_results'/(seq+'.txt');path.parent.mkdir(parents=True, exist_ok=True);path.write_text('')
            streams[case['id']]=dict(sequence=seq,frames=frames,track_path=str(path),track_sha256=sha(path),
                detection_packet_sha256=packet.hexdigest(),detections=0,track_rows=0)
        write_json(directory/'sequence_manifests'/(seq+'.json'), dict(status='COMPLETE_F4_CAUSAL_SEQUENCE_OBSERVATIONS',
            formal_Dev_Test_sequence_coverage=True, split='test', sequence=seq, frames=frames, complete_source_sequence_frames=frames,
            cases=cases, streams=streams, original_image_bytes_delivery_sha256='d'*64, GT_read=False,
            future_frame_input=False, previous_outputs_rewritten=False, independent_input_batch_size=1,
            provenance=dict(specification_sha256=sha(spec_path), pipeline_registration_sha256=registration_sha,
                source_versions=versions, physical_gpu='4'), prefix_frames=32, prefix_source_rows=prefix,
            prefix_packet_sha256={c['id']: prefix_packet.hexdigest() for c in cases},
            prefix_literal_MOT_sha256={c['id']: sha(track) for c in cases}))
        anchors[seq]=dict(path=str(original_path), sha256=sha(original_path))
    spec['image_source_anchors']=anchors;write_json(spec_path, spec)
    for seq in LENGTHS['test']:
        p=directory/'sequence_manifests'/(seq+'.json');v=read(p);v['provenance']['specification_sha256']=sha(spec_path);write_json(p,v)
    workflow=SimpleNamespace(work=root,base=root/'fusion_memory',study=dict(parent_studies={family:dict(sha256='a'*64)}),
        validate_specification=lambda _: (spec, dict(source_hashes=versions)),
        parents=SimpleNamespace(plan=dict(role_visual_keys=dict(fresh_RGB=fresh['family']+':'+fresh['id']))))
    return workflow,spec_path,cases,base,directory


@pytest.mark.parametrize('kind', ['rgb', 'quality', 'spatial'])
def test_full_synthetic_source_references_require_original_visual_and_zero_memory_identity(tmp_path, kind):
    workflow,spec,cases,_,_=source_fixture(tmp_path,kind)
    references,evidence=runtime_references(workflow,spec,cases[1])
    assert list(references)==list(LENGTHS['test']) and evidence['no_Test_accuracy_or_GT_read']
    assert sum(p['full_sequence_frames'] for p in evidence['source_pins'].values())==15985
    unchanged_runtime_sources(evidence)
    path=next(iter(evidence['source_pins'].values()))['original_visual_source']['path']
    original_bytes=Path(path).read_bytes()
    with open(path,'a') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='Original visual runtime source changed'):
        unchanged_runtime_sources(evidence)
    Path(path).write_bytes(original_bytes)
    track=next(iter(evidence['source_pins'].values()))['original_visual_source']['track_path']
    with open(track,'a') as stream:stream.write('1,1,1,1,2,2,1,-1,-1,-1\n')
    with pytest.raises(ValueError,match='Runtime source MOT bytes changed'):
        unchanged_runtime_sources(evidence)


@pytest.mark.parametrize('intervention', ['IR_unavailable', 'IR_image_spatial_roll'])
def test_interventions_keep_the_identical_original_source_and_missing_memory_control(tmp_path, intervention):
    workflow,spec,cases,_,_=source_fixture(tmp_path,'quality',intervention)
    _,evidence=runtime_references(workflow,spec,cases[1])
    assert all(pin['original_visual_source']['model_id'].endswith('_RGB' if intervention=='IR_unavailable' else '_rolled')
        for pin in evidence['source_pins'].values())


def test_missing_ir_rejects_a_different_reliability_floor_rgb_control(tmp_path):
    workflow,spec,cases,_,_=source_fixture(tmp_path,'quality','IR_unavailable',fresh_floor=0.)
    with pytest.raises(ValueError,match='same-floor/weight'):
        runtime_references(workflow,spec,cases[1])


@pytest.mark.parametrize('change', ['zero_MOT','packet','source_prefix','original_GT','original_freeze'])
def test_forged_runtime_sources_rejected_before_costs(tmp_path,change):
    workflow,spec,cases,base,directory=source_fixture(tmp_path,'spatial' if change=='source_prefix' else 'rgb')
    first=next(iter(LENGTHS['test']));p=directory/'sequence_manifests'/(first+'.json');v=read(p)
    if change=='zero_MOT':
        path=v['streams'][cases[0]['id']]['track_path'];Path(path).write_text('CHANGED')
    elif change=='packet':v['streams'][cases[1]['id']]['detection_packet_sha256']='f'*64
    elif change=='source_prefix':v['prefix_source_rows'][0][-1]='f'*64
    elif change=='original_GT':
        path=tmp_path/'original_streams/sequence_manifests'/(first+'.json');value=read(path);value['GT_read']=True;write_json(path,value)
    else:
        path=base/'native_pyramid_specs/test_frozen.json';value=read(path);value['study_sha256']='f'*64;write_json(path,value)
    write_json(p,v)
    with pytest.raises(ValueError):runtime_references(workflow,spec,cases[1])
