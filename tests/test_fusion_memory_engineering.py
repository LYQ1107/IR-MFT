"""Artificial CPU NN/image/source fixtures; cannot authorize actual F4 GPU work."""
import copy
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from test_fusion_memory_online import CALIBRATION, TinyTeacher, engine, inputs, module, visual
from tools.fusion_memory_design import make_case
from tools.fusion_memory_engineering import (EngineeringMismatch, compare_prefix, engineering_cases,
    original_Dev_prefixes, require_active_primary_memory, unchanged_Dev_prefixes)
from tools.fusion_memory_evaluation import read, sha
from tools.fusion_memory_observation import LENGTHS, write_json
from tools.fusion_memory_online import visual_routes
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_stream_tracks import IndependentByteTrack, mot_line
import tools.probe_f4_gpu_engineering as entry


def plan(kind='quality'):
    ir, rgb = visual(kind), visual(baseline=True)
    cases = [make_case(ir, 'byte', 0.)]+[make_case(ir, profile, weight) for profile in
        ['rgb_uniform', 'rgb_attention_proxy', 'fused_uniform', 'fused_attention_proxy'] for weight in [.05, .1]]
    cases += [make_case(rgb, 'byte', 0.)]+[make_case(rgb, 'rgb_uniform', weight) for weight in [.05, .1]]
    return dict(normal_cases=cases, role_visual_keys=dict(primary_IR=cases[0]['visual_key'], fresh_RGB=cases[-1]['visual_key']),
                visuals={cases[0]['visual_key']: ir, cases[-1]['visual_key']: rgb})


class ArtificialImages:
    def __init__(self, sequence='artificial', frames=4):
        self.sequence, self.frames, self.last_frame = sequence, frames, 0
    def read_frame(self, frame, needs_ir):
        assert frame == self.last_frame+1
        self.last_frame = frame
        rgb, ir = inputs(frame)
        if frame % 2 == 0:
            rgb = rgb//64  # Deliberately varied artificial descriptors exercise memory cost.
        return rgb, ir if needs_ir else None, [self.sequence, frame, hashlib.sha256(rgb.tobytes()).hexdigest(),
                                             hashlib.sha256(ir.tobytes()).hexdigest()]


def pair(kind='quality'):
    p = plan(kind)
    cases = engineering_cases(p)
    composed, _, _ = engine(cases, kind)
    routes, _ = visual_routes(cases)
    modules = {name: None if route['visual']['baseline'] else module(kind) for name, route in routes.items()}
    reference = OnlineNativeDetector(TinyTeacher(), [r['model'] for r in routes.values()], modules, (64, 96), CALIBRATION)
    tracker, digest = IndependentByteTrack(), hashlib.sha256()
    detections = np.asarray([[9., 4., 21., 16., .95], [29., 4., 41., 16., .9]], np.float32)
    for frame in range(1, 5):
        digest.update(''.join(mot_line(row) for row in tracker.update(detections, frame)).encode())
    expected = {c['id']: digest.hexdigest() for c in cases if c['profile'] == 'byte'}
    return p, cases, composed, reference, expected


@pytest.mark.parametrize('kind', ['quality', 'temporal', 'spatial'])
def test_independent_sources_all_profiles_missing_roll_and_numeric_memory_state(kind):
    p, cases, composed, reference, expected = pair(kind)
    result = compare_prefix(composed, reference, ArtificialImages(), cases, expected, frames=4)
    assert len(cases) == 30 and result['prefix_frames'] == 4
    assert all(r['all_independent_source_packets_KF_ID_and_memories_exact'] for r in result['per_frame'])
    assert not result['GPU_authorization_established_here'] and not result['full_Dev_Test_coverage']
    require_active_primary_memory(p, [result])
    assert composed.closed and not composed.detector.model.backbone._forward_hooks
    assert all(not m._forward_hooks for m in composed.detector.modules.values() if m is not None)


def test_packet_difference_has_case_frame_and_both_hashes_without_a_success_receipt():
    _, cases, composed, reference, expected = pair()
    original = reference.step
    def changed(*args):
        packets, info = original(*args)
        for value in packets.values():
            value[0, 4] -= np.float32(.0001)
        return packets, info
    reference.step = changed
    with pytest.raises(EngineeringMismatch) as failure:
        compare_prefix(composed, reference, ArtificialImages(), cases, expected, frames=4)
    detail = failure.value.details
    assert detail['kind'] == 'DETECTION_PACKET' and detail['frame'] == 1
    assert detail['case'] == cases[0]['id'] and detail['actual_sha256'] != detail['expected_sha256']
    assert detail['maximum_absolute_difference'] > 0
    assert composed.closed


@pytest.mark.parametrize('field,kind', [('KF','KF_ID_TRACK_STATE'), ('memory','RETAINED_MEMORY_OR_COST')])
def test_unchanged_MOT_is_insufficient_when_internal_state_is_corrupted(field, kind):
    _, cases, composed, reference, expected = pair()
    name = next(c['id'] for c in cases if c['profile'] == 'fused_uniform' and c['appearance_weight'] == .05)
    original = composed.step
    def changed(*args):
        result = original(*args)
        track = composed.trackers[name].core.tracked_stracks[0]
        if field == 'KF': track.mean[0] += 1
        else: track.rgb_memory[0] += np.float32(.1)
        return result
    composed.step = changed
    with pytest.raises(EngineeringMismatch) as failure:
        compare_prefix(composed, reference, ArtificialImages(), cases, expected, frames=4)
    assert failure.value.details['kind'] == kind and failure.value.details['case'] == name
    assert composed.closed


@pytest.mark.parametrize('kind', ['MOT', 'PACKET'])
def test_original_zero_prefix_is_required_even_if_two_new_paths_agree(kind):
    _, cases, composed, reference, expected = pair()
    name = next(iter(expected)); original_packets = None
    if kind == 'MOT': expected[name] = '0'*64
    else: original_packets = {n: '0'*64 for n in expected}
    with pytest.raises(EngineeringMismatch) as failure:
        compare_prefix(composed, reference, ArtificialImages(), cases, expected, frames=4, original_zero_packets=original_packets)
    assert failure.value.details['kind'] == 'ORIGINAL_ZERO_'+kind+'_PREFIX'
    if kind == 'PACKET': assert len(failure.value.details['actual_per_frame_packet_sha256']) == 4
    assert not composed.detector.model.backbone._forward_hooks


@pytest.mark.parametrize('shared', ['RGB', 'fusion'])
def test_model_aliasing_cannot_claim_independent_source_and_cleans_hooks(shared):
    _, cases, composed, reference, expected = pair()
    if shared == 'RGB': reference.model = composed.detector.model
    else: reference.modules = composed.detector.modules
    with pytest.raises(ValueError, match='Independent RGB|independently loaded'):
        compare_prefix(composed, reference, ArtificialImages(), cases, expected, frames=4)
    assert composed.closed and not composed.detector.model.backbone._forward_hooks


def test_stale_ROI_metadata_is_rejected():
    _, cases, composed, reference, expected = pair()
    capture = next(iter(composed.captures.values()))
    take = capture.take_packet
    def stale(*args):
        packet = take(*args); packet['metadata']['frame'] += 1; return packet
    capture.take_packet = stale
    with pytest.raises(ValueError, match='Current ROI metadata'):
        compare_prefix(composed, reference, ArtificialImages(), cases, expected, frames=4)
    assert composed.closed


@pytest.mark.parametrize('field', ['retained_fused_vectors','changed_edges','distinct_fusion'])
def test_inactive_primary_cannot_obtain_GPU_qualification(field):
    p, cases, composed, reference, expected = pair()
    result = compare_prefix(composed, reference, ArtificialImages(), cases, expected, frames=4)
    name = next(c['id'] for c in p['normal_cases'] if c['profile'] == 'fused_uniform' and c['appearance_weight'] == .05)
    if field == 'retained_fused_vectors': result[field][name] = 0
    elif field == 'changed_edges': result['memory_statistics'][name][field] = 0
    else:
        for row in result['feature_rows'].values(): row['fused_differs_from_RGB_ROIs'] = 0
    with pytest.raises(ValueError, match='not exercised'):
        require_active_primary_memory(p, [result])


def source_fixture(root):
    p = plan()
    ir, fresh = p['visuals'][p['role_visual_keys']['primary_IR']], p['visuals'][p['role_visual_keys']['fresh_RGB']]
    roll = copy.deepcopy(ir); roll['model']['id'] += '_rolled'
    for sequence, frames in LENGTHS['dev'].items():
        for index, visual_row in enumerate([ir, fresh, roll]):
            track = root/sequence/(str(index)+'.txt'); track.parent.mkdir(parents=True,exist_ok=True); track.write_text('')
            producer = root/sequence/(str(index)+'.json')
            write_json(producer, dict(status='COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT',
                split='dev', sequence=sequence, frames=frames, GT_read=False, future_frame_input=False,
                previous_outputs_rewritten=False, streams={visual_row['model']['id']: dict(track_path=str(track),track_sha256=sha(track))}))
            visual_row.setdefault('source_receipts',{})[sequence]=dict(source_manifest=str(producer),
                source_manifest_sha256=sha(producer),track_sha256=sha(track))
    # Cases are complete object snapshots; bind the new artificial source receipts.
    for c in p['normal_cases']: c['visual'] = copy.deepcopy(ir if c['visual_key']==p['role_visual_keys']['primary_IR'] else fresh)
    workflow = SimpleNamespace(parents=SimpleNamespace(plan=p,evidence=dict(selection=dict(chosen_family='native',
        records=[dict(family='native',interventions=dict(IR_image_spatial_roll=roll))]))))
    return workflow, engineering_cases(p)


def test_original_complete_Dev_prefixes_and_missing_source_are_byte_bound(tmp_path):
    workflow,cases=source_fixture(tmp_path)
    expected,pins,rows=original_Dev_prefixes(workflow,cases)
    assert set(expected)=={'BT-005','SN-013'} and not rows  # Actual native has no stored prefix rows.
    assert all(len(p)==4 for p in pins.values())
    unchanged_Dev_prefixes(pins)
    path=next(iter(pins['BT-005'].values()))['track_path']
    with open(path,'a') as stream:stream.write('1,1,1,1,2,2,1,-1,-1,-1\n')
    with pytest.raises(ValueError,match='changed during'):
        unchanged_Dev_prefixes(pins)


@pytest.mark.parametrize('field', ['GT_read','frames'])
def test_incomplete_or_GT_dependent_Dev_prefix_sources_are_rejected(tmp_path,field):
    workflow,cases=source_fixture(tmp_path)
    visual_row=workflow.parents.plan['visuals'][workflow.parents.plan['role_visual_keys']['primary_IR']]
    receipt=visual_row['source_receipts']['BT-005'];p=receipt['source_manifest'];value=read(p)
    value[field]=True if field=='GT_read' else 32;write_json(p,value);receipt['source_manifest_sha256']=sha(p)
    for c in cases:
        if c['visual_key']==workflow.parents.plan['role_visual_keys']['primary_IR'] and c['intervention']=='normal':
            c['visual']['source_receipts']['BT-005']=receipt
    with pytest.raises(ValueError,match='complete original Dev'):
        original_Dev_prefixes(workflow,cases)


@pytest.mark.parametrize('phase,ready', [('test_frozen',True),('dev_normal',False)])
def test_GPU_entry_rejects_wrong_phase_or_live_predecessor_before_loading_tensors(tmp_path,monkeypatch,phase,ready):
    class Workflow:
        base=tmp_path
        def validate_specification(self,path):return dict(phase=phase,split='test' if phase=='test_frozen' else 'dev'),{}
        def predecessor_ready(self):return ready
        @property
        def parents(self):pytest.fail('No model/source load before phase and release checks')
    monkeypatch.setattr(entry,'FusionMemoryWorkflow',Workflow)
    monkeypatch.setattr('sys.argv',['probe_f4_gpu_engineering.py','--specification',str(tmp_path/'artificial.json')])
    with patch.object(entry.torch,'load',side_effect=AssertionError('Tensor loader attempted')) as loader:
        with pytest.raises(ValueError,match='normal Dev|actual process exits'):
            entry.main()
    loader.assert_not_called()
