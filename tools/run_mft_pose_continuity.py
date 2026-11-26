#!/usr/bin/env python3
"""CPU-only causal pose/identity continuity study on the sealed native-IR identity banks.

The separately frozen R2 GPU experiment retains ownership. This controller
selects exclusively from complete Dev before it reads any new Test accuracy.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.audit_mft_tracklet_r2 import bank_audit
from tools.fusion_memory_evaluation import run_official_tracks, validate_mot
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.mft_pose_continuity import PoseContinuity, policies
from tools.native_pyramid_experiment import context, frozen_json, read
from tools.native_pyramid_stream_tracks import mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_tracklet_permanence_r2 import registered as peer_registered
from tools.run_mft_train_motion_prior import registered as previous_registered
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_queue import Queue
from yolox.tracker.byte_tracker import STrack

SPEC = ROOT / 'configs/perceptual_pose_continuity_20261127.json'
PREFIX = 'POSE_CONTINUITY_20261127'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    values = context()
    work = values[4]
    return (*values, work / 'pose_continuity_20261127',
            work / 'tracklet_permanence_20261127_r2')


def peer_dev_ready(peer):
    dev = read(peer / 'complete_Dev.json')
    assert dev['Test_read_for_selection'] is False
    state = read(peer / 'state.json')
    jobs = [job for name, rows in state['phases'].items()
            if name.endswith(('_GPU_ENGINEERING', '_DEV_NN', '_DEV_REPLAY', '_DEV_OFFICIAL'))
            for job in rows.values()]
    assert len(jobs) == 11
    assert all(j['status'] == 'COMPLETE' and type(j.get('exit_code')) is int
               and j['exit_code'] == 0 and identity(j['pid']) is None for j in jobs)
    assert read(peer / 'Test_selection_frozen.json')['Test_read_for_selection'] is False
    return dev


def previous_ready(work):
    prior=work/'train_motion_prior_20261127'
    execution=read(prior/'controller.execution.json')
    assert execution['actual_child_exit_code']==0 and execution['source_unchanged']
    assert identity(execution['identity']['PID']) is None
    state=read(prior/'state.json')
    assert state['status']=='COMPLETE_TRAIN_MOTION_PRIOR_NO_GAIN_OVER_LATE_ONLY_NO_NEW_TEST'
    jobs=[j for rows in state['phases'].values() for j in rows.values()]
    assert len(jobs)==8 and all(j['status']=='COMPLETE' and j['exit_code']==0
        and j['argv_identity_observed'] and identity(j['pid']) is None for j in jobs)
    audit=read(prior/'independent_Dev_audit_complete.json')
    assert audit['official_numeric_fields_checked']==1269 and audit['Test_accuracy_parsed'] is False
    audit_execution=read(prior/'independent_Dev_audit_r1.actual_completion.json')
    assert audit_execution['actual_tool_exit_code']==0 and audit_execution['source_unchanged']


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint, base, peer = locations()
    parent = previous_registered()
    previous_ready(work)
    peer_dev_ready(peer)
    proof = read(base / 'CPU_regressions_r2.json')
    assert proof['actual_tool_exit_code'] == 0 and proof['tests_passed'] == 4
    for name, digest in proof['source_hashes'].items():
        assert sha256(ROOT / name) == digest
    fit = read(base / 'Train_thresholds.json')
    execution = read(base / 'Train_threshold_fit_r1.execution.json')
    assert execution['actual_child_exit_code'] == 0 and execution['source_unchanged']
    assert identity(execution['identity']['PID']) is None
    assert fit['Dev_rows'] == fit['Test_rows'] == 0 and not fit['GPU_work']
    assert fit['specification_sha256'] == sha256(ROOT / 'configs/perceptual_ambiguity_reasoning_20261127.json')
    sources = dict(parent['source_hashes'])
    for name in ['tools/mft_pose_continuity.py', 'tools/fit_mft_ambiguity_thresholds.py',
                 'tools/run_mft_pose_continuity.py', 'tools/audit_mft_tracklet_r2.py',
                 'tools/task_utility_queue.py', 'tests/test_mft_pose_continuity.py',
                 str(SPEC.relative_to(ROOT)), 'tools/fit_mft_train_motion_prior.py',
                 'configs/perceptual_train_motion_prior_20261127.json']:
        sources[name] = sha256(ROOT / name)
    paths = [base / 'Train_thresholds.json', base / 'Train_threshold_fit_r1.execution.json',
             base / 'CPU_regressions_r2.json', base / 'CPU_regressions_r2.stdout.txt',
             peer / 'pipeline_registered.json', peer / 'complete_Dev.json',
             peer / 'Test_selection_frozen.json',
             work / 'train_motion_prior_20261127/controller.execution.json',
             work / 'train_motion_prior_20261127/complete_Dev.json',
             work / 'train_motion_prior_20261127/independent_Dev_audit_complete.json',
             work / 'train_motion_prior_20261127/independent_Dev_audit_r1.actual_completion.json']
    banks = {}
    for sequence in LENGTHS['dev']:
        banks[sequence] = bank_audit(peer, 'dev', sequence, peer_registered())
        meta_path = peer / 'cache/dev' / (sequence + '.json')
        paths.extend([meta_path, Path(read(meta_path)['path']),
                      peer / 'streams/dev/sequence_manifests' / (sequence + '.json')])
    prior_hypothesis=read(base/'hypothesis_before_agent_Test_review.json')
    current_hypothesis=read(SPEC)
    prior_hypothesis.pop('status');current_hypothesis.pop('status');current_hypothesis.pop('timing_scope')
    assert prior_hypothesis==current_hypothesis
    paths.extend([base/'hypothesis_before_agent_Test_review.json',base/'hypothesis_timing_context.json'])
    motion=read(base/'Train_motion_statistics.json')
    motion_execution=read(base/'Train_motion_fit_r1.actual_completion.json')
    assert motion_execution['actual_tool_exit_code']==0 and motion_execution['source_unchanged']
    assert motion['Dev_rows']==motion['Test_rows']==0 and motion['complete_Train_videos']==6
    assert not motion['new_tracking_accuracy'] and not motion['GPU_work']
    for name,digest in motion['source_hashes'].items():assert sha256(name)==digest,name
    paths.extend([base/'Train_motion_statistics.json',base/'fit_registered.json',base/'Train_motion_fit_r1.actual_completion.json'])
    artifacts={str(p):sha256(p)for p in paths}
    artifacts.update(motion['source_hashes'])
    guard_storage(work)
    value = dict(status='REGISTERED_CPU_ONLY_POSE_IDENTITY_CONTINUITY', UTC=utc(),
                 source_hashes=sources, artifact_hashes=artifacts,
                 specification_sha256=sha256(SPEC), normal_cases=policies(), Dev_banks=banks,
                 original_peer_registration_sha256=sha256(peer / 'pipeline_registered.json'),
                 peer_GPU_experiment_unchanged=True, new_GPU_work=False, fixed_seed=0,
                 new_pose_Test_accuracy_parsed=False, historical_Test_previously_exposed=True,
                 agent_current_R2_Test_reviewed_before_pose_Dev=True,
                 eight_policy_hypothesis_authored_before_agent_R2_Test_review=True,
                 pose_implementation_and_registration_after_agent_R2_Test_review=True,
                 goal_met=False, supported_innovations=0)
    assert not (base / 'pipeline_registered.json').exists()
    frozen_json(base / 'pipeline_registered.json', value)
    print(json.dumps(dict(status=value['status'], sources=len(sources), CPU_only=True)), flush=True)


def registered():
    plan, study, protocol, calibration, work, checkpoint, base, peer = locations()
    previous_registered()
    reg = read(base / 'pipeline_registered.json')
    assert reg['normal_cases'] == policies() and reg['specification_sha256'] == sha256(SPEC)
    for name, digest in reg['source_hashes'].items():
        assert sha256(ROOT / name) == digest, name
    for name, digest in reg['artifact_hashes'].items():
        assert sha256(name) == digest, name
    return reg


def split_cases(split):
    plan, study, protocol, calibration, work, checkpoint, base, peer = locations()
    if split == 'dev':
        return policies()
    freeze = read(base / 'Test_selection_frozen.json')
    assert freeze['Test_read_for_selection'] is False
    assert freeze['complete_Dev_sha256'] == sha256(base / 'complete_Dev.json')
    return freeze['cases']


def make_tracker(case,thresholds):
    base=locations()[-2]
    motion=read(base/'Train_motion_statistics.json')
    return PoseContinuity(case,thresholds,motion['equal_video_RMS_acceleration_sigma'])


def prefix_rows(arrays, cases, thresholds, sequence, count):
    trackers = {c['id']: make_tracker(c, thresholds) for c in cases}
    rows = {c['id']: [] for c in cases}
    for i, frame in enumerate(arrays['frame_ids'][:count]):
        lo, hi = map(int, arrays['offsets'][i:i + 2])
        for c in cases:
            values = trackers[c['id']].update(sequence, int(frame), arrays['detections'][lo:hi],
                                             arrays['rgb'][lo:hi], arrays['ir'][lo:hi])
            rows[c['id']].extend(values)
    return {k: ''.join(map(mot_line, values)) for k, values in rows.items()}


def identity_score_hash(path):
    digest=hashlib.sha256()
    with Path(path).open()as stream:
        for line in stream:
            fields=line.split(',')
            digest.update(','.join([*fields[:2],*fields[6:]]).encode())
    return digest.hexdigest()


def replay(split, sequence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    reg = registered()
    plan, study, protocol, calibration, work, checkpoint, base, peer = locations()
    cases = split_cases(split)
    bank = bank_audit(peer, split, sequence, peer_registered())
    meta = read(peer / 'cache' / split / (sequence + '.json'))
    with np.load(meta['path']) as data:
        arrays = {k: data[k].copy() for k in data.files}
    thresholds = read(base / 'Train_thresholds.json')['thresholds']
    original_shared_filter = STrack.shared_kalman
    limit = min(32, LENGTHS[split][sequence])
    expected_prefix = prefix_rows(arrays, cases, thresholds, sequence, limit)
    # Only two suffix arrays change; current prefix values remain byte exact.
    cut = int(arrays['offsets'][limit])
    original_future_detections = arrays['detections'][cut:].copy()
    arrays['detections'][cut:, :4] += 777.
    for cue in ['rgb', 'ir']:
        arrays[cue][cut:] *= -1
    assert prefix_rows(arrays, cases, thresholds, sequence, limit) == expected_prefix
    assert STrack.shared_kalman is original_shared_filter
    arrays['detections'][cut:] = original_future_detections
    for cue in ['rgb', 'ir']:
        arrays[cue][cut:] *= -1
    output = base / 'streams' / split
    writer = FusionMemoryStreamWriter(output, sequence, [c['id'] for c in cases])
    trackers = {c['id']: make_tracker(c, thresholds) for c in cases}
    started = time.monotonic()
    try:
        for i, frame in enumerate(arrays['frame_ids']):
            lo, hi = map(int, arrays['offsets'][i:i + 2])
            detections = arrays['detections'][lo:hi]
            rows = {c['id']: trackers[c['id']].update(sequence, int(frame), detections,
                       arrays['rgb'][lo:hi], arrays['ir'][lo:hi]) for c in cases}
            assert STrack.shared_kalman is original_shared_filter
            writer.append(int(frame), rows, {c['id']: detections for c in cases})
            if int(frame) % 200 == 0:
                path = base / 'progress' / split / (sequence + '.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(frame=int(frame), frames=LENGTHS[split][sequence],
                    CPU_only=True, seconds=time.monotonic() - started)) + '\n')
        streams = writer.close(LENGTHS[split][sequence])
    finally:
        writer.abort()
    reference = read(peer / 'streams' / split / 'sequence_manifests' / (sequence + '.json'))
    for ours, old in [('PC_native_zero', 'TP_native_zero'), ('PC_late_only', 'TP_late_RGB')]:
        assert streams[ours]['track_sha256'] == reference['streams'][old]['track_sha256']
    assert all(s['detection_packet_sha256'] == bank['packet_sha256'] for s in streams.values())
    anchor_proofs={}
    for case in cases:
        if case['render']!='filtered':
            old_id='TP_late_RGB'if case['late']else'TP_native_zero'
            own=identity_score_hash(streams[case['id']]['track_path'])
            expected_meta=identity_score_hash(reference['streams'][old_id]['track_path'])
            assert own==expected_meta
            anchor_proofs[case['id']]=dict(actual_frame_ID_score_order_hash=own,reference_hash=expected_meta,reference=old_id)
    registered()
    frozen_json(output / 'sequence_manifests' / (sequence + '.json'), dict(
        status='COMPLETE_CPU_POSE_CONTINUITY_CAUSAL_BANK_REPLAY', UTC=utc(), split=split,
        sequence=sequence, frames=LENGTHS[split][sequence], cases=cases, streams=streams,
        statistics={k: t.statistics for k, t in trackers.items()}, source_versions=reg['source_hashes'],
        bank=bank, anchor_frame_ID_score_order_exact_proofs=anchor_proofs, zero_and_late_only_MOT_exact=True, future_suffix_corruption_prefix_invariant=True,
        scoped_prediction_filter_restored=True, only_current_frame_values_supplied=True,
        compact_feature_cache_read=True, writer_local_false_cache_flag_scope='Writer creates only MOT, not the input bank',
        GT_read=False, future_frame_input=False, previous_outputs_rewritten=False,
        encoder_and_image_cost_excluded_from_replay=True, CPU_replay_seconds=time.monotonic() - started))


def evaluate(split, case_id):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    reg = registered()
    plan, study, protocol, calibration, work, checkpoint, base, peer = locations()
    case = next(c for c in split_cases(split) if c['id'] == case_id)
    coverage = {}
    for sequence, count in LENGTHS[split].items():
        record = read(base / 'streams' / split / 'sequence_manifests' / (sequence + '.json'))
        assert record['frames'] == count and record['source_versions'] == reg['source_hashes']
        assert record['zero_and_late_only_MOT_exact'] and record['future_suffix_corruption_prefix_invariant']
        stream = record['streams'][case_id]
        assert sha256(stream['track_path']) == stream['track_sha256']
        validate_mot(stream['track_path'], count, stream['track_rows'])
        coverage[sequence] = stream
    output = base / 'evaluation' / split / case_id
    assert not output.exists()
    output.mkdir(parents=True)
    values = run_official_tracks(base / 'streams' / split / case_id / 'track_results', output, plan, split)
    registered()
    frozen_json(output / 'metrics.json', dict(status='COMPLETE_OFFICIAL_CPU_POSE_CONTINUITY_METRICS',
        case=case, frames=sum(LENGTHS[split].values()), coverage=coverage,
        source_versions=reg['source_hashes'], **values, goal_met=False, supported_innovations=0))


def run_phase(queue, split):
    plan, study, protocol, calibration, work, checkpoint, base, peer = locations()
    cases = split_cases(split)
    cmd = [sys.executable, '-u', str(Path(__file__).relative_to(ROOT))]
    queue.jobs(PREFIX + '_' + split.upper() + '_REPLAY',
        [dict(id=s, command=cmd + ['replay', '--split', split, '--sequence', s]) for s in LENGTHS[split]],
        False, lambda j: read(base / 'streams' / split / 'sequence_manifests' / (j['id'] + '.json'))['zero_and_late_only_MOT_exact'])
    queue.jobs(PREFIX + '_' + split.upper() + '_OFFICIAL',
        [dict(id=c['id'], command=cmd + ['evaluate', '--split', split, '--case', c['id']]) for c in cases],
        False, lambda j: read(base / 'evaluation' / split / j['id'] / 'metrics.json')['frames'] == sum(LENGTHS[split].values()))
    return [read(base / 'evaluation' / split / c['id'] / 'metrics.json') for c in cases]


def wait_peer_test(base, peer, queue):
    expected = read(peer / 'controller.execution.json')['identity']
    queue.state.update(status='WAITING_UNCHANGED_R2_TEST_TO_REUSE_SEALED_BANKS')
    queue.save()
    while True:
        execution = read(peer / 'controller.execution.json')
        assert execution['identity'] == expected
        if execution.get('actual_child_exit_code') is not None:
            assert execution['actual_child_exit_code'] == 0, 'Preserve peer failure; never restart it'
            assert identity(expected['PID']) is None
            for sequence in LENGTHS['test']:
                assert (peer / 'cache/test' / (sequence + '.json')).exists()
            break
        live = identity(expected['PID'])
        if live is not None:
            assert all(live[k] == expected[k] for k in ['PID', 'start_ticks', 'raw_argv_sha256'])
        time.sleep(15)
    # No score from the completed peer Test is used here or in case selection.


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    reg = registered()
    plan, study, protocol, calibration, work, checkpoint, base, peer = locations()
    peer_dev = peer_dev_ready(peer)
    previous_ready(work)
    with (base / 'state.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (base / 'state.json').exists()
        state = dict(status='STARTING_CPU_ONLY_POSE_CONTINUITY_DEV', phases={}, data_root=plan['data_root'],
                     controller_pid=os.getpid(), new_GPU_work=False, fixed_seed=0, goal_met=False,
                     supported_innovations=0, physical_GPU_owner='Unchanged tracklet R2 experiment')
        queue = Queue(work, base, state, plan['data_root'], registered)
        queue.save()
        try:
            values = run_phase(queue, 'dev')
            for row, peer_id in [(values[0], 'TP_native_zero'), (values[1], 'TP_late_RGB')]:
                ref = next(r for r in peer_dev['rows'] if r['case']['id'] == peer_id)
                assert row['combined47'] == ref['combined47'] and row['per_sequence47'] == ref['per_sequence47']
            chosen = min([r for r in values if r['case'] in policies()], key=lambda r: (*(-r['combined47'][k] for k in ['HOTA', 'AssA', 'IDF1']),
                                               reg['normal_cases'].index(r['case'])))
            gain = chosen['combined47']['HOTA'] - values[1]['combined47']['HOTA']
            frozen_json(base / 'complete_Dev.json', dict(status='COMPLETE_ALL8_CPU_POSE_CONTINUITY_DEV', UTC=utc(),
                rows=values, chosen_case=chosen['case'], gain_over_late_only_pp=gain,
                Test_read_for_selection=False, reference_all47_and_actual_MOT_exact=True,
                scoped_prediction_filter_restored=True, goal_met=False, supported_innovations=0))
            if gain <= 0:
                state.update(status='COMPLETE_POSE_CONTINUITY_NO_GAIN_OVER_LATE_ONLY_NO_NEW_TEST', Dev_gain_pp=gain)
                queue.save()
                return
            frozen_json(base / 'Test_selection_frozen.json', dict(
                status='FROZEN_POSE_CONTINUITY_FROM_COMPLETE_DEV_BEFORE_NEW_TEST_ACCURACY_READ', UTC=utc(),
                cases=[policies()[0], policies()[1], chosen['case']], chosen_case=chosen['case'],
                complete_Dev_sha256=sha256(base / 'complete_Dev.json'), Test_read_for_selection=False,
                historical_Test_previously_exposed=True, peer_Test_score_read_by_selector=False,
                peer_Test_score_previously_reviewed_by_agent=True,
                architecture_hypothesis_authored_before_agent_peer_Test_review=True,
                peer_Test_report_exists_at_freeze=(peer / 'full_Test_results.json').exists()))
            wait_peer_test(base, peer, queue)
            test = run_phase(queue, 'test')
            selected = next(r for r in test if r['case'] == chosen['case'])
            baseline = read(work / 'native_pyramid_evaluation/test/NP_source_canonical_RGB/metrics.json')
            csv = work / 'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv) == 'c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            from tools.export_perceptual_metrics47 import official_schema
            from tools.fusion_memory_evaluation import official_csv47
            schema, _ = official_schema(plan['evaluation']['TrackEval_root'])
            original = official_csv47(csv, schema, 'test', baseline['trackeval'])
            delta = selected['combined47']['HOTA'] - original['combined47']['HOTA']
            frozen_json(base / 'full_Test_results.json', dict(
                status='COMPLETE_DEV_FROZEN_CPU_POSE_CONTINUITY_FULL7_TEST', UTC=utc(), rows=test,
                original_RGB_baseline=original, selected_case=chosen['case'],
                HOTA_gain_over_original_RGB_pp=delta, target_plus5_met=delta >= 5,
                target_plus10_met=delta >= 10, Test_used_for_selection=False,
                historical_Test_previously_exposed=True, goal_met=False, supported_innovations=0))
            state.update(status='COMPLETE_POSE_CONTINUITY_FULL_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT',
                         HOTA_gain_over_original_RGB_pp=delta)
            queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_CPU_POSE_CONTINUITY_ATTEMPT', error=repr(error))
            queue.save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['register', 'run', 'replay', 'evaluate'])
    parser.add_argument('--split', choices=['dev', 'test'])
    parser.add_argument('--sequence')
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.mode == 'register':
        register()
    elif args.mode == 'run':
        run()
    elif args.mode == 'replay':
        replay(args.split, args.sequence)
    else:
        evaluate(args.split, args.case)


if __name__ == '__main__':
    main()
