#!/usr/bin/env python3
"""Forward Train/full Dev selection for short measured trusted-ID continuations."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.mft_protected_peripheral_experiment import registered as parent_registered, locations, cache
from tools.mft_protected_peripheral import PeripheralPublisher
from tools.mft_trusted_shadow_bridge import TrustedShadowBridge, policies
from tools.mft_peripheral_frame_bank import SequentialFrameBank
from tools.audit_mft_protected_peripheral import audit_projection
from tools.native_pyramid_experiment import read, frozen_json
from tools.native_pyramid_stream_tracks import mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.fit_mft_proposal_quality import utc
from tools.run_pseudo_ir_comparison import sha256
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_evaluation import validate_mot, official_csv47, run_official_tracks
from tools.export_perceptual_metrics47 import official_schema
from tools.run_mft_coverage_recall import window_csv47
from tools.task_utility_queue import Queue

SOURCES = ['tools/mft_trusted_shadow_bridge.py', 'tools/run_mft_trusted_shadow_bridge.py',
           'tests/test_mft_trusted_shadow_bridge.py', 'configs/perceptual_trusted_shadow_bridge_20261127.json']


def context():
    plan, protocol, work, parent, previous = locations()
    return plan, protocol, work, parent, work / 'trusted_shadow_bridge_20261127'


def register():
    old = parent_registered()
    plan, protocol, work, parent, base = context()
    for name in ['run_r1.execution.json', 'audit_r2.execution.json']:
        e = read(parent / name)
        assert e['actual_child_exit_code'] == 0 and e['source_unchanged']
    spec = read(ROOT / SOURCES[-1])
    val = read(parent / 'complete_forward_Train_validation.json')
    dev = read(parent / 'complete_Dev.json')
    chosen = read(parent / 'Test_selection_frozen.json')['chosen_case']
    assert chosen['id'] == spec['fixed_parent_case']
    baseline_val = next(v for v in val['rows'] if v['case'] == chosen)
    baseline_dev = next(v for v in dev['rows'] if v['case'] == chosen)
    sources = dict(old['source_hashes'])
    for name in SOURCES + ['tools/audit_mft_protected_peripheral.py']:
        sources[name] = sha256(ROOT / name)
    artifacts = [parent / n for n in ['registered.json', 'source_Train_Dev_ready.json',
                 'complete_forward_Train_validation.json', 'complete_Dev.json', 'Test_selection_frozen.json',
                 'run_r1.execution.json', 'audit_r2.execution.json', 'independent_results_audit.json']]
    base.mkdir(parents=True, exist_ok=True)
    budget = guard_storage(work)
    assert budget['work_file_bytes'] + spec['constraints']['new_study_private_file_cap_bytes'] < budget['work_cap_bytes']
    frozen_json(base / 'registered.json', dict(status='REGISTERED_TRUSTED_SHADOW_BEFORE_NEW_HOTA', UTC=utc(),
        source_hashes=sources, artifact_hashes={str(p): sha256(p) for p in artifacts}, specification=spec,
        normal_cases=policies(), qualified=old['qualified'], validation_GT_views=old['validation_GT_views'],
        validation_lengths=old['specification']['validation_lengths'], selected_parent=chosen,
        parent_validation47=baseline_val['combined47'], parent_Dev47=baseline_dev['combined47'],
        prior_Test_exposed=True, new_worker_Test_metrics_read_for_registration=False,
        budget=budget, goal_met=False, supported_innovations=0))


def registered():
    plan, protocol, work, parent, base = context()
    r = read(base / 'registered.json')
    for name, h in r['source_hashes'].items():
        assert sha256(ROOT / name) == h, name
    for name, h in r['artifact_hashes'].items():
        assert sha256(name) == h, name
    budget = guard_storage(work)
    own_bytes = sum(p.lstat().st_size for p in base.rglob('*') if p.is_file())
    assert own_bytes <= r['specification']['constraints']['new_study_private_file_cap_bytes']
    return r


def engineering():
    r = registered(); plan, protocol, work, parent, base = context(); rows = []
    for sequence in LENGTHS['train']:
        meta, a = cache('train', sequence)
        readers = [SequentialFrameBank(sequence, a), SequentialFrameBank(sequence, a)]
        source = PeripheralPublisher(r['selected_parent'])
        trackers = [TrustedShadowBridge(c) for c in policies()]
        prefixes = {c['id']: [] for c in policies()}
        n = min(128, len(a['frame_ids']))
        for frame in range(1, n + 1):
            value = readers[0].next(frame)
            expected = ''.join(map(mot_line, source.update(value)))
            for c, p in zip(policies(), trackers):
                output = p.update(value)
                rendered = ''.join(map(mot_line, output))
                if c['bridge_gap'] == 0:
                    assert rendered == expected
                if frame <= 64:
                    prefixes[c['id']].append(rendered)
            assert all(p.main_projection.hexdigest() == source.main_projection.hexdigest() for p in trackers)
        prefixes2 = {c['id']: TrustedShadowBridge(c) for c in policies()}
        for frame in range(1, min(64, n) + 1):
            value = readers[1].next(frame)
            for c in policies():
                assert ''.join(map(mot_line, prefixes2[c['id']].update(value))) == prefixes[c['id']][frame - 1]
        rows.append(dict(sequence=sequence, frames=n, causal_prefix_frames=min(64, n),
                         disabled_literal_parent=True, all_main_projections_exact=True))
    registered()
    frozen_json(base / 'engineering.json', dict(status='COMPLETE_TRUSTED_SHADOW_REAL_TRAIN_ENGINEERING',
        UTC=utc(), rows=rows, GT_labels_read=False, Test_read=False))


def cases_for(split):
    if split != 'test':
        return policies()
    plan, protocol, work, parent, base = context()
    return read(base / 'Test_selection_frozen.json')['Test_cases']


def publish(split, sequence):
    r = registered(); plan, protocol, work, parent, base = context()
    logical = 'train' if split == 'trainval' else split
    meta, a = cache(logical, sequence); cases = cases_for(split)
    pubs = {c['id']: TrustedShadowBridge(c) for c in cases}; reader = SequentialFrameBank(sequence, a)
    cut = r['validation_GT_views'][sequence]['window_cut'] if logical == 'train' else 0
    files = {}; counts = {key: 0 for key in pubs}; hashes = {key: hashlib.sha256() for key in pubs}
    started = time.monotonic()
    for key in pubs:
        path = base / 'streams' / split / key / 'track_results' / (sequence + '.txt')
        path.parent.mkdir(parents=True, exist_ok=True); files[key] = (path, path.open('x'))
    try:
        for frame in range(1, len(a['frame_ids']) + 1):
            value = reader.next(frame)
            for key, p in pubs.items():
                output = p.update(value)
                if frame <= cut:
                    continue
                text = ''.join(mot_line((f-cut, box, identity, score)) for f, box, identity, score in output)
                files[key][1].write(text); hashes[key].update(text.encode()); counts[key] += len(output)
    finally:
        for path, stream in files.values():
            stream.close()
    manifest = {}
    for key, p in pubs.items():
        path = files[key][0]; assert sha256(path) == hashes[key].hexdigest()
        assert p.main_projection.hexdigest() == meta['original_main_MOT_sha256']
        validate_mot(path, len(a['frame_ids'])-cut, counts[key])
        main_count, all_count = audit_projection(path, a, cut, p.main_alias, meta['scored_reference_MOT_sha256'])
        manifest[key] = dict(track_path=str(path), track_sha256=sha256(path), track_rows=counts[key],
            frames=len(a['frame_ids'])-cut, statistics=p.statistics, main_aliases=p.main_alias,
            independent_main_rows_checked=main_count, independent_output_rows_checked=all_count)
    old = read(parent / 'streams' / split / 'sequence_manifests' / (sequence + '.json'))
    assert manifest['SB_reference']['track_sha256'] == old['streams'][r['selected_parent']['id']]['track_sha256']
    registered()
    frozen_json(base / 'streams' / split / 'sequence_manifests' / (sequence + '.json'),
        dict(status='COMPLETE_CAUSAL_TRUSTED_SHADOW_STREAMS', UTC=utc(), split=split, sequence=sequence,
             frames=len(a['frame_ids'])-cut, all_past_frames_consumed=len(a['frame_ids']), validation_window_cut=cut,
             cases=cases, streams=manifest, source_frame_bank_sha256=meta['sha256'], source_versions=r['source_hashes'],
             main_projection_exact=True, GT_labels_read=False, future_frames_supplied=False, past_outputs_rewritten=False,
             seconds=time.monotonic()-started, cost_excludes_source_neural_images_and_generation=True))


def evaluate(split, case_id):
    r = registered(); plan, protocol, work, parent, base = context()
    case = next(c for c in cases_for(split) if c['id'] == case_id)
    lengths = r['validation_lengths'] if split == 'trainval' else LENGTHS[split]
    for sequence, n in lengths.items():
        m = read(base / 'streams' / split / 'sequence_manifests' / (sequence + '.json'))
        assert m['frames'] == n and m['main_projection_exact']
        v = m['streams'][case_id]; assert sha256(v['track_path']) == v['track_sha256']
    directory = base / 'streams' / split / case_id / 'track_results'
    output = base / 'evaluation' / split / case_id; assert not output.exists(); output.mkdir(parents=True)
    if split == 'trainval':
        from tools.eval_mft26ir import run_trackeval
        gt_root = Path(next(iter(r['validation_GT_views'].values()))['validation_GT_path']).parents[3]
        headline = run_trackeval(str(directory), str(gt_root), str(output), list(lengths), plan['evaluation']['TrackEval_root'], data_split='train')
        schema, families = official_schema(plan['evaluation']['TrackEval_root'])
        result = dict(metrics39=headline, official_schema=schema, official_family_field_counts=families,
                      **window_csv47(output / 'trackeval_output/pedestrian_detailed.csv', schema, lengths, headline))
    else:
        result = run_official_tracks(directory, output, plan, split)
    frozen_json(output / 'metrics.json', dict(status='COMPLETE_TRUSTED_SHADOW_OFFICIAL47', case=case,
        frames=sum(lengths.values()), source_versions=r['source_hashes'], **result))
    receipts = []
    for sequence in lengths:
        duplicate = output / 'trackeval_format' / (sequence + '.txt')
        canonical = directory / (sequence + '.txt'); assert sha256(duplicate) == sha256(canonical)
        receipts.append(dict(path=str(duplicate), bytes=duplicate.stat().st_size, sha256=sha256(duplicate), retained_canonical=str(canonical)))
    for v in receipts:
        Path(v['path']).unlink()
    (output / 'trackeval_format').rmdir()
    frozen_json(output / 'derived_duplicate_cleanup.json', dict(rows=receipts, canonical_retained=True))
    registered()


def phase(queue, split):
    r = registered(); plan, protocol, work, parent, base = context()
    cmd = [sys.executable, '-u', str(Path(__file__).relative_to(ROOT))]
    lengths = r['validation_lengths'] if split == 'trainval' else LENGTHS[split]
    queue.jobs('SHADOW_' + split.upper() + '_PUBLISH', [dict(id=s, command=cmd+['publish', '--split', split, '--sequence', s]) for s in lengths], False,
        lambda j: read(base/'streams'/split/'sequence_manifests'/(j['id']+'.json'))['main_projection_exact'])
    queue.jobs('SHADOW_' + split.upper() + '_OFFICIAL', [dict(id=c['id'], command=cmd+['evaluate', '--split', split, '--case', c['id']]) for c in cases_for(split)], False,
        lambda j: read(base/'evaluation'/split/j['id']/'metrics.json')['frames'] == sum(lengths.values()))
    return [read(base/'evaluation'/split/c['id']/'metrics.json') for c in cases_for(split)]


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    r = registered(); plan, protocol, work, parent, base = context()
    with (base/'state.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); assert not (base/'state.json').exists()
        state = dict(status='STARTING_TRUSTED_SHADOW_STUDY', controller_pid=os.getpid(), phases={}, seed=0,
                     new_GPU_work=False, goal_met=False, supported_innovations=0)
        queue = Queue(work, base, state, plan['data_root'], registered); queue.save()
        try:
            cmd = [sys.executable, '-u', str(Path(__file__).relative_to(ROOT))]
            queue.jobs('SHADOW_ENGINEERING', [dict(id='all6', command=cmd+['engineering'])], False,
                lambda j: len(read(base/'engineering.json')['rows']) == 6)
            val = phase(queue, 'trainval'); dev = phase(queue, 'dev')
            reference = next(v for v in val if v['case']['bridge_gap'] == 0)
            dev_ref = next(v for v in dev if v['case']['bridge_gap'] == 0)
            assert reference['combined47'] == r['parent_validation47']
            assert dev_ref['combined47'] == r['parent_Dev47']
            gains = {v['case']['id']: v['combined47']['HOTA']-reference['combined47']['HOTA'] for v in val}
            eligible = [v for v in dev if v['case']['bridge_gap'] > 0 and v['case']['bridge_agreement'] == 3
                and gains[v['case']['id']] >= r['specification']['minimum_validation_HOTA_gain_over_selected_parent_pp']
                and v['combined47']['HOTA'] >= dev_ref['combined47']['HOTA']-r['specification']['full_Dev_noninferiority_tolerance_pp']]
            chosen = min(eligible, key=lambda v: (-gains[v['case']['id']], *(-v['combined47'][k] for k in ['HOTA','AssA','IDF1']), v['case']['id'])) if eligible else None
            frozen_json(base/'complete_Train_Dev.json', dict(status='COMPLETE_TRUSTED_SHADOW_TRAIN_DEV', UTC=utc(),
                validation_rows=val, Dev_rows=dev, validation_HOTA_gains_over_PP_pp=gains,
                eligible_ids=[v['case']['id'] for v in eligible], selected_case=None if chosen is None else chosen['case'],
                Test_read_for_selection=False, historical_Test_already_exposed=True, goal_met=False, supported_innovations=0))
            if chosen is None:
                state['status'] = 'COMPLETE_TRUSTED_SHADOW_NO_ELIGIBLE_TRAIN_DEV_CASE_NO_TEST'; queue.save(); return
            case = chosen['case']; zero = next(c for c in policies() if c['bridge_gap'] == 0)
            frozen_json(base/'Test_selection_frozen.json', dict(status='FROZEN_SHADOW_FROM_TRAIN_DEV_BEFORE_NEW_TEST', UTC=utc(),
                chosen_case=case, Test_cases=[zero,case], complete_Train_Dev_sha256=sha256(base/'complete_Train_Dev.json'), Test_read_for_selection=False))
            rows = phase(queue, 'test'); selected = next(v for v in rows if v['case'] == case)
            csv = work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv) == 'c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            schema, _ = official_schema(plan['evaluation']['TrackEval_root'])
            baseline = official_csv47(csv, schema, 'test', read(csv.parent.parent/'metrics.json')['trackeval'])
            gain = selected['combined47']['HOTA']-baseline['combined47']['HOTA']
            frozen_json(base/'full_Test_results.json', dict(status='COMPLETE_FROZEN_TRUSTED_SHADOW_TEST7', UTC=utc(), rows=rows,
                selected_case=case, original_RGB_baseline=baseline, HOTA_gain_over_original_RGB_pp=gain,
                target_plus5_met=gain>=5, target_plus10_met=gain>=10, Test_used_for_selection=False,
                historical_Test_exposed=True, goal_met=False, supported_innovations=0))
            state.update(status='COMPLETE_TRUSTED_SHADOW_TEST_REQUIRES_ASSESSMENT', HOTA_gain_over_original_RGB_pp=gain); queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_TRUSTED_SHADOW_ATTEMPT', error=repr(error)); queue.save(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['register','run','engineering','publish','evaluate'])
    p.add_argument('--split', choices=['trainval','dev','test']); p.add_argument('--sequence'); p.add_argument('--case')
    a = p.parse_args()
    if a.mode == 'publish':
        publish(a.split, a.sequence)
    elif a.mode == 'evaluate':
        evaluate(a.split, a.case)
    else:
        {'register': register, 'run': run, 'engineering': engineering}[a.mode]()
