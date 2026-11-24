#!/usr/bin/env python3
"""Cover the remaining original seed0 native-query behavioral configurations.

Group only checkpoint epochs of the same behavior; preserve every route,
memory, cue, frontend and RGB-core choice. Original complete Dev selects the
snapshot. No Test result is read during registration, and no GPU is used.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.native_common_test_backfill import context, parity, read
from tools.run_pseudo_ir_comparison import dump, sha256

DOCS = ROOT / 'docs/fusion_research'
PARENT = ROOT / 'configs/perceptual_native_common_test_backfill_20261124.json'
SPEC = ROOT / 'configs/perceptual_native_remaining_test_backfill_20261124.json'
CASES = ROOT / 'configs/perceptual_native_remaining_test_cases_20261124'
PARENT_REPORT = DOCS / 'PERCEPTUAL_MFT_20261124_NATIVE_COMMON_TEST_BACKFILL.json'


def signature(config, curve=False):
    """Only ignore identifiers/prose and explicit-false defaults of known flags."""
    result = copy.deepcopy(config)
    for key in ['id', 'registration']:
        result.pop(key, None)
    for key in ['disable_native_increment', 'shuffle_native_cue', 'disable_native_alignment']:
        if not result.get(key):
            result.pop(key, None)
    if curve:
        path = result['native_query_checkpoint']
        assert re.search(r'/epoch[0-9]+\.pt$', path)
        result['native_query_checkpoint'] = re.sub(r'/epoch[0-9]+\.pt$', '/epoch{Dev_selected}.pt', path)
    return json.dumps(result, sort_keys=True, separators=(',', ':'))


def selected_row(rows):
    return min(rows, key=lambda row: (-row['HOTA'], row['epoch'], row['config']['id']))


def finished(pid):
    path = Path('/proc') / str(pid) / 'stat'
    try:
        return path.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    except FileNotFoundError:
        return True


def predecessor_ready(state, report_exists, process_finished=finished):
    if 'FAIL' in state.get('status', ''):
        raise RuntimeError('Preserve failed parent26 before starting more CPU cases')
    if state.get('status') != 'COMPLETE_ALL26_FROZEN_NATIVE36_TEST' or not report_exists:
        return False
    dev, test = state.get('dev_jobs', {}), state.get('test_jobs', {})
    if len(dev) != 8 or len(test) != 26:
        return False
    if not all(row.get('status') == 'COMPLETE' and row.get('source_proof', {}).get(
            'all39metrics_tracks_official_CSV_exact') for row in dev.values()):
        return False
    if not all(row.get('status') == 'COMPLETE' for row in test.values()):
        return False
    pid = state.get('controller_pid')
    return bool(pid and process_finished(pid))


def register():
    _, work, protocol = context()
    assert not SPEC.exists() and not CASES.exists()
    parent = read(PARENT)
    delegated = {signature(row['config']): row['id'] for row in parent['cases']}
    groups = {}
    for path in sorted(work.glob('*/*/metrics.json')):
        data = read(path)
        manifest = data.get('manifest', {})
        config = manifest.get('config', {})
        if (data.get('status') != 'COMPLETE' or manifest.get('split') != 'dev'
                or config.get('native_seed') != 0 or 'native_query_checkpoint' not in config
                or path.parent.parent.name == 'native_common_test_backfill_dev'):
            continue
        assert manifest['frames'] == 4834 and manifest['sequences'] == protocol['dev']
        assert len(data['trackeval']) == 39
        assert config.get('native_feature_family', 'native36') == 'native36'
        checkpoint = Path(config['native_query_checkpoint'])
        assert checkpoint.parent.parent.name in ['native_query', 'native_robust_query']
        assert sha256(checkpoint) == manifest['native_query_checkpoint_sha256']
        row = dict(config=config, HOTA=data['trackeval']['HOTA'], epoch=int(checkpoint.stem[5:]),
            source_Dev_metrics_path=str(path), source_Dev_metrics_sha256=sha256(path),
            checkpoint_sha256=sha256(checkpoint))
        groups.setdefault(signature(config, True), []).append(row)
    assert len(groups) == 58 and sum(len(rows) for rows in groups.values()) == 130
    inventory, cases = [], []
    checkpoint_sources = dict(parent['checkpoint_sources'])
    for index, (behavior, curve) in enumerate(sorted(groups.items())):
        selected = selected_row(curve)
        original = selected['config']
        label = f"NR{index:02d}_{original['backend']}_{original['detection_arm']}_{original['native_cue']}"
        row = dict(group_id=label, behavior_sha256=hashlib.sha256(behavior.encode()).hexdigest(),
            registered_Dev_epochs=sorted(r['epoch'] for r in curve), selected=selected,
            source_curve=[{k: r[k] for k in ['HOTA', 'epoch', 'source_Dev_metrics_path',
                'source_Dev_metrics_sha256', 'checkpoint_sha256']} for r in curve])
        identity = signature(original)
        if identity in delegated:
            row.update(delegated_parent_case=delegated[identity], new_Test_required=False)
        else:
            config = copy.deepcopy(original)
            config['id'] = label
            config['registration'] = (config.get('registration', '') +
                ' Frozen original complete-Dev behavior/epoch selection for formal all7Test backfill. '
                'Keep all registered route variants; no Test-based primary promotion or new seed work.')
            assert signature(config) == identity
            row.update(new_Test_required=True, case_id=label)
            cases.append(dict(id=label, config=config, source=selected, group_id=label))
        inventory.append(row)
        for key in ['native_query_checkpoint', 'association_checkpoint']:
            checkpoint_sources[original[key]] = sha256(original[key])
    assert len(cases) == 50 and sum(not r['new_Test_required'] for r in inventory) == 8
    sources = dict(parent['source_hashes'])
    sources['tools/native_remaining_test_backfill.py'] = sha256(__file__)
    spec = dict(status='FROZEN_REMAINING50_OF_ORIGINAL58_NATIVE_BEHAVIORS_SEED0',
        parent_specification_sha256=sha256(PARENT), groups=inventory, cases=cases,
        registered_original_behavior_groups=58, original_complete_Dev_rows=130,
        delegated_parent26_groups=8, remaining_full_Test_cases=50,
        checkpoint_sources=checkpoint_sources, artifact_sources=parent['artifact_sources'], source_hashes=sources,
        required_Test_sequences=protocol['historical_test'], required_Test_frames_each=15985,
        fixed_seed=0, additional_seed_work=False, GPU_work=False, maximum_CPU_workers=2,
        current_visual_primary_unchanged=True, goal_met=False, supported_innovations=0,
        selection='Within the same registered behavior, use maximum original complete-Dev HOTA, '
            'then earlier epoch, then case ID. No additional Dev sweep and no Test accuracy read.',
        scope='Exact subset:130 existing full Dev rows of seed0 native36 query/route/memory configurations '
            'in the two original native head directories.58 behavior groups;8 are delegated to the already '
            'frozen parent26,50 run here regardless of negative Dev. Encoders, ROI confirmation, other '
            'feature families and current visual architectures remain outside this subset.',
        queue='Start only after parent26 official report exists and its actual controller exits. '
            'Then reproduce all50 original full Dev results before any new Test replay.',
        reporting='All39 summary fields, per-sequence values, unmodified276column detailed CSV and '
            'individual manifests. Cached replay FPS excludes detector/encoder/point preparation. '
            'HistoricalTest exposed; no blind-test, innovation or goal claim from coverage alone.')
    dump(SPEC, spec)
    CASES.mkdir()
    for case in cases:
        dump(CASES / (case['id'] + '.json'), case['config'])
    print(json.dumps(dict(status=spec['status'], original_groups=58, remaining_cases=50,
                          original_Dev_rows=130, Test_accuracy_read=False, GPU_work=False)))


def verify():
    spec = read(SPEC)
    assert spec['parent_specification_sha256'] == sha256(PARENT)
    assert all(sha256(ROOT / path) == digest for path, digest in spec['source_hashes'].items())
    assert all(sha256(path) == digest for path, digest in spec['checkpoint_sources'].items())
    assert all(sha256(path) == digest for path, digest in spec['artifact_sources'].items())
    for group in spec['groups']:
        assert all(sha256(row['source_Dev_metrics_path']) == row['source_Dev_metrics_sha256']
                   for row in group['source_curve'])
    for case in spec['cases']:
        assert read(CASES / (case['id'] + '.json')) == case['config']
        assert case['config']['native_seed'] == 0
        assert signature(case['config']) == signature(case['source']['config'])
    return spec


def run():
    plan, work, protocol = context()
    spec = verify()
    path = work / 'native_remaining_test_backfill_state.json'
    lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists()
    cases = {row['id']: row for row in spec['cases']}
    dev_jobs = {key: dict(status='PENDING') for key in cases}
    test_jobs = {key: dict(status='PENDING') for key in cases}
    state = dict(status='WAITING_FOR_PARENT26_REPORT_AND_PROCESS_EXIT', controller_pid=os.getpid(),
        specification_sha256=sha256(SPEC), dev_jobs=dev_jobs, test_jobs=test_jobs,
        fixed_seed=0, additional_seed_work=False, GPU_work=False, current_visual_primary_unchanged=True,
        Test_accuracy_used_for_selection=False, goal_met=False, supported_innovations=0)
    dump(path, state)
    active = {}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', MFT26IR_ROOT=plan['data_root'],
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    try:
        while not predecessor_ready(read(work / 'native_common_test_backfill_state.json'), PARENT_REPORT.exists()):
            time.sleep(10)
        parent_report = read(PARENT_REPORT)
        assert parent_report['status'] == 'COMPLETE_FROZEN_NATIVE36_ALL26_TEST_BACKFILL'
        assert parent_report['specification_sha256'] == sha256(PARENT)
        state['parent_report_sha256'] = sha256(PARENT_REPORT)
        state['status'] = 'REPRODUCING_ALL50_ORIGINAL_DEV_BEHAVIORS'
        dump(path, state)
        for split, jobs in [('dev', dev_jobs), ('test', test_jobs)]:
            if split == 'test':
                assert all(j['status'] == 'COMPLETE' and j['source_proof']['all39metrics_tracks_official_CSV_exact']
                           for j in dev_jobs.values())
                state['status'] = 'RUNNING_ALL50_FROZEN_FORMAL_TEST_CASES'
                dump(path, state)
            while True:
                for key, proc in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    del active[key]
                    assert code == 0, jobs[key]['log']
                    jobs[key]['status'] = 'COMPLETE'
                    if split == 'dev':
                        jobs[key]['source_proof'] = parity(work / 'native_remaining_test_backfill_dev' / key,
                            Path(cases[key]['source']['source_Dev_metrics_path']).parent, protocol['dev'])
                for key, job in jobs.items():
                    if len(active) >= 2 or job['status'] != 'PENDING':
                        continue
                    verify()
                    assert shutil.disk_usage(work).free >= 40 * 2**30
                    assert int(subprocess.check_output(['du', '-s', '-B1', str(work)], text=True).split()[0]) < 64 * 2**30
                    output = work / ('native_remaining_test_backfill_' + split) / key
                    command = [sys.executable, '-u', 'tools/replay_perceptual_mft.py', '--split', split,
                        '--config', str(CASES / (key + '.json')), '--output', str(output)]
                    log = work / (key + '.' + split + '.native_remaining_backfill.log')
                    with log.open('xb', buffering=0) as stream:
                        proc = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    active[key] = proc
                    job.update(status='RUNNING', pid=proc.pid, command=command, log=str(log))
                dump(path, state)
                if not active:
                    assert all(j['status'] == 'COMPLETE' for j in jobs.values())
                    break
                time.sleep(5)
        state['status'] = 'COMPLETE_ALL50_REMAINING_NATIVE_TEST'
        dump(path, state)
        export()
    except Exception as error:
        for proc in active.values():
            if proc.poll() is None:
                proc.terminate()
        state.update(status='FAILED_PRESERVE_NATIVE_REMAINING_BACKFILL', error=repr(error))
        dump(path, state)
        raise


def export():
    from tools.summarize_pseudo_ir_comparison import detailed_metrics
    _, work, protocol = context()
    spec = verify()
    state_path = work / 'native_remaining_test_backfill_state.json'
    state = read(state_path)
    assert state['status'] == 'COMPLETE_ALL50_REMAINING_NATIVE_TEST'
    assert state['specification_sha256'] == sha256(SPEC)
    assert sha256(PARENT_REPORT) == state['parent_report_sha256']
    parent_rows = {r['id']: r for r in read(PARENT_REPORT)['rows']}
    rows, by_group = {}, {}
    destination = DOCS / 'perceptual_native_remaining_test_20261124'
    destination.mkdir(exist_ok=False)
    for case in spec['cases']:
        key = case['id']
        folder = work / 'native_remaining_test_backfill_test' / key
        data = read(folder / 'metrics.json')
        manifest = data['manifest']
        assert data['status'] == 'COMPLETE' and manifest['split'] == 'test' and manifest['frames'] == 15985
        assert manifest['sequences'] == protocol['historical_test'] and manifest['config'] == case['config']
        assert manifest['config_sha256'] == sha256(CASES / (key + '.json'))
        assert len(data['trackeval']) == 39 and not manifest['GT_inference_input'] and not manifest['future_frame_input']
        csv = folder / 'trackeval_output/pedestrian_detailed.csv'
        published = DOCS / 'perceptual_20261122_official_tables/native_remaining_test_backfill_test' / (key + '.csv')
        published.parent.mkdir(exist_ok=True)
        if not published.exists():
            shutil.copy2(csv, published)
        assert sha256(csv) == sha256(published)
        per_sequence = detailed_metrics(csv, protocol['historical_test'], data['trackeval'])
        assert sum(row['CLR_Frames'] for row in per_sequence.values()) == 15985
        record = dict(id=key, config=case['config'], metrics=data['trackeval'], per_sequence=per_sequence,
            manifest=manifest, metrics_sha256=sha256(folder / 'metrics.json'),
            official_detailed_csv_sha256=sha256(csv), official_detailed_csv_file=str(published.relative_to(ROOT)))
        dump(destination / (key + '.json'), record)
        assert (destination / (key + '.json')).stat().st_size < 5 * 2**20
        rows[key] = {k: record[k] for k in ['id', 'config', 'metrics', 'per_sequence',
            'metrics_sha256', 'official_detailed_csv_sha256', 'official_detailed_csv_file']}
        rows[key]['full_manifest_file'] = str((destination / (key + '.json')).relative_to(ROOT))
    for group in spec['groups']:
        row = rows[group['case_id']] if group['new_Test_required'] else parent_rows[group['delegated_parent_case']]
        assert signature(row['config']) == signature(group['selected']['config'])
        by_group[group['group_id']] = dict(case=row['id'], metrics=row['metrics'],
            original_Dev_HOTA=group['selected']['HOTA'], original_Dev_epoch=group['selected']['epoch'],
            delegated_to_parent26=not group['new_Test_required'])
    assert len(by_group) == 58
    report = dict(status='COMPLETE_ORIGINAL58_NATIVE_BEHAVIORS_SEED0_ALL7_TEST', rows=list(rows.values()),
        original58_behavior_results=by_group, original58_group_coverage_complete=True,
        specification_sha256=sha256(SPEC), complete_state_sha256=sha256(state_path),
        parent_report_sha256=state['parent_report_sha256'],
        source_Dev_proofs={k: j['source_proof'] for k, j in state['dev_jobs'].items()},
        scope=spec['scope'], reporting=spec['reporting'], fixed_seed=0, additional_seed_work=False,
        GPU_work=False, current_visual_primary_unchanged=True, scientific_assessment_required=True,
        Test_accuracy_used_for_selection=False,
        goal_met=False, supported_innovations=0)
    stem = DOCS / 'PERCEPTUAL_MFT_20261124_NATIVE_REMAINING_TEST_BACKFILL'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    dump(stem.with_suffix('.json'), report)
    lines = ['# Original native36 seed0 behavior coverage', '', spec['scope'], '',
        '| Original group | Dev selection | Dev HOTA | Test HOTA | Test IDF1 | Test AssA | Parent26 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for key, row in by_group.items():
        m = row['metrics']
        lines.append(f"| {key} | {row['original_Dev_epoch']} | {row['original_Dev_HOTA']:.3f} | "
                     f"{m['HOTA']:.3f} | {m['IDF1']:.3f} | {m['AssA']:.3f} | {row['delegated_to_parent26']} |")
    lines += ['', spec['selection'], '', spec['reporting'], '',
              'Completeness applies to the58 frozen behavior groups only. Other research requirements remain; '
              'no Test-based primary promotion, supported-innovation count or goal-completion claim follows automatically.']
    stem.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], original_groups=58, new_Test_cases=50)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run', 'export'])
    globals()[parser.parse_args().action]()
