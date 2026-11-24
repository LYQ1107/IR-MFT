#!/usr/bin/env python3
"""Frozen native36 robust/consensus formal Test, using existing cached evidence.

No GPU work or new fitting. Reproduce all eight original Dev choices first;
then evaluate every normal/unavailable/mismatched seed0 case on all seven Test
sequences. This closes a defined subset of the formal-coverage backlog.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256

DOCS = ROOT / 'docs/fusion_research'
SPEC = ROOT / 'configs/perceptual_native_common_test_backfill_20261124.json'
CASES = ROOT / 'configs/perceptual_native_common_test_cases_20261124'
DECISION = DOCS / 'PERCEPTUAL_MFT_20261123_NATIVE_COMMON36_DECISION.json'


def read(path):
    return json.loads(Path(path).read_text())


def context():
    path = ROOT / 'configs/perceptual_mft_20261122.json'
    assert sha256(path) == '388e636d45840f3da8eae44cde2efc4731eeceadc7c169a39f19bf30488ecc03'
    plan = read(path)
    return plan, Path(plan['work_root']), read(ROOT / plan['protocol'])


def choices(decision):
    assert decision['status'] == 'STOP_ALL_MATCHED_HEADS' and decision['epoch'] == 36 and decision['stopped_at_cap']
    selected = []
    for row in decision['rows']:
        if row['seed'] != 0:
            continue
        curve = row['curve']
        assert sorted(p['epoch'] for p in curve) == [6, 12, 18, 24, 30, 36]
        best = min(curve, key=lambda p: (-p['HOTA'], p['epoch']))
        assert best == row['best']
        selected.append(row)
    identities = {(r['algorithm'], r['frontend'], r['cue']) for r in selected}
    assert identities == {(a, f, c) for a in ['robust', 'consensus'] for f in ['new', 'rgb'] for c in ['ir', 'rgb']}
    return sorted(selected, key=lambda r: (r['algorithm'], r['frontend'], r['cue']))


def register():
    plan, work, protocol = context()
    assert not SPEC.exists() and not CASES.exists()
    curves = choices(read(DECISION))
    cases, selected, checkpoint_sources, artifacts = [], [], {}, {}
    for row in curves:
        best = row['best']
        path = Path(best['path'])
        assert sha256(path) == best['sha256']
        data = read(path)
        manifest = data['manifest']
        assert data['status'] == 'COMPLETE' and manifest['split'] == 'dev' and manifest['frames'] == 4834
        assert manifest['sequences'] == protocol['dev'] and data['trackeval']['HOTA'] == best['HOTA']
        assert len(data['trackeval']) == 39
        config = manifest['config']
        assert config['native_seed'] == 0 and config['native_cue'] == row['cue'] and config['detection_arm'] == row['frontend']
        assert manifest['native_query']['epoch'] == best['epoch']
        assert '/native_robust_query/' in config['native_query_checkpoint']
        assert not config.get('disable_native_increment') and not config.get('shuffle_native_cue')
        for key in ['native_query_checkpoint', 'association_checkpoint']:
            checkpoint_sources[config[key]] = sha256(config[key])
        assert checkpoint_sources[config['native_query_checkpoint']] == manifest['native_query_checkpoint_sha256']
        prefix = f"NC36_{row['algorithm']}_{row['frontend']}_{row['cue']}"
        selected.append(dict(id=prefix + '_normal', algorithm=row['algorithm'], frontend=row['frontend'],
            cue=row['cue'], epoch=best['epoch'], source_Dev_HOTA=best['HOTA'],
            source_Dev_metrics_path=str(path), source_Dev_metrics_sha256=sha256(path)))
        for intervention in ['normal', 'unavailable', 'mismatched']:
            cfg = copy.deepcopy(config)
            cfg['id'] = prefix + '_' + intervention
            cfg['disable_native_increment'] = intervention == 'unavailable'
            cfg['shuffle_native_cue'] = intervention == 'mismatched'
            cases.append(dict(id=cfg['id'], algorithm=row['algorithm'], frontend=row['frontend'], cue=row['cue'],
                intervention=intervention, config=cfg, source_Dev_normal_case=prefix + '_normal'))
    for arm in ['rgb', 'new']:
        cfg = dict(id='NC36_' + arm + '_ByteTrack', backend='byte', detection_arm=arm)
        cases.append(dict(id=cfg['id'], config=cfg, frontend=arm, intervention='ByteTrack'))
    # Unchanged source code and cached-image manifests are pinned before accuracy.
    source_names = [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'yolox/tracker').glob('*.py'))]
    source_names += ['tools/native_common_test_backfill.py', 'tools/replay_perceptual_mft.py',
        'tools/native_query_replay.py', 'tools/eval_mft26ir.py', 'tools/run_pseudo_ir_comparison.py',
        'tools/summarize_pseudo_ir_comparison.py', 'tools/export_perceptual_official_tables.py',
        'tools/extract_native_eval_features.py', 'tools/native_cue_io.py', 'tools/native_neck_io.py',
        'configs/perceptual_mft_20261122.json', 'configs/cfp/protocol_v2.json',
        'configs/perceptual_single_seed_policy_20261123.json',
        'configs/perceptual_native_query_20261123.json', 'configs/perceptual_native_robust_query_20261123.json']
    for split, sequences in [('dev', protocol['dev']), ('test', protocol['historical_test'])]:
        for seq in sequences:
            for folder in [work / 'features' / split / seq, work / 'transport' / split / seq]:
                meta_path = folder / 'complete.json'
                assert read(meta_path)['status'] == 'COMPLETE'
                artifacts[str(meta_path)] = sha256(meta_path)
            for cue in ['rgb', 'ir']:
                meta_path = work / 'native_eval_features' / split / seq / cue / 'complete.json'
                meta = read(meta_path)
                assert meta['status'] == 'COMPLETE_NATIVE_CURRENT_FEATURES' and not meta['GT_read'] and not meta['future_frame_input']
                assert meta['plan_sha256'] == sha256(ROOT / 'configs/perceptual_mft_20261122.json')
                artifacts[str(meta_path)] = sha256(meta_path)
    for cue in ['rgb', 'ir']:
        for path in [work / 'native_train_features' / (cue + '_s0') / 'complete.json',
                     work / 'native_alignment' / (cue + '_s0.npz')]:
            artifacts[str(path)] = sha256(path)
    artifacts[str(work / 'native_alignment/complete.json')] = sha256(work / 'native_alignment/complete.json')
    spec = dict(status='FROZEN_NATIVE36_COMMON_DEV_SELECTED_ALL8_SEED0_WITH_INTERVENTIONS',
        decision_sha256=sha256(DECISION), fixed_seed=0, additional_seed_work=False,
        selected=selected, cases=cases, checkpoint_sources=checkpoint_sources, artifact_sources=artifacts,
        source_hashes={name: sha256(ROOT / name) for name in source_names},
        comparison_cases=26, source_Dev_reproduction_cases=8, required_Test_sequences=protocol['historical_test'],
        required_Test_frames_each=15985, GPU_work=False, maximum_CPU_workers=2,
        current_visual_primary_unchanged=True, supported_innovations=0, goal_met=False,
        scope='Two already trained native36 query/memory backends, both frozen detector frontends and both '
              'matched nativeRGB/NovemberIR cues. Each original complete6-snapshot Dev curve selects its own '
              'checkpoint before this Test. All normal/off/current-frame-shuffled cases run regardless of '
              'Dev sign. No Test-based promotion; historicalTest exposed. Other formal families remain outside this subset.',
        unavailable_scope='Disables only the native residual. Preserves the complete learned RGB real/null prior '
              'and backend lifecycle; it is not assumed equal to plain ByteTrack.',
        compute_scope='Existing complete current-frame encoded features and point packets, no new encoding, '
              'no extra seeds and no GPU processes. Cached replay FPS excludes detector/encoders/points.')
    assert len(cases) == 26 and len(selected) == 8
    dump(SPEC, spec)
    CASES.mkdir()
    for case in cases:
        dump(CASES / (case['id'] + '.json'), case['config'])
    print(json.dumps(dict(status=spec['status'], cases=26, GPU_work=False)))


def verify():
    spec = read(SPEC)
    assert spec['decision_sha256'] == sha256(DECISION)
    assert all(sha256(ROOT / p) == h for p, h in spec['source_hashes'].items())
    assert all(sha256(p) == h for p, h in spec['checkpoint_sources'].items())
    assert all(sha256(p) == h for p, h in spec['artifact_sources'].items())
    for case in spec['cases']:
        assert read(CASES / (case['id'] + '.json')) == case['config']
        assert case['config'].get('native_seed', 0) == 0
    return spec


def parity(current, original, sequences):
    a, b = read(current / 'metrics.json'), read(original / 'metrics.json')
    for data in [a, b]:
        manifest = data['manifest']
        assert data['status'] == 'COMPLETE' and manifest['split'] == 'dev' and manifest['frames'] == 4834
        assert manifest['sequences'] == sequences and not manifest['GT_inference_input'] and not manifest['future_frame_input']
        assert len(data['trackeval']) == 39
    assert a['trackeval'] == b['trackeval']
    assert sha256(current / 'trackeval_output/pedestrian_detailed.csv') == sha256(original / 'trackeval_output/pedestrian_detailed.csv')
    for seq in sequences:
        assert sha256(current / 'track_results' / (seq + '.txt')) == sha256(original / 'track_results' / (seq + '.txt'))
    return dict(all39metrics_tracks_official_CSV_exact=True,
        current_metrics_sha256=sha256(current / 'metrics.json'), original_metrics_sha256=sha256(original / 'metrics.json'))


def run():
    plan, work, protocol = context()
    spec = verify()
    path = work / 'native_common_test_backfill_state.json'
    lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists()
    normal = {row['id']: row for row in spec['selected']}
    dev_jobs = {key: dict(status='PENDING') for key in normal}
    test_jobs = {row['id']: dict(status='PENDING') for row in spec['cases']}
    state = dict(status='REPRODUCING_ALL8_ORIGINAL_COMPLETE_DEV_CHOICES', controller_pid=os.getpid(),
        specification_sha256=sha256(SPEC), dev_jobs=dev_jobs, test_jobs=test_jobs,
        GPU_work=False, fixed_seed=0, additional_seed_work=False, Test_accuracy_read=False,
        current_visual_primary_unchanged=True, goal_met=False, supported_innovations=0)
    dump(path, state)
    active = {}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', MFT26IR_ROOT=plan['data_root'],
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    try:
        for split, jobs in [('dev', dev_jobs), ('test', test_jobs)]:
            if split == 'test':
                assert all(row['status'] == 'COMPLETE' and row['source_proof']['all39metrics_tracks_official_CSV_exact']
                           for row in dev_jobs.values())
                state['status'] = 'RUNNING_ALL26_FROZEN_FORMAL_TEST_CASES'
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
                        jobs[key]['source_proof'] = parity(work / 'native_common_test_backfill_dev' / key,
                            Path(normal[key]['source_Dev_metrics_path']).parent, protocol['dev'])
                for key, job in jobs.items():
                    if len(active) >= 2 or job['status'] != 'PENDING':
                        continue
                    verify()
                    assert shutil.disk_usage(work).free >= 40 * 2**30
                    assert int(subprocess.check_output(['du', '-s', '-B1', str(work)], text=True).split()[0]) < 64 * 2**30
                    command = [sys.executable, '-u', 'tools/replay_perceptual_mft.py', '--split', split,
                        '--config', str(CASES / (key + '.json')), '--output', str(work / ('native_common_test_backfill_' + split) / key)]
                    log = work / (key + '.' + split + '.native_common_backfill.log')
                    with log.open('xb', buffering=0) as stream:
                        proc = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    active[key] = proc
                    job.update(status='RUNNING', pid=proc.pid, command=command, log=str(log))
                dump(path, state)
                if not active:
                    assert all(row['status'] == 'COMPLETE' for row in jobs.values())
                    break
                time.sleep(5)
        state['status'] = 'COMPLETE_ALL26_FROZEN_NATIVE36_TEST'
        dump(path, state)
        export()
    except Exception as error:
        for proc in active.values():
            if proc.poll() is None:
                proc.terminate()
        state.update(status='FAILED_PRESERVE_NATIVE_COMMON_BACKFILL', error=repr(error))
        dump(path, state)
        raise


def export():
    from tools.summarize_pseudo_ir_comparison import detailed_metrics
    plan, work, protocol = context()
    spec = verify()
    state_path = work / 'native_common_test_backfill_state.json'
    state = read(state_path)
    assert state['status'] == 'COMPLETE_ALL26_FROZEN_NATIVE36_TEST' and state['specification_sha256'] == sha256(SPEC)
    assert all(row['status'] == 'COMPLETE' for row in state['test_jobs'].values())
    rows = {}
    for case in spec['cases']:
        key = case['id']
        folder = work / 'native_common_test_backfill_test' / key
        data = read(folder / 'metrics.json')
        manifest = data['manifest']
        assert data['status'] == 'COMPLETE' and manifest['split'] == 'test' and manifest['frames'] == 15985
        assert manifest['sequences'] == protocol['historical_test'] and manifest['config'] == case['config']
        assert manifest['config_sha256'] == sha256(CASES / (key + '.json'))
        assert len(data['trackeval']) == 39 and not manifest['GT_inference_input'] and not manifest['future_frame_input']
        csv = folder / 'trackeval_output/pedestrian_detailed.csv'
        published = DOCS / 'perceptual_20261122_official_tables/native_common_test_backfill_test' / (key + '.csv')
        published.parent.mkdir(exist_ok=True)
        if published.exists():
            assert sha256(csv) == sha256(published)
        else:
            shutil.copy2(csv, published)
        assert sha256(csv) == sha256(published)
        per_sequence = detailed_metrics(csv, protocol['historical_test'], data['trackeval'])
        assert sum(row['CLR_Frames'] for row in per_sequence.values()) == 15985
        rows[key] = dict(id=key, config=case['config'], metrics=data['trackeval'], per_sequence=per_sequence,
            manifest=manifest, metrics_sha256=sha256(folder / 'metrics.json'),
            official_detailed_csv_sha256=sha256(csv), official_detailed_csv_file=str(published.relative_to(ROOT)))
    comparisons = []
    for algorithm in ['robust', 'consensus']:
        for frontend in ['new', 'rgb']:
            prefix = f'NC36_{algorithm}_{frontend}_'
            ir, rgb = rows[prefix + 'ir_normal'], rows[prefix + 'rgb_normal']
            comparisons.append(dict(algorithm=algorithm, frontend=frontend,
                IR_minus_matched_RGB_HOTA=ir['metrics']['HOTA'] - rgb['metrics']['HOTA'],
                IR_minus_plain_direct_RGB_HOTA=ir['metrics']['HOTA'] - rows['NC36_rgb_ByteTrack']['metrics']['HOTA'],
                IR_unavailable_HOTA=rows[prefix + 'ir_unavailable']['metrics']['HOTA'],
                IR_mismatched_HOTA=rows[prefix + 'ir_mismatched']['metrics']['HOTA']))
    report = dict(status='COMPLETE_FROZEN_NATIVE36_ALL26_TEST_BACKFILL', rows=list(rows.values()),
        comparisons=comparisons, specification_sha256=sha256(SPEC), complete_state_sha256=sha256(state_path),
        Dev_reproduction_proofs={k: row['source_proof'] for k, row in state['dev_jobs'].items()},
        scope=spec['scope'], compute_scope=spec['compute_scope'], unavailable_scope=spec['unavailable_scope'],
        fixed_seed=0, additional_seed_work=False, GPU_work=False, current_visual_primary_unchanged=True,
        goal_met=False, supported_innovations=0, scientific_assessment_required=True)
    stem = DOCS / 'PERCEPTUAL_MFT_20261124_NATIVE_COMMON_TEST_BACKFILL'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    dump(stem.with_suffix('.json'), report)
    lines = ['# Native36 robust and consensus full Test backfill', '', spec['scope'], '',
        '| Case | HOTA | IDF1 | DetA | AssA | MOTA | IDSW |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key, row in rows.items():
        m = row['metrics']
        lines.append(f"| {key} | {m['HOTA']:.3f} | {m['IDF1']:.3f} | {m['DetA']:.3f} | "
                     f"{m['AssA']:.3f} | {m['MOTA']:.3f} | {m['IDSW']:.0f} |")
    lines += ['', spec['unavailable_scope'], '', spec['compute_scope'], '',
              'All26 cases cover7Test sequences/15985frames. No partial-case winner selection or automatic goal claim.']
    stem.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], cases=len(rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run', 'export'])
    globals()[parser.parse_args().action]()
