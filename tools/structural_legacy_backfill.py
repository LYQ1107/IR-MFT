#!/usr/bin/env python3
"""Close the initial structural formal-Test gaps without changing the primary.

Register from the already complete original Dev, then queue behind the complete
full-Train pipeline. GPU work uses only physical 4/5. Existing prediction math,
ByteTrack, TrackEval and frozen checkpoints are reused without modification.
"""
import argparse
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

REGISTRY = ROOT / 'configs/perceptual_structural_legacy_backfill_20261124.json'
SPEC = ROOT / 'configs/perceptual_structural_legacy_backfill_test_20261124.json'
CASES = ROOT / 'configs/perceptual_structural_legacy_backfill_cases_20261124'
BASE = ROOT / 'configs/perceptual_structural_relative_20261123.json'
POLICY = ROOT / 'configs/perceptual_single_seed_policy_20261123.json'
DOCS = ROOT / 'docs/fusion_research'
PREDICTIONS = 'legacy_structural_backfill/structural_relative_predictions'
SOURCES = [
    'tools/structural_legacy_backfill.py', 'tools/extract_structural_relative.py',
    'tools/replay_structural_nocycle_test.py', 'tools/structural_relative_inference.py',
    'tools/build_correlation_cache.py', 'tools/indexed_postprocess.py',
    'tools/structural_grid.py', 'tools/structural_correspondence.py',
    'tools/structural_gate_observer.py', 'tools/correlation_geometry_checks.py',
    'tools/recover_correlation_formal_test.py', 'tools/eval_mft26ir.py',
    'tools/export_perceptual_official_tables.py',
    'tools/perceptual_gpu_handoff.py', 'configs/perceptual_gpu_handoff_20261123.json',
    'yolox/models/structural_relative_repair.py', 'yolox/models/frozen_head_repair.py',
    'yolox/models/yolo_head.py', 'yolox/tracker/foveated_regions.py',
    'yolox/tracker/priority_observations.py', 'yolox/tracker/byte_tracker.py',
    'configs/perceptual_mft_20261122.json', 'configs/cfp/protocol_v2.json',
    'configs/perceptual_structural_relative_20261123.json',
    'configs/perceptual_single_seed_policy_20261123.json',
    'configs/perceptual_correlation_repair_20261123.json',
]


def read(path):
    return json.loads(Path(path).read_text())


def context():
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    assert sha256(BASE) == '909eb6fae97ce7eb64c79a31ee93e76d713003c51cc7d9c13dbd276cf49519fd'
    assert sha256(POLICY) == '1f278840ccc0f60207f38fa4b31c492947f8fa62805979aae96913a92f0f6bd0'
    assert read(POLICY)['fixed_seed'] == 0 and not read(POLICY)['additional_seed_inference']
    return plan, Path(plan['work_root']), read(ROOT / plan['protocol'])


def verified_spec():
    spec = read(SPEC)
    assert spec['registry_sha256'] == sha256(REGISTRY)
    assert spec['config_sha256'] == sha256(BASE)
    assert spec['single_seed_user_policy_sha256'] == sha256(POLICY)
    assert all(sha256(ROOT / name) == digest for name, digest in spec['source_hashes'].items())
    assert all(row.get('seed', 0) == 0 for row in spec['models'])
    return spec


def register():
    plan, work, protocol = context()
    assert not SPEC.exists() and not CASES.exists()
    assert not (work / 'structural_legacy_backfill_selection_frozen.json').exists()
    registry = read(REGISTRY)
    dev_path = DOCS / 'PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_ANALYSIS.json'
    assert sha256(dev_path) == registry['source_Dev_analysis_sha256']
    dev_state = read(work / 'structural_relative_dev_state.json')
    assert dev_state['status'] == 'COMPLETE' and len(dev_state['jobs']) == 59
    assert all(row['status'] == 'COMPLETE' for row in dev_state['jobs'].values())
    dev = read(dev_path)
    models, cases, roles = [], [], {}

    def add(role, row, intervention=None):
        name = 'LB_' + role
        path = Path(row['checkpoint'])
        assert sha256(path) == row['checkpoint_sha256']
        model = dict(id=name, cue=row['cue'], variant=row['variant'], epoch=row['epoch'],
                     checkpoint=str(path), checkpoint_sha256=row['checkpoint_sha256'], seed=0,
                     expected_hard_cycle=True)
        if intervention:
            model['intervention'] = intervention
        else:
            suffix = '_raw' if row['policy'] == 'raw' else ''
            model['original_Dev_model_id'] = row['source_Dev_case'].removesuffix(suffix) if suffix else row['source_Dev_case']
        models.append(model)
        case = dict(model, id=name + '_' + row['policy'], model_id=name,
                    proposal_key=name + ('_raw' if row['policy'] == 'raw' else ''),
                    output_policy=row['policy'])
        cases.append(case)
        roles[role] = case['id']

    for row in registry['cases']:
        add(row['role'], row)
    original = dev['best_by_cue']['ir']
    cfg = original['config']
    assert original['id'] == 'SR_ir_full_s0_e06_raw' and cfg['variant'] == 'full'
    anchor = dict(checkpoint=cfg['checkpoint'], checkpoint_sha256=sha256(cfg['checkpoint']),
        cue='ir', variant='full', epoch=6, policy='raw', source_Dev_case=original['id'])
    add('restored_cycle_IR_reference', anchor)
    add('IR_image_roll', anchor, 'IR_image_spatial_roll')
    add('IR_unavailable', anchor, 'unavailable')
    models.append(dict(id='CG3_source_RGB', cue='rgb', seed=0, baseline=True))
    for label, fresh in [('source_RGB_fresh', True), ('source_RGB_canonical', False)]:
        case = dict(id='LB_' + label, model_id='CG3_source_RGB', proposal_key='CG3_source_RGB',
                    baseline=True, fresh_RGB=fresh, cue='rgb', seed=0, output_policy='baseline')
        roles[label] = case['id']
        cases.append(case)
    parent = DOCS / 'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_TEST.json'
    spec = dict(status='FROZEN_ORIGINAL59DEV_LEGACY_FORMAL_TEST_BACKFILL',
        registry_sha256=sha256(REGISTRY), config_sha256=sha256(BASE),
        single_seed_user_policy_sha256=sha256(POLICY), fixed_seed=0,
        models=models, cases=cases, roles=roles, produced_streams=15, source_zero_streams=9,
        source_hashes={name: sha256(ROOT / name) for name in SOURCES},
        original59Dev_analysis_sha256=sha256(dev_path),
        original59Dev_state_sha256=sha256(work / 'structural_relative_dev_state.json'),
        parent_Test_report_sha256=sha256(parent), parent_model_id='NT_ir_full_s0_e06',
        parent_reference_case='NT_cycle_restored_Dev_selected',
        required_sequences=protocol['historical_test'], frames_each=15985,
        selection='Four missing roles were frozen from original complete59Dev. No Test-based primary replacement.',
        current_primary_unchanged=True, additional_seed_work=False,
        scope='Initial full-cycle RGB/gray and full-cycle no-centering formal coverage only; '
              'not another architecture or evidence of a new multimodal contribution. HistoricalTest is exposed.',
        GPU_queue='Wait for complete full-Train fits,50Dev,runtime,allTest and official tables; physical4/5 only.',
        goal_met=False, supported_innovations=0)
    assert len(models) == 8 and len(cases) == 9 and len(roles) == 9
    dump(SPEC, spec)
    CASES.mkdir()
    for case in cases:
        dump(CASES / (case['id'] + '.json'), dict(case, fusion_plan_sha256=sha256(BASE),
            specification_sha256=sha256(SPEC), single_seed_user_policy_sha256=sha256(POLICY),
            producer_directory=PREDICTIONS))
    dump(work / 'structural_legacy_backfill_selection_frozen.json', dict(
        status='FROZEN_ORIGINAL_COMPLETE59DEV_CHOICES', specification_sha256=sha256(SPEC),
        registry_sha256=sha256(REGISTRY), complete_original59Dev_before_Test=True,
        current_primary_unchanged=True, fixed_seed=0))
    print(json.dumps(dict(status=spec['status'], cases=len(cases), models=len(models))))


def pid_finished(pid):
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return True
    try:
        return path.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    except FileNotFoundError:
        return True


def predecessor_ready(fits, dev, advance, test, finished=pid_finished):
    """A queue entry or state file alone does not prove the two GPUs are free."""
    states = [fits, dev, advance, test]
    if any('FAIL' in row.get('status', '') for row in states):
        raise RuntimeError('Preserve failed full-Train predecessor; do not steal GPUs or bypass its checks')
    if not (fits.get('status') == dev.get('status') == test.get('status') == 'COMPLETE'
            and advance.get('status') == 'COMPLETE_FULL_TRAIN_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT'):
        return False
    for data, key, count in [(fits, 'jobs', 6), (dev, 'jobs', 50), (test, 'producers', 7)]:
        rows = data.get(key, {})
        if len(rows) != count or not all(row.get('status') == 'COMPLETE' for row in rows.values()):
            return False
    if not test.get('jobs') or not all(r.get('status') == 'COMPLETE' for r in test['jobs'].values()):
        return False
    gpu_jobs = list(fits['jobs'].values()) + list(test['producers'].values())
    return all(row.get('pid') and finished(row['pid']) for row in gpu_jobs)


def require_predecessor(work):
    names = ['structural_full_train_fit_state.json', 'structural_full_train_dev_state.json',
             'structural_full_train_next_stage_state.json', 'structural_full_train_test_state.json']
    rows = [read(work / name) if (work / name).exists() else {} for name in names]
    return predecessor_ready(*rows)


def extract(args):
    import cv2
    import torch
    from tools.extract_structural_relative import produce
    from tools.structural_gate_observer import StructuralGateObserver, SCOPE
    from yolox.exp import get_exp
    from yolox.models.structural_relative_repair import StructuralRepair

    plan, work, protocol = context()
    assert require_predecessor(work), 'Current full-Train GPU work has priority'
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical in ['4', '5']
    spec = verified_spec()
    freeze = read(work / 'structural_legacy_backfill_selection_frozen.json')
    assert freeze['specification_sha256'] == sha256(SPEC) and freeze['complete_original59Dev_before_Test']
    assert args.sequence in protocol['dev' if args.split == 'dev' else 'historical_test']
    assert args.source_check in [0, 8] and args.limit_frames in [0, 32]
    assert not args.limit_frames or args.split == 'dev'
    if args.source_check:
        spec = dict(models=[dict(id='CG3_source_RGB', cue='rgb', baseline=True)] + [
            dict(id='LB_zero_' + cue + '_' + variant, cue=cue, variant=variant, zero_adapter=True, seed=0)
            for cue, variant in [('rgb', 'full'), ('gray', 'full'), ('ir', 'full'), ('ir', 'no_centering')]])
    config = read(BASE)
    parent = read(ROOT / 'configs/perceptual_correlation_repair_20261123.json')
    assert config['parent_plan_sha256'] == sha256(ROOT / 'configs/perceptual_correlation_repair_20261123.json')
    calibration_path = work / 'correlation_cache/calibration.json'
    calibration = read(calibration_path)
    assert calibration['config_sha256'] == config['parent_plan_sha256']
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    checkpoint = Path(parent['source_RGB_checkpoint'])
    assert sha256(checkpoint) == parent['source_RGB_sha256']
    model = get_exp(str(ROOT / 'exps/yolox_mft26ir_visible.py'), None).get_model()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model = model.cuda().half().eval().requires_grad_(False)
    loaded, information, needed = {}, {}, {'rgb'}
    architecture = config['architecture']
    observer = StructuralGateObserver()
    for row in spec['models']:
        name, cue = row['id'], row['cue']
        needed.add('ir_image_roll' if row.get('intervention') == 'IR_image_spatial_roll' else cue)
        if row.get('baseline'):
            loaded[name], information[name] = None, dict(row)
            continue
        adapter = StructuralRepair(latent=architecture['latent_channels'], radius=architecture['local_radius'],
            temperature=architecture['structural_temperature'], bound=architecture['feature_residual_bound_RMS_fraction'],
            risk_threshold=architecture['teacher_risk_threshold'], variant=row['variant'],
            tolerance=architecture['forward_backward_tolerance_cells'])
        assert all(level.cycle for level in adapter.levels), 'Both old variants keep the hard cycle'
        info = dict(row)
        if not row.get('zero_adapter'):
            path = Path(row['checkpoint'])
            assert sha256(path) == row['checkpoint_sha256']
            saved = torch.load(path, map_location='cpu', weights_only=False)
            meta = saved['metadata']
            assert saved['config_sha256'] == sha256(BASE) and saved['completed_epochs'] == row['epoch']
            assert (meta['cue'], meta['variant'], meta['seed']) == (cue, row['variant'], 0)
            assert meta['source_RGB_sha256'] == parent['source_RGB_sha256']
            assert meta['single_seed_user_policy_sha256'] == sha256(POLICY)
            adapter.load_state_dict(saved['adapter'], strict=True)
            info['metadata'] = meta
        assert sum(value.numel() for value in adapter.parameters()) == 145600
        loaded[name] = adapter.cuda().float().eval().requires_grad_(False)
        information[name] = info
        observer.attach(name, loaded[name])
    try:
        produce(args, physical, plan, config, BASE, POLICY, spec, None if args.source_check else sha256(SPEC),
            calibration, calibration_path, work / 'legacy_structural_backfill', model, loaded, information, needed)
    finally:
        observer.close()
    out = work / PREDICTIONS / args.split / args.sequence
    if args.source_check:
        out /= 'source_check_8'
    elif args.limit_frames:
        out /= 'prefix_check_32'
    meta = read(out / 'complete.json')
    meta['source_hashes'].update({name: sha256(ROOT / name) for name in [
        'tools/structural_legacy_backfill.py', 'tools/structural_gate_observer.py']})
    for name in loaded:
        statistics = observer.summary(name)
        if statistics:
            meta['models'][name]['gate_statistics'] = statistics
    meta['gate_statistics_scope'] = SCOPE
    dump(out / 'complete.json', meta)


def check_producer(folder, spec, frames, source=False, prefix=False):
    meta = read(folder / 'complete.json')
    assert meta['status'] == ('SMOKE_COMPLETE' if source or prefix else 'COMPLETE_STRUCTURAL_RELATIVE_OBSERVATIONS')
    assert meta['frames'] == frames and not meta['GT_read'] and not meta['future_frame_input']
    assert meta['config_sha256'] == spec['config_sha256']
    assert len(meta['outputs']) == (spec['source_zero_streams'] if source else spec['produced_streams'])
    assert meta['source_hashes'] == {name: spec['source_hashes'][name] for name in meta['source_hashes']}
    if source:
        assert len(meta['fresh_RGB_parity']) == frames
        assert all(row['array_exact'] for row in meta['fresh_RGB_parity'])
    else:
        assert meta['specification_sha256'] == sha256(SPEC)
    assert all(sha256(info['path']) == info['sha256'] for info in meta['outputs'].values())
    return meta


def packet(path):
    import numpy as np
    with np.load(path) as z:
        return {name: z[name] for name in ['frame_ids', 'offsets', 'boxes', 'scores']}


def prediction_proof(folder, seq, spec, plan, work, prefix=False):
    import numpy as np
    from tools.correlation_geometry_checks import compare_packets
    from tools.recover_correlation_formal_test import aligned_source_audit
    split = 'dev' if prefix else 'test'
    canonical = Path(plan['frozen_detection_sources']['rgb_' + split]) / 'detections' / (seq + '.npz')
    off = next(m['id'] for m in spec['models'] if m.get('intervention') == 'unavailable')
    proof = dict(unavailable_raw=compare_packets(folder / (off + '_raw.npz'), folder / 'CG3_source_RGB.npz'),
        unavailable_priority=compare_packets(folder / (off + '.npz'), canonical, 32 if prefix else None))
    if prefix:
        proof['fresh_original_Dev'] = compare_packets(folder / 'CG3_source_RGB.npz',
            work / 'structural_relative_predictions/dev' / seq / 'CG3_source_RGB.npz', 32)
        proof['source8_prefix'] = compare_packets(folder.parent / 'source_check_8/CG3_source_RGB.npz',
            folder / 'CG3_source_RGB.npz', 8)
        for model in spec['models']:
            if model.get('original_Dev_model_id'):
                for suffix in ['', '_raw']:
                    proof[model['id'] + suffix] = compare_packets(folder / (model['id'] + suffix + '.npz'),
                        work / 'structural_relative_predictions/dev' / seq / (model['original_Dev_model_id'] + suffix + '.npz'), 32)
    else:
        proof['fresh_RGB_tail'] = aligned_source_audit(packet(folder / 'CG3_source_RGB.npz'), packet(canonical), tail_frames=4)
        proof['source8_full'] = compare_packets(folder / 'source_check_8/CG3_source_RGB.npz', folder / 'CG3_source_RGB.npz', 8)
        anchor = next(m['id'] for m in spec['models'] if m['id'] == 'LB_restored_cycle_IR_reference')
        for suffix in ['', '_raw']:
            proof['parent_IR' + suffix] = compare_packets(folder / (anchor + suffix + '.npz'),
                work / 'nocycle_formal_test/structural_relative_predictions/test' / seq / (spec['parent_model_id'] + suffix + '.npz'))
    count = 0
    for model in spec['models']:
        if model.get('baseline'):
            continue
        with np.load(folder / (model['id'] + '_raw.npz')) as z:
            assert len(z['boxes']) == len(z['anchor_indices']) == len(z['rgb_reference_scores'])
            assert ((z['anchor_indices'] >= 0) & (z['anchor_indices'] < 23625)).all()
            assert ((z['rgb_reference_scores'] >= 0) & (z['rgb_reference_scores'] <= 1)).all()
        count += 1
    proof['raw_anchor_provenance_streams'] = count
    return proof


def run():
    plan, work, protocol = context()
    spec = verified_spec()
    state_path = work / 'structural_legacy_backfill_state.json'
    lock = state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not state_path.exists()
    jobs = {row['id']: dict(status='PENDING', config=str(CASES / (row['id'] + '.json')),
        sha256=sha256(CASES / (row['id'] + '.json'))) for row in spec['cases']}
    state = dict(status='WAITING_FOR_COMPLETE_FULL_TRAIN_PIPELINE', controller_pid=os.getpid(),
        specification_sha256=sha256(SPEC), source_hashes=spec['source_hashes'], jobs=jobs,
        physical_gpus=['4', '5'], fixed_seed=0, additional_seed_work=False,
        Test_accuracy_read=False, current_primary_unchanged=True, goal_met=False, supported_innovations=0)
    dump(state_path, state)
    active = {}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', MFT26IR_ROOT=plan['data_root'],
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')

    def launch(command, name, gpu=''):
        verified_spec()
        assert shutil.disk_usage(work).free >= 40 * 2**30
        assert int(subprocess.check_output(['du', '-s', '-B1', str(work)], text=True).split()[0]) < 64 * 2**30
        log = work / (name + '.legacy_backfill.log')
        with log.open('xb', buffering=0) as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=dict(environment, CUDA_VISIBLE_DEVICES=gpu),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        return proc, dict(pid=proc.pid, command=command, log=str(log), physical_gpu=gpu)

    try:
        while not require_predecessor(work):
            time.sleep(10)
        verified_spec()
        state['status'] = 'SOURCE8_AND_ORIGINAL_TRAINED32_PARITY'
        dump(state_path, state)
        for seq, gpu in zip(protocol['dev'], ['4', '5']):
            for mode in ['source', 'prefix']:
                command = [sys.executable, '-u', str(Path(__file__).relative_to(ROOT)), 'extract',
                           '--split', 'dev', '--sequence', seq]
                command += ['--source-check', '8'] if mode == 'source' else ['--limit-frames', '32']
                proc, info = launch(command, seq + '_dev_' + mode, gpu)
                state['engineering_job'] = info
                dump(state_path, state)
                assert proc.wait() == 0, info['log']
                folder = work / PREDICTIONS / 'dev' / seq / ('source_check_8' if mode == 'source' else 'prefix_check_32')
                check_producer(folder, spec, 8 if mode == 'source' else 32,
                               source=mode == 'source', prefix=mode == 'prefix')
            state.setdefault('original_Dev_prefix_proofs', {})[seq] = prediction_proof(folder, seq, spec, plan, work, True)
            dump(state_path, state)
        lengths = {seq: len(packet(Path(plan['frozen_detection_sources']['rgb_test']) / 'detections' / (seq + '.npz'))['frame_ids'])
                   for seq in protocol['historical_test']}
        assert len(lengths) == 7 and sum(lengths.values()) == 15985
        producers = {seq: dict(status='PENDING', frames=lengths[seq]) for seq in sorted(lengths, key=lambda s: -lengths[s])}
        state.update(status='PRODUCING_MISSING_FORMAL_ALL7_TEST_WITHOUT_GT', producers=producers)
        while True:
            for gpu, (seq, mode, proc) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                del active[gpu]
                assert code == 0, producers[seq]['log']
                folder = work / PREDICTIONS / 'test' / seq
                if mode == 'source':
                    check_producer(folder / 'source_check_8', spec, 8, True)
                    producers[seq]['status'] = 'SOURCE_PASSED'
                else:
                    check_producer(folder, spec, lengths[seq])
                    producers[seq].update(status='COMPLETE', complete_sha256=sha256(folder / 'complete.json'),
                        source_proof=prediction_proof(folder, seq, spec, plan, work))
            for gpu in ['4', '5']:
                if gpu in active:
                    continue
                candidates = [s for s, r in producers.items() if r['status'] == 'SOURCE_PASSED']
                candidates += [s for s, r in producers.items() if r['status'] == 'PENDING']
                if not candidates:
                    continue
                seq = candidates[0]
                mode = 'source' if producers[seq]['status'] == 'PENDING' else 'full'
                command = [sys.executable, '-u', str(Path(__file__).relative_to(ROOT)), 'extract', '--split', 'test', '--sequence', seq]
                if mode == 'source':
                    command += ['--source-check', '8']
                proc, info = launch(command, seq + '_test_' + mode, gpu)
                active[gpu] = (seq, mode, proc)
                producers[seq].update(info, status=mode.upper() + '_RUNNING')
            dump(state_path, state)
            if not active:
                assert all(r['status'] == 'COMPLETE' for r in producers.values())
                break
            time.sleep(5)
        state['status'] = 'REPLAYING_ALL_FROZEN_FORMAL_CASES'
        while True:
            for key, proc in list(active.items()):
                code = proc.poll()
                if code is not None:
                    jobs[key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                    del active[key]
            for key, job in jobs.items():
                if len(active) >= 2 or any(r['status'] == 'FAILED' for r in jobs.values()):
                    break
                if job['status'] != 'PENDING':
                    continue
                assert sha256(job['config']) == job['sha256']
                assert read(job['config'])['specification_sha256'] == sha256(SPEC)
                command = [sys.executable, '-u', 'tools/replay_structural_nocycle_test.py', '--split', 'test',
                    '--config', job['config'], '--output', str(work / 'structural_legacy_backfill_test' / key)]
                proc, info = launch(command, key + '_replay')
                active[key] = proc
                job.update(info, status='RUNNING')
            dump(state_path, state)
            if not active:
                assert all(r['status'] == 'COMPLETE' for r in jobs.values())
                break
            time.sleep(5)
        subprocess.run([sys.executable, 'tools/export_perceptual_official_tables.py'], cwd=ROOT, check=True)
        state['status'] = 'COMPLETE_ALL9_FORMAL_BACKFILL_CASES'
        dump(state_path, state)
        export()
    except Exception as error:
        # Stop only children owned by this controller; preserve every artifact.
        for item in active.values():
            proc = item[-1] if isinstance(item, tuple) else item
            if proc.poll() is None:
                proc.terminate()
        state.update(status='FAILED_PRESERVE_LEGACY_BACKFILL', error=repr(error))
        dump(state_path, state)
        raise


def export():
    from tools.summarize_pseudo_ir_comparison import detailed_metrics
    plan, work, protocol = context()
    spec = verified_spec()
    state_path = work / 'structural_legacy_backfill_state.json'
    state = read(state_path)
    assert state['status'] == 'COMPLETE_ALL9_FORMAL_BACKFILL_CASES'
    assert state['specification_sha256'] == sha256(SPEC)
    rows = {}
    for case in spec['cases']:
        key = case['id']
        folder = work / 'structural_legacy_backfill_test' / key
        path = folder / 'metrics.json'
        data = read(path)
        manifest = data['manifest']
        assert data['status'] == 'COMPLETE' and manifest['frames'] == 15985
        assert manifest['sequences'] == protocol['historical_test'] and manifest['split'] == 'test'
        assert manifest['config_sha256'] == state['jobs'][key]['sha256'] == sha256(state['jobs'][key]['config'])
        csv = folder / 'trackeval_output/pedestrian_detailed.csv'
        published_csv = DOCS / 'perceptual_20261122_official_tables/structural_legacy_backfill_test' / (key + '.csv')
        assert sha256(published_csv) == sha256(csv)
        per_sequence = detailed_metrics(csv, protocol['historical_test'], data['trackeval'])
        assert len(data['trackeval']) == 39 and sum(v['CLR_Frames'] for v in per_sequence.values()) == 15985
        rows[key] = dict(id=key, config=manifest['config'], metrics=data['trackeval'], per_sequence=per_sequence,
            manifest=manifest, metrics_sha256=sha256(path), official_detailed_csv_sha256=sha256(csv),
            official_detailed_csv_file=str(published_csv.relative_to(ROOT)))
    roles = spec['roles']
    off, fresh = [rows[roles[r]] for r in ['IR_unavailable', 'source_RGB_fresh']]
    assert off['metrics'] == fresh['metrics']
    assert off['official_detailed_csv_sha256'] == fresh['official_detailed_csv_sha256']
    parent_path = DOCS / 'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_TEST.json'
    assert sha256(parent_path) == spec['parent_Test_report_sha256']
    parent = next(r for r in read(parent_path)['rows'] if r['id'] == spec['parent_reference_case'])
    anchor = rows[roles['restored_cycle_IR_reference']]
    assert anchor['metrics'] == parent['metrics']
    assert anchor['official_detailed_csv_sha256'] == parent['official_detailed_csv_sha256']
    for seq in protocol['historical_test']:
        assert off['manifest']['coverage'][seq]['output_sha256'] == fresh['manifest']['coverage'][seq]['output_sha256']
        assert anchor['manifest']['coverage'][seq]['output_sha256'] == parent['manifest']['coverage'][seq]['output_sha256']
        proof = state['producers'][seq]['source_proof']
        assert proof['parent_IR']['all4_arrays_exact'] and proof['parent_IR_raw']['all4_arrays_exact']
    report = dict(status='COMPLETE_INITIAL_STRUCTURAL_FORMAL_ALL7_TEST_BACKFILL', rows=list(rows.values()), roles=roles,
        specification_sha256=sha256(SPEC), registry_sha256=sha256(REGISTRY), complete_state_sha256=sha256(state_path),
        source_audits={seq: row['source_proof'] for seq, row in state['producers'].items()},
        original_Dev_prefix_proofs=state['original_Dev_prefix_proofs'],
        parent_IR_predictions_tracks_all39metrics_CSV_exact=True, unavailable_exact_to_fresh=True,
        fixed_seed=0, additional_seed_work=False, current_primary_unchanged=True,
        scope=spec['scope'], source_sha256=sha256(__file__), goal_met=False, supported_innovations=0)
    stem = DOCS / 'PERCEPTUAL_MFT_20261124_STRUCTURAL_LEGACY_BACKFILL_TEST'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    dump(stem.with_suffix('.json'), report)
    lines = ['# Initial structural formal Test backfill', '', spec['scope'], '',
        '| Role | HOTA | IDF1 | DetA | AssA | MOTA | IDSW |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for role, key in roles.items():
        m = rows[key]['metrics']
        lines.append(f"| {role} | {m['HOTA']:.3f} | {m['IDF1']:.3f} | {m['DetA']:.3f} | "
                     f"{m['AssA']:.3f} | {m['MOTA']:.3f} | {m['IDSW']:.0f} |")
    lines += ['', 'All9 cases cover7 sequences/15985frames; all39 combined metrics, per-sequence core values and officialCSV hashes are retained.',
              'Current primary remains unchanged. No goal completion, new architecture or contribution is inferred from this coverage backfill.',
              'Shared producer runtime and cached replay FPS are not individual-method end-to-end FPS.']
    stem.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], rows=len(rows))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run', 'extract', 'export'])
    parser.add_argument('--sequence')
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--source-check', type=int, default=0)
    parser.add_argument('--limit-frames', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'extract':
        assert args.sequence
        extract(args)
    else:
        globals()[args.action]()


if __name__ == '__main__':
    main()
