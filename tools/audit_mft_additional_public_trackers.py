#!/usr/bin/env python3
"""Fixed CPU BoT-SORT/HybridSORT comparisons on the existing detector streams.

No detector, appearance network, fusion fit, GPU, or model selection is run.
BoT-SORT retains its demo ORB camera compensation. HybridSORT retains its
MOT17 ablation preset, including HMIoU, TCM and low-confidence association.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_mft_public_trackers import packets, frame_detections, read, sha, write, headless_metrics

SPEC = ROOT / 'configs/audit_mft_additional_public_trackers_20261126.json'
SCALAR_COMPATIBILITY_PROOF = None


def context():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('The additional public comparisons are CPU only')
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    np.random.seed(0)
    os.nice(10)
    spec = read(SPEC)
    dependency = spec['isolated_CPU_dependency']
    if dependency['target'] not in sys.path:
        sys.path.append(dependency['target'])
    import lap
    if lap.__version__ != dependency['version'] or not Path(lap.__file__).resolve().is_relative_to(
            Path(dependency['target']).resolve()):
        raise ValueError('Use the pinned isolated LAP dependency, not another environment')
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    work = Path(plan['work_root'])
    from tools.neck72_runtime import require_host_memory, require_storage
    require_host_memory(2 * 2**30, maximum_workers=1)
    require_storage(work, growth_bytes=256 * 2**20)
    return spec, plan, work, work / spec['output_directory']


def sources(spec):
    if sha(spec['reuse_detector_registration']) != spec['reuse_detector_registration_sha256']:
        raise ValueError('Original completed detector registration changed')
    original = read(spec['reuse_detector_registration'])
    result = dict(original['source_hashes'])
    result[str(Path(__file__))] = sha(Path(__file__))
    result[str(SPEC)] = sha(SPEC)
    result[spec['reuse_detector_registration']] = sha(spec['reuse_detector_registration'])
    dependency = spec['isolated_CPU_dependency']
    for name, row in dependency['files'].items():
        path = Path(dependency['target']) / name
        if sha(path) != row['sha256']:
            raise ValueError('Pinned isolated CPU dependency changed')
        result[str(path)] = row['sha256']
    for reference in spec['references'].values():
        root = Path(reference['root'])
        actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != reference['commit']:
            raise ValueError('Official reference revision changed')
        for name, expected in reference['files'].items():
            path = root / name
            if sha(path) != expected:
                raise ValueError('Official reference source changed: ' + str(path))
            result[str(path)] = expected
    if any(sha(path) != digest for path, digest in result.items()):
        raise ValueError('Frozen original detector/evaluator or comparison sources changed')
    return result


def module_file(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    sys.modules[name] = module
    loader.loader.exec_module(module)
    return module


def tracker(method, spec):
    import cv2
    import numpy as np
    cv2.setRNGSeed(0)
    for name in spec['numpy_legacy_aliases']:
        if name not in np.__dict__:
            setattr(np, name, {'float': float, 'int': int, 'bool': bool}[name])
    settings = spec['trackers'][method]
    reference = Path(spec['references'][method]['root'])
    if method == 'botsort':
        # The official source imports the optional ReID class unconditionally.
        # Block its unused dependency without editing association/GMC sources.
        optional = types.ModuleType('fast_reid.fast_reid_interfece')
        class DisabledAppearanceNetwork:
            def __init__(self, *args, **kwargs):
                raise RuntimeError('Appearance-network use is forbidden in this no-ReID comparison')
        optional.FastReIDInterface = DisabledAppearanceNetwork
        if 'fast_reid' not in sys.modules:
            parent = types.ModuleType('fast_reid')
            parent.__path__ = [str(reference / 'fast_reid')]
            sys.modules['fast_reid'] = parent
        sys.modules['fast_reid.fast_reid_interfece'] = optional
        package = types.ModuleType('tracker')
        package.__path__ = [str(reference / 'tracker')]
        sys.modules['tracker'] = package
        module = importlib.import_module('tracker.bot_sort')
        args = types.SimpleNamespace(**settings['arguments'])
        assert args.with_reid is False and args.cmc_method == 'orb'
        model = module.BoTSORT(args, frame_rate=settings['frame_rate'])
        if hasattr(model, 'encoder'):
            raise ValueError('A no-ReID tracker constructed an appearance encoder')
        return model
    name = 'mft_official_hybrid_396f8d30'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(reference / 'trackers/hybrid_sort_tracker')]
        sys.modules[name] = package
    module = importlib.import_module(name + '.hybrid_sort')
    parser_module = module_file('mft_official_hybrid_argument_defaults', reference / 'utils/args.py')
    args = types.SimpleNamespace(**{action.dest: action.default
        for action in parser_module.make_parser()._actions if action.dest != 'help'})
    for key, value in settings['argument_overrides'].items():
        setattr(args, key, value)
    assert not args.hybrid_sort_with_reid
    # The official update builds a float row from a scalar/simple-score and a
    # length-one Kalman-score array. Normalize only these return layouts; the
    # unmodified official predict performs every state/covariance update.
    cls = module.KalmanBoxTracker
    global SCALAR_COMPATIBILITY_PROOF
    if not getattr(cls.predict, '_mft_score_layout_only', False):
        import copy
        original_predict = cls.predict
        def scalar_scores(instance):
            position, kalman, simple = original_predict(instance)
            kalman, simple = np.asarray(kalman), np.asarray(simple)
            if kalman.size != 1 or simple.size != 1:
                raise ValueError('Legacy HybridSORT scores must each contain one numeric value')
            return position, float(kalman.reshape(-1)[0]), np.asarray([float(simple.reshape(-1)[0])])
        scalar_scores._mft_score_layout_only = True
        probe = cls(np.array([10., 20., 30., 40., .9]), args=args)
        original, adapted = copy.deepcopy(probe), copy.deepcopy(probe)
        old = original_predict(original)
        new = scalar_scores(adapted)
        if (not np.array_equal(old[0], new[0]) or
                not np.array_equal(original.kf.x, adapted.kf.x) or
                not np.array_equal(original.kf.P, adapted.kf.P) or
                float(np.asarray(old[1]).reshape(-1)[0]) != new[1] or
                float(np.asarray(old[2]).reshape(-1)[0]) != new[2][0]):
            raise ValueError('Score-layout compatibility altered the official numeric prediction/state')
        SCALAR_COMPATIBILITY_PROOF = dict(position_exact=True, KF_state_exact=True,
            KF_covariance_exact=True, kalman_score_exact=True, simple_score_exact=True,
            original_score_shapes=[list(np.asarray(old[1]).shape), list(np.asarray(old[2]).shape)],
            adapted_score_shapes=[list(np.asarray(new[1]).shape), list(np.asarray(new[2]).shape)],
            scope='Single-value return layouts only; original predict math and tracker parameters unchanged.')
        cls.predict = scalar_scores
    return module.Hybrid_Sort(args, **settings['constructor'])


def rgb_image(plan, split, sequence, frame):
    import cv2
    import numpy as np
    partition = 'train' if split == 'dev' else 'test'
    path = Path(plan['data_root']) / partition / sequence / 'visible' / f'{frame:06d}.jpg'
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('The original RGB camera-compensation image could not be decoded')
    return image, hashlib.sha256(data).hexdigest()


def update(method, model, detections, image):
    import numpy as np
    if method == 'hybridsort':
        return model.update(detections, (1, 1), (1, 1))
    return np.asarray([(*item.tlbr, item.track_id) for item in model.update(detections, image)],
                      dtype=np.float64).reshape(-1, 5)


def engineering():
    import numpy as np
    spec, plan, work, base = context()
    before = sources(spec)
    old = read(spec['reuse_detector_registration'])
    checks = []
    for method in spec['trackers']:
        for stream in spec['streams']:
            binding = old['streams']['dev'][stream]['BT-005']
            data = packets(binding, binding['frames'])
            model = tracker(method, spec)
            observed = []
            for frame in range(1, 9):
                image, _ = rgb_image(plan, 'dev', 'BT-005', frame) if method == 'botsort' else (None, None)
                result = update(method, model, frame_detections(data, frame), image)
                if result.ndim != 2 or result.shape[1] != 5 or not np.isfinite(result).all():
                    raise ValueError('Reference tracker returned an invalid coordinate/ID array')
                observed.append(len(result))
            # Exercise empty-frame delivery through the genuine official API.
            image, _ = rgb_image(plan, 'dev', 'BT-005', 9) if method == 'botsort' else (None, None)
            result = update(method, model, np.empty((0, 5), np.float32), image)
            if not np.isfinite(result).all():
                raise ValueError('Reference empty-frame update returned nonfinite output')
            reset = tracker(method, spec)
            if method == 'botsort':
                zero = sys.modules['tracker.basetrack'].BaseTrack._count
            else:
                zero = sys.modules['mft_official_hybrid_396f8d30.hybrid_sort'].KalmanBoxTracker.count
            if zero != 0:
                raise ValueError('Reference identity counter did not reset before a new sequence')
            image, _ = rgb_image(plan, 'dev', 'BT-005', 1) if method == 'botsort' else (None, None)
            first = update(method, reset, frame_detections(data, 1), image)
            if len(first) and np.min(first[:, 4]) != 1:
                raise ValueError('Reference IDs did not reset for a new video')
            checks.append(dict(method=method, stream=stream, real_Dev_frames=8,
                original_api_empty_frame_delivered=True, sequence_ID_reset=True,
                counter_zero_before_new_sequence=True, output_counts=observed))
            write(base / 'engineering_progress.json', dict(source_hashes=before, checks=checks,
                scalar_score_compatibility=SCALAR_COMPATIBILITY_PROOF), exclusive=False)
            print(json.dumps(dict(event='REFERENCE_INTERFACE_CASE_PASS', method=method,
                                  stream=stream, completed_cases=len(checks))), flush=True)
    headless_metrics()
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        raise ValueError('CPU comparison engineering initialized CUDA')
    if before != sources(spec):
        raise ValueError('Engineering mutated the official/reference sources')
    write(base / 'engineering.json', dict(status='COMPLETE_CPU_NO_REID_REFERENCE_INTERFACES',
        source_hashes=before, checks=checks, GPU_work=False, new_accuracy=False,
        runtime_versions=dict(numpy=np.__version__, lap=sys.modules['lap'].__version__,
                              opencv=sys.modules['cv2'].__version__),
        scalar_score_compatibility=SCALAR_COMPATIBILITY_PROOF,
        no_ReID_constructor_allowed=True, official_source_bytes_unmodified=True))


def register():
    spec, _, _, base = context()
    proof = read(base / 'engineering.json')
    versions = sources(spec)
    if proof['source_hashes'] != versions:
        raise ValueError('Preserve an earlier engineering attempt before changing its sources')
    original = read(spec['reuse_detector_registration'])
    if original['primary_fusion_model_reselected']:
        raise ValueError('Original detector selection must remain unchanged')
    write(base / 'registered.json', dict(status='REGISTERED_ADDITIONAL_CPU_PUBLIC_COMPARISON',
        source_hashes=versions, specification_sha256=sha(SPEC), streams=original['streams'],
        engineering_sha256=sha(base / 'engineering.json'), fixed_seed=0,
        historical_Test_already_exposed=True, parameters_searched=False,
        original_fusion_primary_reselected=False, all8_Dev_before_all8_Test=True,
        registered_UTC=datetime.now(timezone.utc).isoformat()))


def worker(method, stream, split):
    import cv2
    import numpy as np
    spec, plan, _, base = context()
    reg = read(base / 'registered.json')
    if method not in spec['trackers'] or stream not in spec['streams'] or split not in ['dev', 'test']:
        raise ValueError('Only frozen complete comparison cases are allowed')
    if reg['source_hashes'] != sources(spec) or reg['specification_sha256'] != sha(SPEC):
        raise ValueError('Registered comparison sources changed')
    if split == 'test':
        freeze = read(base / 'test_parameters_frozen.json')
        if not freeze['all8_complete_Dev_before_Test'] or freeze['registration_sha256'] != sha(base / 'registered.json'):
            raise ValueError('All complete Dev cases must precede Test')
    output = base / split / (method + '__' + stream)
    output.mkdir(parents=True, exist_ok=False)
    from tools.fusion_memory_evaluation import validate_mot, run_official_tracks
    coverage = {}
    started, update_seconds = time.monotonic(), 0.
    for sequence, binding in reg['streams'][split][stream].items():
        data = packets(binding, binding['frames'])
        model = tracker(method, spec)
        image_rollup, image_frames = hashlib.sha256(), 0
        target = output / 'track_results' / (sequence + '.txt')
        target.parent.mkdir(exist_ok=True)
        count = 0
        with target.open('x') as handle:
            for frame in data['frame_ids']:
                frame = int(frame)
                image = None
                if method == 'botsort':
                    image, digest = rgb_image(plan, split, sequence, frame)
                    image_rollup.update(f'{frame}:{digest}\n'.encode())
                    image_frames += 1
                tick = time.monotonic()
                result = update(method, model, frame_detections(data, frame), image)
                update_seconds += time.monotonic() - tick
                if not np.isfinite(result).all():
                    raise ValueError('Nonfinite official tracker output')
                for x1, y1, x2, y2, identity in result:
                    width, height = float(x2-x1), float(y2-y1)
                    if width <= 0 or height <= 0 or width * height <= spec['minimum_box_area']:
                        continue
                    if int(identity) != identity or identity <= 0:
                        raise ValueError('Tracker identity must be a positive integer')
                    handle.write(f'{frame},{int(identity)},{float(x1):.3f},{float(y1):.3f},'
                                 f'{width:.3f},{height:.3f},1.000000,-1,-1,-1\n')
                    count += 1
        validate_mot(target, binding['frames'], count)
        if sha(binding['path']) != binding['sha256'] or sha(binding['producer']) != binding['producer_sha256']:
            raise ValueError('Frozen detector/producer changed during replay')
        coverage[sequence] = dict(frames=binding['frames'], output_rows=count,
            output_sha256=sha(target), RGB_camera_compensation_frames=image_frames,
            RGB_content_rollup_sha256=image_rollup.hexdigest() if image_frames else None)
        print(json.dumps(dict(event='SEQUENCE_TRACKS_COMPLETE', method=method, stream=stream,
            split=split, sequence=sequence, **coverage[sequence])), flush=True)
    manifest = dict(status='COMPLETE_PUBLIC_TRACKS_BEFORE_GT_EVALUATION', method=method,
        stream=stream, split=split, frames=sum(row['frames'] for row in coverage.values()),
        coverage=coverage, sources=reg['streams'][split][stream], source_versions=reg['source_hashes'],
        registration_sha256=sha(base / 'registered.json'), parameters=spec['trackers'][method],
        tracker_update_seconds=update_seconds, replay_seconds=time.monotonic()-started,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        GT_inference_input=False, future_frame_input=False, previous_outputs_rewritten=False,
        appearance_network_used=False, GPU_work=False, fixed_seed=0,
        scalar_score_compatibility=SCALAR_COMPATIBILITY_PROOF,
        runtime_versions=dict(numpy=np.__version__, lap=sys.modules['lap'].__version__, opencv=cv2.__version__),
        cost_scope='CPU cached tracking; update includes ORB GMC for BoT-SORT. Replay also includes '
                   'original RGB reads/decoding. Excludes detector/fusion/IR generation and official '
                   'evaluation; not end-to-end FPS. Shared-host load uncontrolled.')
    write(output / 'inference_manifest.json', manifest)
    protocol = read(ROOT / plan['protocol'])
    for sequence in coverage:
        part = 'train' if split == 'dev' else 'test'
        path = Path(plan['evaluation']['GT_root']) / part / sequence / 'gt/gt.txt'
        if sha(path) != protocol['derived_gt_sha256'][part + '/' + sequence]:
            raise ValueError('Fixed ProtocolV2 evaluation GT changed')
    headless_metrics()
    official = run_official_tracks(output / 'track_results', output, plan, split)
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        raise ValueError('CPU comparison/evaluation initialized CUDA')
    if reg['source_hashes'] != sources(spec):
        raise ValueError('Frozen comparison implementation changed during evaluation')
    write(output / 'metrics.json', dict(status='COMPLETE_CPU_ADDITIONAL_PUBLIC_TRACKER_FULL_SPLIT',
        manifest=manifest, **official, goal_met=False, supported_innovations=0))
    print(json.dumps(dict(event='OFFICIAL_COMPLETE', method=method, stream=stream, split=split,
                         HOTA=official['combined47']['HOTA'])), flush=True)


def identity(pid, expected_command=None):
    proc = Path('/proc') / str(pid)
    for _ in range(100):
        raw = (proc / 'cmdline').read_bytes()
        if expected_command is None or raw == b'\0'.join(x.encode() for x in expected_command) + b'\0':
            break
        time.sleep(.01)
    fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), raw_argv_sha256=hashlib.sha256(raw).hexdigest(),
                actual_argv=raw.decode().split('\0')[:-1], expected_argv_matches=(
                    expected_command is None or raw == b'\0'.join(x.encode() for x in expected_command) + b'\0'))


def run():
    spec, _, _, base = context()
    reg = read(base / 'registered.json')
    state = dict(status='RUNNING_ALL8_COMPLETE_DEV_FIRST', controller=identity(os.getpid()),
                 registration_sha256=sha(base / 'registered.json'), jobs={}, GPU_work=False)
    state_path = base / 'state.json'
    write(state_path, state)
    for split in ['dev', 'test']:
        for method in spec['trackers']:
            for stream in spec['streams']:
                key = split + '__' + method + '__' + stream
                command = [sys.executable, '-u', str(Path(__file__)), 'worker',
                           '--method', method, '--stream', stream, '--split', split]
                log = base / 'logs' / (key + '.txt')
                log.parent.mkdir(exist_ok=True)
                started = time.monotonic()
                with log.open('x') as handle:
                    child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
                    job = dict(status='RUNNING', command=command, process=identity(child.pid, command),
                               log=str(log), CUDA_VISIBLE_DEVICES='', max_sampled_RSS_GiB=0.)
                    if not job['process']['expected_argv_matches']:
                        raise ValueError('Actual original CPU worker argv did not match its command')
                    state['jobs'][key] = job
                    write(state_path, state, exclusive=False)
                    print(json.dumps(dict(event='STARTED', job=key, **job['process'])), flush=True)
                    while child.poll() is None:
                        try:
                            pages = int((Path('/proc') / str(child.pid) / 'statm').read_text().split()[1])
                            job['max_sampled_RSS_GiB'] = max(job['max_sampled_RSS_GiB'], pages*os.sysconf('SC_PAGE_SIZE')/2**30)
                        except (OSError, IndexError, ValueError):
                            pass
                        time.sleep(1)
                    code = child.wait()
                job.update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code,
                           wall_seconds=time.monotonic()-started, log_sha256=sha(log))
                if code:
                    state['status'] = 'FAILED_PRESERVE_ORIGINAL_CPU_ATTEMPT'
                    write(state_path, state, exclusive=False)
                    raise RuntimeError('Original CPU comparison worker failed: ' + key)
                metrics = base / split / (method + '__' + stream) / 'metrics.json'
                job['metrics_sha256'] = sha(metrics)
                write(state_path, state, exclusive=False)
                print(json.dumps(dict(event='EXITED', job=key, exit_code=code,
                                      wall_seconds=job['wall_seconds'])), flush=True)
        if split == 'dev':
            if len(state['jobs']) != 8 or any(j['exit_code'] != 0 for j in state['jobs'].values()):
                raise ValueError('All eight full Dev cases must actually finish')
            write(base / 'test_parameters_frozen.json', dict(
                all8_complete_Dev_before_Test=True, registration_sha256=sha(base / 'registered.json'),
                complete_Dev_metrics={k:v['metrics_sha256'] for k,v in state['jobs'].items()},
                identical_parameters=spec['trackers'], no_parameter_search=True,
                original_fusion_primary_reselected=False, all8_Test_mandatory=True,
                freeze_UTC=datetime.now(timezone.utc).isoformat()))
            state['status'] = 'RUNNING_ALL8_FIXED_TEST_CASES'
            write(state_path, state, exclusive=False)
    state['status'] = 'COMPLETE_ALL8_DEV_AND_ALL8_TEST_ADDITIONAL_CPU_PUBLIC_TRACKERS'
    write(state_path, state, exclusive=False)
    print(json.dumps(dict(status=state['status'], jobs=len(state['jobs']))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['engineering', 'register', 'run', 'worker'])
    parser.add_argument('--method', choices=['botsort', 'hybridsort'])
    parser.add_argument('--stream')
    parser.add_argument('--split', choices=['dev', 'test'])
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.method, args.stream, args.split)
    else:
        globals()[args.action]()


if __name__ == '__main__':
    main()
