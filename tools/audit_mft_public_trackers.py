#!/usr/bin/env python3
"""Frozen SORT/OC-SORT comparisons on existing complete detector streams.

Reference implementations stay outside Git, unmodified, with fixed commits,
licenses and byte hashes. This CPU experiment does not train or select a new
fusion model. Dev completes before all preregistered historical-Test cases.
"""
import argparse
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
import urllib.request
import base64

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = ROOT / 'configs/audit_mft_public_trackers_20261125.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(encoded)
    else:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(encoded)
        os.replace(temporary, path)


def context():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This supplement is CPU only; explicitly hide CUDA')
    spec = read(SPEC)
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    protocol = read(ROOT / plan['protocol'])
    work = Path(plan['work_root'])
    base = work / spec['output_directory']
    from tools.neck72_runtime import require_host_memory, require_storage
    require_host_memory(2 * 2**30, maximum_workers=1)
    require_storage(work, growth_bytes=256 * 2**20)
    return spec, plan, protocol, work, base


def prepare():
    spec, _, _, work, _ = context()
    for reference in spec['reference_sources']:
        for row in reference['files']:
            target = work / spec['reference_directory'] / reference['directory'] / row['path']
            if not target.exists():
                url = ('https://api.github.com/repos/' + reference['repo'] + '/contents/'
                       + row['path'] + '?ref=' + reference['commit'])
                request = urllib.request.Request(url, headers={
                    'User-Agent': 'IR-MFT-source-provenance', 'Accept': 'application/vnd.github+json'})
                with urllib.request.urlopen(request, timeout=25) as response:
                    value = json.load(response)
                if value['encoding'] != 'base64' or value['size'] != row['bytes']:
                    raise ValueError('Unexpected reference source download')
                encoded = base64.b64decode(value['content'])
                if hashlib.sha256(encoded).hexdigest() != row['sha256']:
                    raise ValueError('Fixed-commit reference bytes differ')
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(encoded)
            if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
                raise ValueError('Original reference source changed: ' + str(target))


def sources(spec, work):
    names = [str(Path(__file__).relative_to(ROOT)), str(SPEC.relative_to(ROOT)),
        'configs/perceptual_mft_20261122.json', 'configs/cfp/protocol_v2.json',
        'configs/perceptual_single_seed_policy_20261123.json',
        'configs/perceptual_structural_fulltrain_test_20261123.json',
        'tools/eval_mft26ir.py', 'tools/export_perceptual_metrics47.py',
        'tools/fusion_memory_evaluation.py', 'tools/fusion_memory_observation.py',
        'yolox/tracker/byte_tracker.py', 'yolox/tracker/basetrack.py',
        'yolox/tracker/kalman_filter.py', 'yolox/tracker/matching.py']
    names.extend(case['original_Test_configuration'] for case in spec['streams'].values())
    result = {str(ROOT / name): sha(ROOT / name) for name in names}
    for reference in spec['reference_sources']:
        for row in reference['files']:
            path = work / spec['reference_directory'] / reference['directory'] / row['path']
            if sha(path) != row['sha256']:
                raise ValueError('Reference implementation/notice bytes changed')
            result[str(path)] = row['sha256']
    return result


def reference_module(method, spec, work):
    folder = work / spec['reference_directory']
    if method == 'ocsort':
        name = 'mft_official_ocsort_8462'
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(folder / 'OC_SORT/trackers/ocsort_tracker')]
            sys.modules[name] = package
        return importlib.import_module(name + '.ocsort')
    if method == 'sort':
        # The original CLI imports skimage for its unused visual demo. Append
        # the installed dependency path without shadowing the active NumPy.
        if importlib.util.find_spec('skimage') is None:
            sys.path.append(spec['SORT_demo_dependency_directory'])
        name = 'mft_official_sort_2236'
        if name not in sys.modules:
            loader = importlib.util.spec_from_file_location(name, folder / 'SORT/sort.py')
            module = importlib.util.module_from_spec(loader)
            sys.modules[name] = module
            loader.loader.exec_module(module)
        return sys.modules[name]
    raise ValueError('Unregistered public tracker')


def tracker(method, spec, work):
    module = reference_module(method, spec, work)
    module.KalmanBoxTracker.count = 0
    return (module.OCSort if method == 'ocsort' else module.Sort)(**spec['trackers'][method]['parameters'])


def packets(binding, frames):
    import numpy as np
    path = Path(binding['path'])
    if sha(path) != binding['sha256']:
        raise ValueError('Frozen detection packets changed')
    with np.load(path, allow_pickle=False) as arrays:
        required = {'frame_ids', 'offsets', 'boxes', 'scores'}
        if (not required.issubset(arrays.files)
                or not set(arrays.files).issubset(required | {'anchor_indices', 'rgb_reference_scores'})):
            raise ValueError('Unexpected cached detection schema')
        # Match the original ByteTrack replayer: anchor provenance and RGB
        # reference scores are not extra association inputs for public trackers.
        data = {name: arrays[name].copy() for name in required}
    np.testing.assert_array_equal(data['frame_ids'], np.arange(1, frames + 1))
    offsets, boxes, scores = (data[k] for k in ['offsets', 'boxes', 'scores'])
    if (offsets.shape != (frames + 1,) or offsets.dtype.kind not in 'iu'
            or offsets[0] != 0 or offsets[-1] != len(boxes) or (np.diff(offsets) < 0).any()
            or boxes.shape != (len(scores), 4) or scores.shape != (len(boxes),)
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or (boxes[:, 2:] <= boxes[:, :2]).any() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError('Incomplete/nonfinite cached detection packets')
    return data


def frame_detections(data, frame):
    import numpy as np
    lo, hi = data['offsets'][frame - 1:frame + 1]
    return np.c_[data['boxes'][lo:hi], data['scores'][lo:hi]].astype(np.float32)


def update(method, model, detections, spec):
    if method == 'sort':
        return model.update(detections[detections[:, 4] > spec['SORT_detection_threshold']])
    return model.update(detections, (1, 1), (1, 1))


def headless_metrics():
    # SORT's unmodified demo imports select TkAgg globally. TrackEval draws
    # figures after computing metrics, so select its noninteractive renderer
    # here, after tracker imports. No association or evaluation rule changes.
    import matplotlib
    matplotlib.use('Agg', force=True)


def bindings(spec, protocol, work):
    from tools.fusion_memory_observation import LENGTHS
    result = {}
    for split in ['dev', 'test']:
        if list(LENGTHS[split]) != protocol['dev' if split == 'dev' else 'historical_test']:
            raise ValueError('Protocol sequence order changed')
        result[split] = {}
        for stream, case in spec['streams'].items():
            result[split][stream] = {}
            original_case = read(ROOT / case['original_Test_configuration'])
            if (original_case['model_id'] != case['model_id']
                    or original_case['proposal_key'] != case['proposal_key']):
                raise ValueError('Preserve each original complete-Dev selected proposal policy')
            for seq, frames in LENGTHS[split].items():
                meta_path = work / spec['producer_directory'] / split / seq / 'complete.json'
                meta = read(meta_path)
                if (meta['status'] != 'COMPLETE_STRUCTURAL_RELATIVE_OBSERVATIONS'
                        or meta['split'] != split or meta['sequence'] != seq or meta['frames'] != frames
                        or meta['GT_read'] or meta['future_frame_input']
                        or meta['previous_outputs_rewritten']
                        or meta['config_sha256'] != original_case['fusion_plan_sha256']
                        or meta['full_train_study_sha256'] != original_case['full_train_study_sha256']
                        or meta['source_RGB_sha256'] != spec['source_RGB_checkpoint_sha256']):
                    raise ValueError('Original full current-image producer required')
                info = meta['outputs'][case['proposal_key']]
                item = dict(path=info['path'], sha256=info['sha256'], frames=frames,
                    producer=str(meta_path), producer_sha256=sha(meta_path),
                    original_Dev_selected_case=case['Dev_case'], model=meta['models'][case['model_id']])
                packets(item, frames)
                result[split][stream][seq] = item
    return result


def engineering():
    import numpy as np
    spec, _, protocol, work, base = context()
    prepare()
    versions = sources(spec, work)
    bound = bindings(spec, protocol, work)
    checks = []
    for method in spec['trackers']:
        model = tracker(method, spec, work)
        first_id = None
        for index in range(12):
            det = np.array([[10 + index, 20, 30 + index, 40, .9]], dtype=np.float32)
            if index in [3, 4]:
                det = np.empty((0, 5), dtype=np.float32)
            result = update(method, model, det, spec)
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite reference tracker on an occlusion/empty-frame case')
            if index == 0:
                first_id = int(result[0, 4])
            if method == 'ocsort' and index == 11 and int(result[0, 4]) != first_id:
                raise ValueError('Reference OC-SORT failed its simple retained-identity check')
        fresh = update(method, tracker(method, spec, work),
            np.array([[10, 20, 30, 40, .9]], dtype=np.float32), spec)
        if int(fresh[0, 4]) != 1:
            raise ValueError('Reference state did not reset between sequences')
        checks.append(dict(method=method, empty_frames_delivered=True, sequence_reset=True))
    headless_metrics()
    import matplotlib.pyplot as plt
    import io
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png')
    plt.close(figure)
    if not buffer.getvalue().startswith(b'\x89PNG'):
        raise ValueError('Official plotting cannot use the headless renderer')
    checks.append(dict(headless_metric_plot_after_original_SORT_import=True))
    from tools.eval_mft26ir import write_sequence_results
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracker.basetrack import BaseTrack
    from types import SimpleNamespace
    for stream, case in spec['streams'].items():
        BaseTrack._count = 0
        data = packets(bound['dev'][stream]['BT-005'], 1834)
        model = BYTETracker(SimpleNamespace(track_thresh=.6, track_buffer=30,
            match_thresh=.9, mot20=False), frame_rate=25)
        rows = []
        for frame in range(1, 9):
            rows.extend((frame, obj.tlwh.copy(), obj.track_id, obj.score)
                for obj in model.update(frame_detections(data, frame), (1, 1), (1, 1))
                if obj.tlwh[2] * obj.tlwh[3] > spec['minimum_box_area'])
        path = base / 'engineering' / (stream + '_ByteTrack8.txt')
        if path.exists():
            raise ValueError('Preserve an earlier engineering attempt')
        write_sequence_results(str(path), rows)
        reference = work / 'structural_full_train_dev' / case['Dev_case'] / 'track_results/BT-005.txt'
        expected = ''.join(line for line in reference.read_text().splitlines(keepends=True)
                           if int(line.split(',')[0]) <= 8).encode()
        if path.read_bytes() != expected:
            raise ValueError('Cached coordinate/score/ID delivery differs from original ByteTrack')
        checks.append(dict(stream=stream, actual_8frame_ByteTrack_bytes_exact=True,
            reference=str(reference), reference_sha256=sha(reference), prefix_sha256=sha(path)))
    write(base / 'engineering.json', dict(status='COMPLETE_CPU_REFERENCE_AND_CACHE_ENGINEERING',
        source_hashes=versions, checks=checks, sources_validated=bound,
        GT_used_for_engineering=False, GPU_work=False, fixed_seed=0))


def register():
    spec, _, protocol, work, base = context()
    proof = read(base / 'engineering.json')
    if proof['source_hashes'] != sources(spec, work):
        raise ValueError('Engineering source changed; preserve the earlier attempt')
    original = read(work / 'structural_full_train_Test_selection_frozen.json')
    if not original['complete50Dev_before_Test'] or original['primary'] != 'FTT_primary_IR':
        raise ValueError('The original complete-Dev choices must remain frozen')
    write(base / 'registered.json', dict(status='REGISTERED_FIXED_PUBLIC_TRACKER_PARAMETERS',
        source_hashes=proof['source_hashes'], specification_sha256=sha(SPEC),
        engineering_sha256=sha(base / 'engineering.json'),
        streams=bindings(spec, protocol, work), original_fusion_selection_sha256=sha(
            work / 'structural_full_train_Test_selection_frozen.json'),
        historical_Test_already_exposed=True, new_Test_or_Dev_used_to_set_tracker_parameters=False,
        parameter_selection='One fixed official/default setting per tracker; all cases retained.',
        primary_fusion_model_reselected=False, fixed_seed=0, GPU_work=False,
        registered_unix=time.time()))


def worker(method, stream, split):
    spec, plan, _, work, base = context()
    if method not in spec['trackers'] or stream not in spec['streams'] or split not in ['dev', 'test']:
        raise ValueError('Only preregistered complete cases are supported')
    reg = read(base / 'registered.json')
    if reg['source_hashes'] != sources(spec, work) or reg['specification_sha256'] != sha(SPEC):
        raise ValueError('Frozen experiment/reference bytes changed')
    if split == 'test':
        frozen = read(base / 'test_parameters_frozen.json')
        if frozen['registration_sha256'] != sha(base / 'registered.json') or not frozen['all8_complete_Dev_before_Test']:
            raise ValueError('Complete Dev must precede the fixed Test supplement')
    identifier = method + '__' + stream
    output = base / split / identifier
    output.mkdir(parents=True, exist_ok=False)
    from tools.fusion_memory_evaluation import validate_mot, run_official_tracks
    coverage = {}
    started = time.monotonic()
    update_seconds = 0.
    for seq, binding in reg['streams'][split][stream].items():
        data = packets(binding, binding['frames'])
        model = tracker(method, spec, work)
        target = output / 'track_results' / (seq + '.txt')
        target.parent.mkdir(exist_ok=True)
        count = 0
        with target.open('x') as handle:
            for frame in data['frame_ids']:
                detections = frame_detections(data, int(frame))
                tick = time.monotonic()
                result = update(method, model, detections, spec)
                update_seconds += time.monotonic() - tick
                for x1, y1, x2, y2, identity in result:
                    width, height = float(x2 - x1), float(y2 - y1)
                    if width <= 0 or height <= 0 or width * height <= spec['minimum_box_area']:
                        continue
                    if int(identity) != identity:
                        raise ValueError('Noninteger tracker identity')
                    # These official APIs return no track confidence. Constant
                    # 1 is a MOT serialization field, never an association cue.
                    handle.write(f'{int(frame)},{int(identity)},{float(x1):.3f},{float(y1):.3f},'
                                 f'{width:.3f},{height:.3f},1.000000,-1,-1,-1\n')
                    count += 1
        validate_mot(target, binding['frames'], count)
        if sha(binding['path']) != binding['sha256'] or sha(binding['producer']) != binding['producer_sha256']:
            raise ValueError('Original inference inputs changed during replay')
        coverage[seq] = dict(frames=binding['frames'], detections=len(data['boxes']),
            output_rows=count, output_sha256=sha(target))
        print(json.dumps(dict(status='SEQUENCE_TRACKS_COMPLETE', id=identifier,
            split=split, sequence=seq, **coverage[seq])), flush=True)
    manifest = dict(status='COMPLETE_CAUSAL_TRACKS_BEFORE_GT_EVALUATION',
        split=split, sequences=list(coverage), frames=sum(v['frames'] for v in coverage.values()),
        method=method, stream=stream, parameters=spec['trackers'][method], coverage=coverage,
        sources=reg['streams'][split][stream], source_versions=reg['source_hashes'],
        configuration_sha256=sha(SPEC), registration_sha256=sha(base / 'registered.json'),
        GT_inference_input=False, future_frame_input=False, previous_outputs_rewritten=False,
        tracker_update_seconds=update_seconds, replay_seconds=time.monotonic() - started,
        cost_scope='CPU cached tracking only; excludes detector/fusion/IRgeneration; not end-to-end FPS.',
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        fixed_seed=0, GPU_work=False)
    write(output / 'inference_manifest.json', manifest)
    for seq in coverage:
        part = 'train' if split == 'dev' else 'test'
        path = Path(plan['evaluation']['GT_root']) / part / seq / 'gt/gt.txt'
        if sha(path) != read(ROOT / plan['protocol'])['derived_gt_sha256'][part + '/' + seq]:
            raise ValueError('Protocol-derived evaluation GT changed')
    headless_metrics()
    official = run_official_tracks(output / 'track_results', output, plan, split)
    if reg['source_hashes'] != sources(spec, work):
        raise ValueError('Frozen implementation changed during the official worker')
    write(output / 'metrics.json', dict(status='COMPLETE_CPU_PUBLIC_TRACKER_FULL_SPLIT',
        manifest=manifest, **official, goal_met=False, supported_innovations=0))
    print(json.dumps(dict(status='OFFICIAL_COMPLETE', id=identifier, split=split,
        HOTA=official['combined47']['HOTA'], IDF1=official['combined47']['IDF1'])), flush=True)


def identity(pid):
    proc = Path('/proc') / str(pid)
    return dict(pid=pid, start_ticks=int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]),
        argv_sha256=hashlib.sha256((proc / 'cmdline').read_bytes()).hexdigest())


def run():
    spec, _, _, _, base = context()
    state_path = base / 'state.json'
    if state_path.exists():
        raise ValueError('Preserve the existing controller and any failed/completed attempt')
    read(base / 'registered.json')
    state = dict(status='RUNNING_COMPLETE_DEV_FIRST', controller=identity(os.getpid()),
        registration_sha256=sha(base / 'registered.json'), jobs={}, GPU_work=False)
    write(state_path, state)
    for split in ['dev', 'test']:
        for method in spec['trackers']:
            for stream in spec['streams']:
                key = split + '__' + method + '__' + stream
                command = [sys.executable, '-u', str(Path(__file__)), 'worker',
                    '--method', method, '--stream', stream, '--split', split]
                log_path = base / 'logs' / (key + '.txt')
                log_path.parent.mkdir(exist_ok=True)
                started = time.monotonic()
                with log_path.open('x') as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    job = dict(status='RUNNING', command=command, log=str(log_path),
                        process=identity(process.pid), CUDA_VISIBLE_DEVICES='', max_sampled_RSS_GiB=0.)
                    state['jobs'][key] = job
                    write(state_path, state, exclusive=False)
                    print(json.dumps(dict(status='STARTED', job=key, **job['process'])), flush=True)
                    while process.poll() is None:
                        try:
                            rss = int((Path('/proc') / str(process.pid) / 'statm').read_text().split()[1])
                            job['max_sampled_RSS_GiB'] = max(job['max_sampled_RSS_GiB'],
                                rss * os.sysconf('SC_PAGE_SIZE') / 2**30)
                        except (OSError, ValueError, IndexError):
                            pass
                        time.sleep(1)
                    job.update(exit_code=process.wait(), wall_seconds=time.monotonic() - started,
                               log_sha256=sha(log_path))
                job['status'] = 'COMPLETE' if job['exit_code'] == 0 else 'FAILED'
                if job['exit_code']:
                    state['status'] = 'FAILED_PRESERVE_ORIGINAL_CPU_ATTEMPT'
                    write(state_path, state, exclusive=False)
                    raise RuntimeError('Original CPU worker failed: ' + key)
                metrics_path = base / split / (method + '__' + stream) / 'metrics.json'
                job['metrics_sha256'] = sha(metrics_path)
                write(state_path, state, exclusive=False)
                print(json.dumps(dict(event='EXITED', job=key, **job)), flush=True)
        if split == 'dev':
            dev = {k: v['metrics_sha256'] for k, v in state['jobs'].items()}
            if len(dev) != 8 or any(v['exit_code'] != 0 for v in state['jobs'].values()):
                raise ValueError('All eight full-Dev cases must actually finish')
            write(base / 'test_parameters_frozen.json', dict(
                registration_sha256=sha(base / 'registered.json'), all8_complete_Dev_before_Test=True,
                complete_Dev_metrics=dev, identical_parameters=spec['trackers'],
                no_Dev_or_Test_parameter_search=True, original_fusion_primary_reselected=False,
                mandatory_all8_Test=True, freeze_unix=time.time()))
            state['status'] = 'RUNNING_ALL_FIXED_TEST_CASES'
            write(state_path, state, exclusive=False)
    state['status'] = 'COMPLETE_ALL8_DEV_AND_ALL8_TEST_CPU_PUBLIC_TRACKERS'
    write(state_path, state, exclusive=False)
    print(json.dumps(dict(status=state['status'], jobs=len(state['jobs']),
        registration_sha256=sha(base / 'registered.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'engineering', 'register', 'run', 'worker'])
    parser.add_argument('--method', choices=['sort', 'ocsort'])
    parser.add_argument('--stream')
    parser.add_argument('--split', choices=['dev', 'test'])
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.method, args.stream, args.split)
    else:
        globals()[args.action]()


if __name__ == '__main__':
    main()
