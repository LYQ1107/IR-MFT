#!/usr/bin/env python3
"""All seven Test videos, then every frozen full-Train formal tracking case."""
import configparser
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
import numpy as np
from tools.correlation_geometry_checks import compare_packets
from tools.recover_correlation_formal_test import aligned_source_audit
from tools.run_pseudo_ir_comparison import dump, sha256


def packet(path):
    with np.load(path) as z:
        return {key: z[key] for key in ['frame_ids', 'offsets', 'boxes', 'scores']}


def source_proof(base, sequence, plan, specification):
    canonical = Path(plan['frozen_detection_sources']['rgb_test']) / 'detections' / (sequence + '.npz')
    fresh = base / 'CG3_source_RGB.npz'
    audit = aligned_source_audit(packet(fresh), packet(canonical), tail_frames=4)
    off = [row for row in specification['models'] if row.get('intervention') == 'unavailable']
    assert len(off) == 1
    name = off[0]['id']
    audit['unavailable_raw_exact_to_fresh'] = compare_packets(base / (name + '_raw.npz'), fresh)
    audit['unavailable_priority_exact_to_canonical'] = compare_packets(base / (name + '.npz'), canonical)
    audit['source8_exact_to_full'] = compare_packets(base / 'source_check_8/CG3_source_RGB.npz', fresh, 8)
    count = 0
    for row in specification['models']:
        if row.get('baseline'):
            continue
        with np.load(base / (row['id'] + '_raw.npz')) as z:
            assert len(z['boxes']) == len(z['anchor_indices']) == len(z['rgb_reference_scores'])
            assert ((z['anchor_indices'] >= 0) & (z['anchor_indices'] < 23625)).all()
            assert ((z['rgb_reference_scores'] >= 0) & (z['rgb_reference_scores'] <= 1)).all()
        count += 1
    assert count == len(specification['models']) - 1
    audit['dense_RGB_geometry'] = dict(raw_streams=count, every_selected_anchor_checked_before_publication=True)
    return audit


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    spec_path = ROOT / 'configs/perceptual_structural_fulltrain_test_20261123.json'
    specification = json.loads(spec_path.read_text())
    freeze = json.loads((work / 'structural_full_train_Test_selection_frozen.json').read_text())
    assert freeze['specification_sha256'] == sha256(spec_path) and freeze['complete50Dev_before_Test']
    state_path = work / 'structural_full_train_test_state.json'
    lock = state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not state_path.exists()
    sources = ['tools/launch_structural_full_train_test.py', 'tools/extract_structural_full_train.py',
        'tools/replay_structural_full_train.py', 'tools/extract_structural_relative.py',
        'tools/structural_relative_inference.py', 'tools/structural_grid.py', 'tools/structural_correspondence.py',
        'tools/build_correlation_cache.py', 'tools/indexed_postprocess.py',
        'tools/structural_gate_observer.py', 'tools/recover_correlation_formal_test.py',
        'tools/correlation_geometry_checks.py', 'yolox/models/structural_factorial_repair.py',
        'yolox/models/structural_relative_repair.py', 'yolox/models/frozen_head_repair.py',
        'yolox/models/yolo_head.py', 'yolox/tracker/foveated_regions.py',
        'yolox/tracker/priority_observations.py', 'yolox/tracker/byte_tracker.py']
    jobs = {}
    directory = ROOT / 'configs/perceptual_structural_fulltrain_test_cases_20261123'
    for entry in specification['cases']:
        path = directory / (entry['id'] + '.json')
        case = json.loads(path.read_text())
        assert case['specification_sha256'] == sha256(spec_path)
        jobs[entry['id']] = dict(status='PENDING', config=str(path), sha256=sha256(path))
    lengths = {}
    for sequence in protocol['historical_test']:
        path = Path(plan['evaluation']['GT_root']) / 'test' / sequence / 'seqinfo.ini'
        ini = configparser.ConfigParser()
        ini.read(path)
        lengths[sequence] = ini.getint('Sequence', 'seqLength')
    assert len(lengths) == 7 and sum(lengths.values()) == 15985
    producers = {seq: dict(status='PENDING', frames=lengths[seq])
                 for seq in sorted(lengths, key=lambda s: (-lengths[s], s))}
    state = dict(status='PRODUCING_ALL7_FROZEN_TEST_WITHOUT_GT', controller_pid=os.getpid(),
        jobs=jobs, producers=producers, source_hashes={name: sha256(ROOT / name) for name in sources},
        specification_sha256=sha256(spec_path), fixed_seed=0, additional_seed_work=False,
        Test_accuracy_read=False, goal_met=False, supported_innovations=0)
    dump(state_path, state)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    active = {}
    try:
        while True:
            for physical, (sequence, mode, process) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[physical]
                producer = producers[sequence]
                producer[mode + '_exit_code'] = code
                assert code == 0, producer['log']
                base = work / 'full_train_observations/structural_relative_predictions/test' / sequence
                if mode == 'source':
                    meta = json.loads((base / 'source_check_8/complete.json').read_text())
                    assert len(meta['outputs']) == 17 and all(r['array_exact'] for r in meta['fresh_RGB_parity'])
                    producer['status'] = 'SOURCE_PASSED'
                else:
                    meta_path = base / 'complete.json'
                    meta = json.loads(meta_path.read_text())
                    assert meta['frames'] == lengths[sequence] and len(meta['outputs']) == specification['counts']['output_streams']
                    assert meta['specification_sha256'] == state['specification_sha256']
                    assert meta['source_hashes'] == {key: state['source_hashes'][key] for key in meta['source_hashes']}
                    producer.update(status='COMPLETE', source_proof=source_proof(base, sequence, plan, specification),
                                    complete_sha256=sha256(meta_path))
            for physical in ['4', '5']:
                if physical in active:
                    continue
                candidates = [(seq, row) for seq, row in producers.items() if row['status'] == 'SOURCE_PASSED']
                if not candidates:
                    candidates = [(seq, row) for seq, row in producers.items() if row['status'] == 'PENDING']
                if not candidates:
                    continue
                sequence, producer = candidates[0]
                mode = 'source' if producer['status'] == 'PENDING' else 'full'
                assert all(sha256(ROOT / name) == digest for name, digest in state['source_hashes'].items())
                assert sha256(spec_path) == state['specification_sha256']
                assert shutil.disk_usage(work).free >= 40 * 2**30
                assert int(subprocess.check_output(['du', '-s', '-B1', str(work)], text=True).split()[0]) < 64 * 2**30
                command = [sys.executable, '-u', 'tools/extract_structural_full_train.py', '--split', 'test',
                           '--sequence', sequence]
                command += ['--source-check', '8'] if mode == 'source' else ['--specification', str(spec_path)]
                log = work / (sequence + '.full_train_test_' + mode + '.log')
                with log.open('xb', buffering=0) as stream:
                    process = subprocess.Popen(command, cwd=ROOT, env=dict(environment, CUDA_VISIBLE_DEVICES=physical),
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[physical] = (sequence, mode, process)
                producer.update(status=mode.upper() + '_RUNNING', pid=process.pid, physical_gpu=physical,
                                command=command, log=str(log))
            dump(state_path, state)
            if not active:
                assert all(p['status'] == 'COMPLETE' for p in producers.values())
                break
            time.sleep(5)
        state['status'] = 'RUNNING_ALL_FROZEN_FULL_TRAIN_TEST_REPLAYS'
        active = {}
        while True:
            for key, process in list(active.items()):
                code = process.poll()
                if code is not None:
                    jobs[key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                    del active[key]
            for key, job in jobs.items():
                if len(active) >= 2 or any(v['status'] == 'FAILED' for v in jobs.values()):
                    break
                if job['status'] != 'PENDING':
                    continue
                assert sha256(job['config']) == job['sha256']
                command = [sys.executable, '-u', 'tools/replay_structural_full_train.py', '--split', 'test',
                    '--config', job['config'], '--output', str(work / 'structural_full_train_test' / key)]
                log = work / (key + '.full_train_test_replay.log')
                with log.open('xb', buffering=0) as stream:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[key] = process
                job.update(status='RUNNING', pid=process.pid, command=command, log=str(log))
            dump(state_path, state)
            if not active:
                assert all(v['status'] == 'COMPLETE' for v in jobs.values())
                break
            time.sleep(5)
        state['status'] = 'COMPLETE'
        dump(state_path, state)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_FULL_TRAIN_TEST', error=repr(error))
        dump(state_path, state)
        raise


if __name__ == '__main__':
    main()
