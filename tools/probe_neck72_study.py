#!/usr/bin/env python3
"""CPU proof of neck72 scheduling rules and read-only actual query observation."""
import copy
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.eval_mft26ir import write_sequence_results
from tools.launch_neck72_study import source_files
from tools.native_pyramid_training_loop import state_sha256
from tools.native_query_replay import load_native_query_evidence
from tools.neck72_experiment import context
from tools.neck72_runtime import read, require_storage, require_host_memory
from tools.run_neck72_replay import QueryIncrementObserver, tracker_for
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2); torch.manual_seed(0)
    plan, protocol, work = context(); started = time.monotonic()
    output = work / 'neck72_study_CPU_proof.json'
    assert not output.exists(), 'Preserve previous complete CPU proof'
    before = {p: sha256(ROOT / p) for p in source_files()}
    require_host_memory(4*2**30, maximum_workers=1); require_storage(work)
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_neck72_study.py']
    log = work / 'neck72_study_pytest.log'
    with log.open('x') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    match = re.search(r'\b(\d+) passed\b', log.read_text()); assert match
    tests = int(match.group(1)); assert tests >= 10
    # Old seed0 models exercise the unchanged query classes before the new
    # neck72 heads exist. These are engineering traces, not new method scores.
    records = []
    for algorithm, old_name in [('ordered', 'robust'), ('consensus', 'consensus')]:
        for front in ['new', 'rgb']:
            for cue in ['rgb', 'ir']:
                config_path = ROOT / 'configs/perceptual_native_common_test_cases_20261124' / (
                    f'NC36_{old_name}_{front}_{cue}_normal.json')
                config = read(config_path)
                source_path = work / 'features/dev/BT-005' / (front+'.npz')
                with np.load(source_path) as array:
                    data = {key: array[key] for key in array.files}
                point_path = work / 'transport/dev/BT-005' / (front+'.npz')
                with np.load(point_path) as array:
                    data['foreground_transport'] = {key: array[key] for key in array.files}
                native = load_native_query_evidence(data, work, 'dev', 'BT-005', config)
                outputs, packets = [], []
                for observed in [False, True]:
                    tracker = tracker_for(config)
                    frozen = (state_sha256(tracker.model.state_dict()), state_sha256(tracker.native_model.state_dict()))
                    observer = QueryIncrementObserver(tracker) if observed else None
                    if observed:
                        tracker.query_posterior = observer
                    rows = []
                    for frame in range(1, 33):
                        rows.extend(tracker.update(frame, data))
                    # Force a nonempty valid past/current query even if all
                    # prefix objects happen to remain unambiguous and visible.
                    packet = tracker.query_posterior(32, dict(frame=1, index=int(data['offsets'][0])))
                    assert packet['probability'].size > 0
                    packets.append({key: packet[key].copy() for key in ['probability', 'null', 'delta']})
                    assert frozen == (state_sha256(tracker.model.state_dict()), state_sha256(tracker.native_model.state_dict()))
                    p = work / 'neck72_study_cpu_prefix' / (algorithm+'_'+front+'_'+cue+'_'+str(observed)+'.txt')
                    write_sequence_results(str(p), rows); outputs.append(p)
                    if observed:
                        assert observer.values['packets'] > 0 and observer.values['pair_cells'] > 0
                        observed_values = observer.values
                assert outputs[0].read_bytes() == outputs[1].read_bytes()
                for key in packets[0]:
                    np.testing.assert_array_equal(packets[0][key], packets[1][key])
                records.append(dict(algorithm=algorithm, frontend=front, cue=cue, frames=32,
                    original36encoder_seed0_model_only=True, tracked_outputs_exact=True,
                    nonempty_query_posteriors_and_increments_exact=True, model_weights_and_buffers_unchanged=True,
                    source_config_sha256=sha256(config_path), source_features_sha256=sha256(source_path),
                    source_transport_sha256=sha256(point_path), native_metadata=native,
                    tracks=[dict(path=str(p), sha256=sha256(p)) for p in outputs],
                    query_increment_statistics=observed_values))
    assert len(records) == 8
    assert before == {p: sha256(ROOT / p) for p in source_files()}
    assert not torch.cuda.is_initialized()
    report = dict(status='COMPLETE_CPU_NECK72_STUDY_RULES_AND_ACTUAL_PREFIX_PROOF',
        source_hashes=before, tests_passed=tests, pytest_command=command, test_log_sha256=sha256(log),
        observer_actual_old_seed0_prefix_tracks_exact=True, actual_old_prefix_records=records,
        actual_old_prefix_combinations=8, frames_each=32,
        source_scope='Previously trained native36 seed0 models, not newly fitted neck72 models. '
            'Current observer/sequence reset/backend interfaces only; no accuracy evaluation.',
        fixed_seed=0, additional_seed_work=False, GPU_work=False, new_neck72_fit=False,
        GT_read=False, new_Dev_or_Test_metrics=False, seconds=time.monotonic()-started,
        pending_actual_GPU_engineering=True, full_system_runtime_complete=False,
        goal_met=False, supported_innovations=0)
    require_storage(work); dump(output, report)
    print(report['status'], tests, 'tests;', len(records), 'actual-prefix combinations;',
          report['seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
