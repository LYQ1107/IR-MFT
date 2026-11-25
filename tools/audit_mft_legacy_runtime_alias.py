#!/usr/bin/env python3
"""Bind one legacy alias to its existing exact-condition historical NN profile.

CPU only: cached packet and source hashes, no model loading or new benchmark.
Wrong/missing inputs, different weights and different policies are not aliases.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/fusion_research'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicitly hide CUDA for this file-only audit')
    started = time.monotonic()
    plan_path = ROOT / 'configs/perceptual_mft_20261122.json'
    plan = read(plan_path)
    work = Path(plan['work_root'])
    spec_path = ROOT / 'configs/perceptual_structural_legacy_backfill_test_20261124.json'
    spec = read(spec_path)
    base = ROOT / 'configs/perceptual_structural_relative_20261123.json'
    parent = ROOT / 'configs/perceptual_correlation_repair_20261123.json'
    phase = ROOT / 'configs/perceptual_structural_nocycle_20261123.json'
    benchmark = ROOT / 'tools/benchmark_structural_nocycle.py'
    profile_paths = {
        mode: DOCS / ('PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_SINGLE_METHOD_RUNTIME' + suffix + '.json')
        for mode, suffix in [('live', ''), ('memory', '_MEMORY_COMPONENTS')]}
    profiles = {mode: read(path) for mode, path in profile_paths.items()}
    files = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('A historical source changed: ' + str(path))
        if str(path) in files and files[str(path)] != digest:
            raise ValueError('Source changed during this audit')
        files[str(path)] = digest
        return digest

    for path in [Path(__file__), plan_path, spec_path, base, parent, phase]:
        bind(path)
    for mode, profile in profiles.items():
        bind(profile_paths[mode])
        if (profile['status'] != 'COMPLETE_SINGLE_METHOD_STRUCTURAL_B4_THROUGHPUT'
                or profile['fixed_fitted_seed'] != 0 or profile['additional_seed_work']
                or profile['GT_read'] or profile['accuracy_selection']
                or profile['physical_gpu'] != '4'
                or profile['base_config_sha256'] != bind(base)
                or profile['refinement_plan_sha256'] != bind(phase)
                or profile['raw_images_preloaded_in_RAM'] != (mode == 'memory')):
            raise ValueError('Original fixed historical profile scope differs')
        bind(benchmark, profile['source_sha256'])
        bind(ROOT / 'configs/perceptual_gpu_handoff_20261123.json', profile['gpu_handoff_policy_sha256'])
        bind(ROOT / 'tools/perceptual_gpu_handoff.py', profile['gpu_handoff_helper_sha256'])
    rgb = read(parent)
    if read(base)['parent_plan_sha256'] != bind(parent):
        raise ValueError('Original parent configuration changed')
    bind(rgb['source_RGB_checkpoint'], rgb['source_RGB_sha256'])
    model_map = {model['id']: model for model in spec['models']}
    rows = []
    for case in spec['cases']:
        model = model_map[case['model_id']]
        candidates = []
        if not model.get('baseline') and not model.get('intervention') and case['output_policy'] == 'raw':
            for row in profiles['live']['rows']:
                if (row['checkpoint_sha256'] == model['checkpoint_sha256']
                        and row['cue'] == model['cue'] and row['variant'] == model['variant']
                        and row['epoch'] == model['epoch']):
                    candidates.append(row['id'])
        rows.append(dict(legacy_case=case['id'], historical_candidates=candidates,
            reason='Exact historical non-intervened weight/cue/variant/epoch/raw-policy signature.' if candidates else
                   'No exact non-intervened signature in these profiles; no cross-model or intervention cost substitution.'))
    matched = [row for row in rows if row['historical_candidates']]
    if len(matched) != 1 or matched[0]['legacy_case'] != 'LB_restored_cycle_IR_reference_raw':
        raise ValueError('This frozen audit permits only the known exact legacy alias')
    profile_id = matched[0]['historical_candidates'][0]
    selected = {mode: next(row for row in profile['rows'] if row['id'] == profile_id)
                for mode, profile in profiles.items()}
    model = model_map['LB_restored_cycle_IR_reference']
    if not model['expected_hard_cycle'] or model['variant'] != 'full' or model['seed'] != 0:
        raise ValueError('The actual original hard-cycle model is required')
    bind(model['checkpoint'], model['checkpoint_sha256'])
    checks = []
    for sequence in ['BT-005', 'SN-013']:
        old_dir = work / 'structural_relative_predictions/dev' / sequence
        new_dir = work / 'legacy_structural_backfill/structural_relative_predictions/dev' / sequence / 'prefix_check_32'
        old, new = read(old_dir / 'complete.json'), read(new_dir / 'complete.json')
        bind(old_dir / 'complete.json')
        bind(new_dir / 'complete.json')
        for key in ['config_sha256', 'calibration_sha256', 'source_RGB_sha256', 'global_reference']:
            if old[key] != new[key]:
                raise ValueError('Actual source/configuration/reference differs')
        if old['config_sha256'] != bind(base):
            raise ValueError('Actual producer and measured architecture differ')
        bind(work / 'correlation_cache/calibration.json', old['calibration_sha256'])
        if (old['GT_read'] or new['GT_read'] or old['future_frame_input'] or new['future_frame_input']
                or old['source_RGB_sha256'] != rgb['source_RGB_sha256']
                or old['models']['SR_ir_full_s0_e06']['metadata'] != new['models'][model['id']]['metadata']):
            raise ValueError('Actual model lineage or causal source differs')
        shared = set(old['source_hashes']) & set(new['source_hashes'])
        if len(shared) != 11:
            raise ValueError('Require the original eleven common inference dependencies')
        for name in shared:
            if old['source_hashes'][name] != new['source_hashes'][name]:
                raise ValueError('Historical implementation differs')
            bind(ROOT / name, old['source_hashes'][name])
        old_path = old_dir / 'SR_ir_full_s0_e06_raw.npz'
        new_path = new_dir / 'LB_restored_cycle_IR_reference_raw.npz'
        for mode, profile in selected.items():
            item = next(row for row in profile['per_sequence'] if row['sequence'] == sequence)
            if not item['source32_arrays_exact'] or item['timed_frames'] != 24 or item['warmup_frames'] != 8:
                raise ValueError('Historical measured prefix was not qualified')
            bind(old_path, item['source_packet_sha256'])
        bind(new_path)
        with np.load(old_path, allow_pickle=False) as arrays:
            count = int(arrays['offsets'][32])
            expected = dict(frame_ids=arrays['frame_ids'][:32].copy(), offsets=arrays['offsets'][:33].copy(),
                boxes=arrays['boxes'][:count].copy(), scores=arrays['scores'][:count].copy())
        with np.load(new_path, allow_pickle=False) as arrays:
            for key, value in expected.items():
                np.testing.assert_array_equal(arrays[key], value)
        checks.append(dict(sequence=sequence, frames=32, all_four_detection_arrays_exact=True,
            common_inference_sources_exact=11, original_model_metadata_exact=True))
    for profile in selected.values():
        if (profile['checkpoint_sha256'] != model['checkpoint_sha256'] or profile['adapter_parameters'] != 145600
                or profile['encoder_passes'] != 2 or profile['timed_frames'] != 48
                or not math.isclose(profile['FPS'], 48 / sum(x['seconds'] for x in profile['per_sequence']), rel_tol=1e-12)):
            raise ValueError('Original model cost, frames or aggregation differs')
    schedule = work / 'structural_nocycle_runtime_schedule.json'
    bind(schedule)
    state = read(schedule)
    if state['status'] != 'COMPLETE_BOTH_RUNTIME_CONDITIONS_NO_GPU_CONFLICT':
        raise ValueError('Historical profile controller did not record completion')
    if any(sha(path) != value for path, value in files.items()):
        raise ValueError('Source changed while auditing')
    report = dict(status='COMPLETE_ONE_EXACT_HISTORICAL_PROFILE_ALIAS_EIGHT_CASES_UNMATCHED',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_hashes=files,
        legacy_case=matched[0]['legacy_case'], historical_profile_id=profile_id,
        checkpoint_sha256=model['checkpoint_sha256'], prefix_checks=checks, profiles=selected,
        coverage=rows, historical_schedule=state, historical_numeric_worker_exit_code=None,
        historical_numeric_exit_scope='The old completion state saved no numeric worker exit; do not backfill0.',
        new_GPU_or_NN_work=False, new_benchmark_frames=0, new_Dev_or_Test_accuracy=False,
        fixed_seed=0, additional_seed_work=False, goal_met=False, supported_innovations=0,
        seconds=time.monotonic() - started, peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        scope='Historical reference for this exact model/policy, not a new cost measurement or all9 cost completion. '
              'Two Dev videos x8warmup+24timed frames, B4 throughput. Includes preparation/NN/NMS/ByteTrack; '
              'excludes IRgeneration/model loading/serialization and gate-observer overhead. Memory mode excludes preload/reads. '
              'Shared host/filecache uncontrolled. Never substitute this profile for wrong/missing IR or native-pyramid qualityIR6.')
    args.output_directory.mkdir(exist_ok=False)
    (args.output_directory / 'complete.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], matched_cases=1, unmatched_cases=8,
        live_historical_B4_FPS=selected['live']['FPS'], memory_historical_B4_FPS=selected['memory']['FPS'],
        new_GPU_or_NN_work=False, seconds=report['seconds'])))


if __name__ == '__main__':
    main()
