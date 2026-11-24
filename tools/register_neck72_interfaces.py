#!/usr/bin/env python3
"""Register CPU-verified neck72 worker interfaces, not a complete study queue."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.neck72_runtime import REGISTRY, read, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    protocol = read(ROOT / plan['protocol']); work = Path(plan['work_root'])
    path = work / REGISTRY
    assert not path.exists(), 'Preserve previous worker-interface registration'
    proof_path = work / 'neck72_worker_interfaces_CPU_proof.json'
    proof = read(proof_path)
    assert proof['status'] == 'COMPLETE_CPU_NECK72_WORKER_INTERFACES_WITH_REAL_CACHED_DEV_AND_TRAIN_PIXELS'
    assert proof['tests_passed'] == 9 and proof['source_views'] == 8 and proof['modes_each'] == 3
    assert not proof['GPU_work'] and not proof['formal_fit_started']
    assert not proof['new_Dev_or_Test_accuracy'] and not proof['full_study_scheduler_implemented']
    assert all(sha256(ROOT / p) == h for p, h in proof['source_hashes'].items())
    sources = dict(proof['source_hashes'])
    sources['tools/register_neck72_interfaces.py'] = sha256(__file__)
    artifacts = {str(proof_path): sha256(proof_path), **proof['artifact_hashes']}
    observation_counts, crop_bounds = {}, {}
    for sequence in protocol['train']:
        index_path = work / 'native_candidate_index' / sequence / 'complete.json'
        index = read(index_path)
        assert index['status'] == 'COMPLETE_EXACT_TRAIN_INDEX' and index['every_saved_geometry_field_exact']
        observation_counts[sequence] = index['observations']
        crop_bounds[sequence] = index['observations'] * (128*256*3 + 2*512*4) + 2**30
        assert crop_bounds[sequence] <= 8 * 2**30
        for folder in ['native_candidate_index', 'native_candidate_crops']:
            metadata = work / folder / sequence / 'complete.json'
            artifacts[str(metadata)] = sha256(metadata)
        reference = work / 'native_candidate_features' / sequence / 'rgb/complete.json'
        artifacts[str(reference)] = sha256(reference)
    for cue in ['rgb', 'ir']:
        for name in [work / 'native_neck_metric' / (cue + '_s0/complete.json'),
                     work / 'native_neck_train_features' / (cue + '_s0/complete.json')]:
            artifacts[str(name)] = sha256(name)
        name = work / 'native_neck_alignment' / (cue + '_s0.npz')
        artifacts[str(name)] = sha256(name)
    name = work / 'native_neck_alignment/complete.json'; artifacts[str(name)] = sha256(name)
    for arm in ['new', 'rgb']:
        name = work / 'native_RGB_prior' / (arm + '_s0/complete.json')
        artifacts[str(name)] = sha256(name)
    value = dict(status='COMPLETE_CPU_NECK72_WORKER_INTERFACE_REGISTRATION_PENDING_GPU',
        source_hashes=sources, artifact_hashes=artifacts, fixed_seed=0,
        additional_seed_work=False, physical_gpus=['4', '5'], maximum_physical_cards=2,
        actual_existing_encoder_epochs=72, fit_Train_videos=protocol['train'],
        candidate_observations=observation_counts, decoded_candidate_RAM_bound_bytes=crop_bounds,
        largest_decoded_candidate_RAM_bound_bytes=max(crop_bounds.values()),
        future_data_worker_RAM_bound_bytes=8*2**30, host_MemAvailable_reserve_bytes=40*2**30,
        CPU_plus_GPU_producer_maximum=4, independent_work_disk_reserve_bytes=40*2**30,
        total_work_cap_bytes=64*2**30, dataset_or_FPN_copies=0,
        CPU_interface_proof_sha256=sha256(proof_path), old_Dev_seed0_layout_fixed=True,
        original_pixel_and_optimizer_step_computation_unchanged=True,
        actual_GPU_engineering_pending=True, GPU_workers_started=0, formal_fit_started=False,
        full_study_scheduler_implemented=False, complete_study_queued=False,
        pending=['Register complete seed0 head/Dev/convergence/removal/selection/Test/runtime/export protocol.',
            'Implement complete bounded controller, verify CPU branches, and queue after actual CGR release.',
            'Verify original cached candidate crops against fixed current Train images before pixel encoding.',
            'Both4/5: real GPU source,20Train updates for8matched head conditions, frozen RGB/BN and exact zero fallback.',
            'Complete original Train-only banks, matched fitting, all fullDev and frozen all7Test/reporting.'],
        scope='Worker interfaces only. Eight existing cached Dev views across three readout variants '
            'match the original formula exactly, and four actual Train images passed CPU forwards. '
            'Not a current-GPU pixel reproduction, a head fit, accuracy gain, new architecture or contribution.',
        goal_met=False, supported_innovations=0)
    require_storage(work); dump(path, value)
    print(value['status'], 'sources', len(sources), 'Train_candidates', sum(observation_counts.values()),
          'GPU_workers_started', value['GPU_workers_started'])


if __name__ == '__main__':
    main()
