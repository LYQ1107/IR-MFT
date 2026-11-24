#!/usr/bin/env python3
"""Freeze the neck72 seed0 design before query compilation or head fitting."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.neck72_experiment import STUDY, context, head_id, head_specs
from tools.neck72_runtime import REGISTRY as WORKER_REGISTRY, read, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    plan, protocol, work = context()
    assert not STUDY.exists(), 'Preserve the preregistered design'
    worker_path = work / WORKER_REGISTRY
    worker = read(worker_path)
    assert worker['status'] == 'COMPLETE_CPU_NECK72_WORKER_INTERFACE_REGISTRATION_PENDING_GPU'
    assert worker['fixed_seed'] == 0 and not worker['formal_fit_started']
    assert all(sha256(ROOT / path) == value for path, value in worker['source_hashes'].items())
    for family in ['native_neck_clean_query', 'native_neck_robust_query', 'native_neck_query_bank',
                   'native_neck_mismatch_query_bank', 'native_neck_candidate_features']:
        assert not (work / family).exists(), 'Freeze design before new data/fits: ' + family
    clean_path = ROOT / 'configs/perceptual_native_neck_clean_query_20261123.json'
    robust_path = ROOT / 'configs/perceptual_native_neck_robust_query_20261123.json'
    clean, robust = read(clean_path), read(robust_path)
    differences = {key for key in set(clean) | set(robust) if clean.get(key) != robust.get(key)}
    assert differences == {'feature_schema', 'mismatch_KL_weight', 'reason', 'status'}
    assert clean['mismatch_KL_weight'] == 0 and robust['mismatch_KL_weight'] == 1
    blueprints = {}
    for algorithm, old_name in [('ordered', 'robust'), ('consensus', 'consensus')]:
        for front in ['new', 'rgb']:
            for cue in ['rgb', 'ir']:
                path = ROOT / 'configs/perceptual_native_common_test_cases_20261124' / (
                    f'NC36_{old_name}_{front}_{cue}_normal.json')
                config = read(path)
                assert config['native_seed'] == 0 and config['detection_arm'] == front and config['native_cue'] == cue
                assert config['backend'] == ('ordered_query' if algorithm == 'ordered' else 'consensus_query')
                assert not config['query_ranking'] and config['prebirth_recovery']
                blueprints[f'{algorithm}_{front}_{cue}'] = dict(path=str(path.relative_to(ROOT)),
                    sha256=sha256(path), config=config,
                    origin='Original Dev-frozen36-encoder backend constants; retain every negative result. '
                           'No new Test choice or parameter search.')
    study = dict(status='REGISTERED_BEFORE_NECK72_QUERY_HEAD_AND_NEW_DEV_WORK',
        family='neck_metric72 descriptor-conditioned identity queries', fixed_seed=0,
        additional_seed_work=False, worker_interfaces_sha256=sha256(worker_path),
        main_plan_sha256=sha256(ROOT / 'configs/perceptual_mft_20261122.json'),
        protocol_sha256=sha256(ROOT / plan['protocol']), physical_gpus=['4', '5'], maximum_cards=2,
        maximum_CPU_workers=2, heads=head_specs(work), blueprints=blueprints,
        primary=dict(head=head_id('robust', 'new', 'ir'), algorithm='consensus',
            fixed_before_new_head_Dev=True, no_Test_based_primary_promotion=True),
        fitting=dict(encoder_epochs_already_executed=72, new_encoder_fits=0,
            minimum_query_head_epochs=18, maximum_query_head_epochs=36, Dev_interval=6,
            snapshots=[6, 12, 18, 24, 30, 36],
            stopping='At common18/24/30/36, extend all8 if any of16 complete normal Dev curves '
                     'gains>0.2pp over its own best through12epochs earlier; cap36. '
                     'Use original complete-Dev HOTA, earliest snapshot on ties.',
            observations='Existing exact125830 candidate crops,2863358 local and347782 long query edges; '
                         'query-bank epochs are not complete27251frame epochs.',
            clean_and_robust_difference='Only mismatch KL coefficient0 versus1; equal model/data/seed/draws/budget.'),
        sequence_scope=dict(train=protocol['train'], Dev=protocol['dev'], Dev_frames=4834,
                            Test=protocol['historical_test'], Test_frames=15985),
        required_cases=dict(normal_curves=16, per_normal=['normal', 'missing_native_increment'],
            each_IR_normal=['current-frame_feature_mismatch'],
            primary_removals=['no_alignment', 'no_null', 'no_prebirth', 'no_covisible',
                              'no_object_files', 'no_consensus'],
            primary_paired_snapshots=['clean_no_KL', 'extra_RGB'], ByteTrack_frontends=['rgb', 'new'],
            complete_Dev_for_every_selected_Test_case=True, all7Test_regardless_of_Dev_sign=True,
            deduplication='Only byte-equivalent inference configurations, ignoring id/prose. Preserve role aliases.'),
        intervention_scope='Native identity-vector increment only. Frozen proposal frontend and RGB '
            'point/reference/quality inputs retained. Feature mismatch rolls current-frame native candidates; '
            'it is not an IR image perturbation. Missing disables the entire native-vector increment; '
            'new frontend still retains its pre-existing IR paths. No full-modality-causal claim from this study.',
        data_proof='Before encoding, reconstruct every cached Train crop from the fixed current image '
            'and compare pixels/rectangle exactly. Pin original images to all4successful fullTrain manifests.',
        engineering='Both physical4/5: actual oldDev-prefix reproduction,20Train optimizer updates for '
            'each of8matched heads, original RGB/encoder/BN unchanged and exact zero native posterior/track fallback.',
        cost=dict(required_full_system_FPS=True,
            cached_replay_and_native_encoder_costs_are_components_only=True,
            full_system_benchmark_not_yet_implemented=True,
            no_sum_of_component_FPS=True, source_extraction_and_IR_generation_exclusions_must_be_explicit=True),
        resources=dict(work_cap_bytes=64*2**30, independent_free_reserve_bytes=40*2**30,
            MemAvailable_reserve_bytes=40*2**30, decoded_producer_bound_bytes=8*2**30,
            maximum_producers=4, mandatory_production_compile_barrier=True, no_image_or_FPN_copies=True),
        reporting='All39 summary metrics, all47actual official fields, every sequence and COMBINED, '
            'unmodified276-columnCSV, actual query-increment/route statistics, parameters, costs and failure provenance.',
        limitations=['HistoricalTest already exposed; no blind-test claim.',
            'Original RGB encoder pretraining included currentDev; new fitting is Train-only.',
            'PseudoIR derives from currentRGB, not an independent physical thermal sensor.',
            'This is a representation repair within an existing fusion family, not another architecture.',
            'Full-system runtime and any broader actual-image modality intervention remain goal requirements.'],
        supports_new_architecture_count=False, goal_met=False, supported_innovations=0)
    require_storage(work); dump(STUDY, study)
    print(study['status'], 'matched_heads', len(study['heads']), 'normal_Dev_curves', 16)


if __name__ == '__main__':
    main()
