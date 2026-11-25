#!/usr/bin/env python3
"""Independently preserve every frozen full-Train Test result and its provenance."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_evaluation import official_csv47, validate_mot
from tools.fusion_memory_observation import LENGTHS
from tools.native_pyramid_experiment import read
from tools.neck72_runtime import require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent original-file audit is CPU only')
    started = time.monotonic()
    planpath = ROOT / 'configs/perceptual_mft_20261122.json'
    plan = read(planpath)
    work = Path(plan['work_root'])
    require_host_memory(8 * 2**30, maximum_workers=1)
    require_storage(work, growth_bytes=4 * 2**20)
    protocol = read(ROOT / plan['protocol'])
    specpath = ROOT / 'configs/perceptual_structural_fulltrain_test_20261123.json'
    spec = read(specpath)
    statepath = work / 'structural_full_train_test_state.json'
    state = read(statepath)
    freezepath = work / 'structural_full_train_Test_selection_frozen.json'
    freeze = read(freezepath)
    devpath = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_FULL_TRAIN_DEV_ANALYSIS.json'
    dev = read(devpath)
    dev47path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261125_FULL_TRAIN_DEV47_AUDIT.json'
    dev47 = read(dev47path)
    if (state['status'] != 'COMPLETE' or len(state['jobs']) != 13 or
            any(j['status'] != 'COMPLETE' or j['exit_code'] != 0 for j in state['jobs'].values()) or
            any(p['status'] != 'COMPLETE' or p['source_exit_code'] != 0 or p['full_exit_code'] != 0
                for p in state['producers'].values())):
        raise ValueError('All original thirteen objects and seven source producers must finish')
    if (state['specification_sha256'] != sha256(specpath) or freeze['specification_sha256'] != sha256(specpath) or
            freeze['complete_Dev_analysis_sha256'] != sha256(devpath) or
            spec['complete_Dev_analysis_sha256'] != sha256(devpath) or not freeze['complete50Dev_before_Test'] or
            spec['Test_accuracy_used_for_selection'] or dev['Test_accuracy_used_for_selection'] or
            freeze['roles'] != spec['roles'] or freeze['primary'] != spec['roles']['primary_IR'] or
            dev47['original_Dev_choice_sha256'] != sha256(devpath)):
        raise ValueError('The original complete Dev choice and Test freeze must remain exact')
    files = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('Original file changed: ' + str(path))
        if str(path) in files and files[str(path)] != digest:
            raise ValueError('Original file changed during audit: ' + str(path))
        files[str(path)] = digest
        return digest

    for path in [planpath, ROOT / plan['protocol'], specpath, statepath, freezepath, devpath, dev47path]:
        bind(path)
    for name, digest in state['source_hashes'].items():
        bind(ROOT / name, digest)
    expected_lengths = LENGTHS['test']
    if list(expected_lengths) != protocol['historical_test'] or sum(expected_lengths.values()) != 15985:
        raise ValueError('Original seven Test videos and all frames are required')
    producer_dir = work / 'full_train_observations/structural_relative_predictions/test'
    for sequence, producer in state['producers'].items():
        path = producer_dir / sequence / 'complete.json'
        bind(path, producer['complete_sha256'])
    schema, counts = official_schema(plan['evaluation']['TrackEval_root'])
    expected = {r['id']: r for r in spec['cases']}
    devrows = {r['id']: r for r in dev['rows']}
    if len(schema) != 47 or set(expected) != set(state['jobs']):
        raise ValueError('All original roles and official fields are mandatory')
    rows = []
    csv_sources = {}
    checked_sources = set()
    for name, job in state['jobs'].items():
        directory = work / 'structural_full_train_test' / name
        path = directory / 'metrics.json'
        value = read(path)
        manifest = value['manifest']
        config = manifest['config']
        old = devrows[expected[name]['source_Dev_case']]
        if (value['status'] != 'COMPLETE' or manifest['split'] != 'test' or manifest['frames'] != 15985 or
                manifest['sequences'] != protocol['historical_test'] or config['seed'] != 0 or
                manifest['config_sha256'] != job['sha256'] or config != read(job['config']) or
                manifest['plan_sha256'] != sha256(planpath) or
                manifest['replay_source_sha256'] != sha256(ROOT / 'tools/replay_structural_full_train.py') or
                manifest['byte_tracker_source_sha256'] != sha256(ROOT / 'yolox/tracker/byte_tracker.py') or
                any(manifest[k] for k in ['GT_inference_input', 'future_frame_input', 'previous_outputs_rewritten']) or
                old['metrics_sha256'] != expected[name]['source_Dev_metrics_sha256'] or
                any(config.get(k) != old['config'].get(k) for k in
                    ['model_id', 'policy', 'epoch', 'components', 'cue', 'intervention', 'baseline', 'fresh_RGB'])):
            raise ValueError('Original frozen model, policy, split or full coverage changed: ' + name)
        bind(path)
        bind(work / 'structural_full_train_dev' / expected[name]['source_Dev_case'] / 'metrics.json',
            expected[name]['source_Dev_metrics_sha256'])
        bind(directory / 'inference_manifest.json')
        if read(directory / 'inference_manifest.json') != manifest:
            raise ValueError('The before-GT manifest and final metric manifest disagree')
        bind(job['config'], job['sha256'])
        if {s: c['frames'] for s, c in manifest['coverage'].items()} != expected_lengths:
            raise ValueError('Partial Test video coverage: ' + name)
        for sequence, coverage in manifest['coverage'].items():
            track = directory / 'track_results' / (sequence + '.txt')
            bind(track, coverage['output_sha256'])
            validate_mot(track, expected_lengths[sequence], coverage['output_rows'])
            source = manifest['sources'][sequence]
            if source['path'] not in checked_sources:
                bind(source['path'], source['sha256'])
                checked_sources.add(source['path'])
            elif files[str(Path(source['path']).resolve())] != source['sha256']:
                raise ValueError('A shared original proposal source changed')
        if not config.get('baseline'):
            for model in manifest['model_sources'].values():
                if model['checkpoint_sha256'] != old['checkpoint_sha256']:
                    raise ValueError('Test did not use its frozen Dev checkpoint')
                if model['checkpoint'] not in checked_sources:
                    bind(model['checkpoint'], model['checkpoint_sha256'])
                    checked_sources.add(model['checkpoint'])
        csv = directory / 'trackeval_output/pedestrian_detailed.csv'
        bind(csv)
        parsed = official_csv47(csv, schema, 'test', value['trackeval'])
        rows.append(dict(id=name, source_Dev_case=expected[name]['source_Dev_case'], config=config,
            metrics39=value['trackeval'], **parsed, metrics_file=str(path), metrics_sha256=files[str(path.resolve())],
            coverage=manifest['coverage'], replay_seconds=manifest['replay_seconds'],
            replay_FPS_scope=manifest['replay_FPS_scope']))
        csv_sources[name] = csv
    by_id = {r['id']: r for r in rows}
    roles = spec['roles']
    hota = lambda name: by_id[name]['combined47']['HOTA']
    primary = roles['primary_IR']
    primary_hota = hota(primary)
    reference = roles['source_RGB_fresh' if by_id[primary]['config']['policy'] == 'raw' else 'source_RGB_canonical']
    off = by_id[roles['IR_unavailable']]
    if any(off[k] != by_id[reference][k] for k in ['combined47', 'per_sequence47', 'official_CSV_sha256']):
        raise ValueError('Unavailable IR failed its all47/whole CSV RGB identity check')
    if any(off['coverage'][s]['output_sha256'] != by_id[reference]['coverage'][s]['output_sha256']
            for s in expected_lengths):
        raise ValueError('Unavailable IR failed its full MOT identity check')
    dev_removals = {r['mechanism']: r for r in dev47['removals']}
    removals = [dict(mechanism=m, paired=roles['no_' + m + '_paired'], Dev_best=roles['no_' + m + '_Dev_best'],
        primary_minus_paired_Test_HOTA=primary_hota-hota(roles['no_' + m + '_paired']),
        primary_minus_removal_own_Dev_best_Test_HOTA=primary_hota-hota(roles['no_' + m + '_Dev_best']),
        original_Dev_effects=dev_removals[m]) for m in ['S', 'C', 'R']]
    per_sequence = []
    for i, sequence in enumerate(expected_lengths):
        values = {role: by_id[roles[role]]['per_sequence47'][i]['HOTA']
            for role in ['primary_IR', 'matched_rgb', 'matched_gray', 'source_RGB_canonical', 'source_RGB_fresh', 'IR_image_roll']}
        per_sequence.append(dict(sequence=sequence, HOTA=values,
            primary_minus_matched_RGB_HOTA=values['primary_IR']-values['matched_rgb'],
            primary_minus_matched_gray_HOTA=values['primary_IR']-values['matched_gray']))
    runtime = {}
    for suffix in ['RUNTIME', 'RUNTIME_MEMORY_COMPONENTS']:
        path = ROOT / f'docs/fusion_research/PERCEPTUAL_MFT_20261123_FULL_TRAIN_SINGLE_METHOD_{suffix}.json'
        cost = read(path)
        bind(path)
        if (cost['status'] != 'COMPLETE_FROZEN_FULL_TRAIN_METHOD_RUNTIME' or cost['GT_read'] or
                cost['Test_specification_sha256'] != sha256(specpath) or len(cost['records']) != 10):
            raise ValueError('Original separate current-image B4 cost proof changed')
        for row in cost['records']:
            config = by_id[row['id']]['config']
            if row['model_id'] != config['model_id'] or row['policy'] != config.get('policy', 'raw'):
                raise ValueError('Independent measured cost and frozen accuracy model differ')
        runtime[suffix] = dict(path=str(path), sha256=files[str(path.resolve())], scope=cost['scope'],
            records=cost['records'], actually_measured_case_count=10,
            cases_without_separate_measurement=sorted(set(by_id)-{r['id'] for r in cost['records']}))
    if any(sha256(p) != s for p, s in files.items()):
        raise ValueError('Original evidence changed during independent audit')
    value = dict(status='COMPLETE_INDEPENDENT_ALL13_FULL_TRAIN_ALL7_TEST_OFFICIAL47_AUDIT',
        source_auditor_sha256=sha256(__file__), source_file_hashes=files, original_Dev_choice_sha256=sha256(devpath),
        Test_specification_sha256=sha256(specpath), Test_freeze_sha256=sha256(freezepath), roles=roles,
        complete_cases=13, frames_each=15985, sequences=protocol['historical_test'], official_fields=47,
        official_CSV_columns=276, official_schema=schema, official_family_field_counts=counts, rows=rows,
        frozen_primary=primary, primary_Test_HOTA=primary_hota, original_rounded_baseline_Test_HOTA=29.529,
        primary_minus_fixed_rounded_baseline_Test_HOTA=primary_hota-29.529, target_Test_HOTA=34.529,
        numeric_threshold_met=primary_hota >= 34.529,
        primary_minus_canonical_RGB_Test_HOTA=primary_hota-hota(roles['source_RGB_canonical']),
        primary_minus_fresh_RGB_Test_HOTA=primary_hota-hota(roles['source_RGB_fresh']),
        primary_minus_matched_RGB_Test_HOTA=primary_hota-hota(roles['matched_rgb']),
        primary_minus_matched_gray_Test_HOTA=primary_hota-hota(roles['matched_gray']),
        primary_minus_wrong_IR_Test_HOTA=primary_hota-hota(roles['IR_image_roll']),
        missing_IR_all47_CSV_and_MOT_exact_to_policy_RGB=True, per_sequence_HOTA_comparisons=per_sequence,
        removals=removals, independent_original_B4_costs=runtime,
        original_choice_preserved=True, Test_primary_reselection=False, actual_full_Train_epochs_each=3,
        optimizer_updates_each=20736, fixed_seed=0, additional_seed_work=False, new_GPU_work=False,
        historical_Test_exposed=True, source_RGB_pretraining_saw_Dev=True, auxiliary_is_RGB_derived=True,
        goal_met=False, supported_innovations=0, seconds=time.monotonic()-started,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        scope='Independent extraction of all13frozen/all7Test15985frame original276column CSVs. '
            'All47pervideo/combined,actualMOT/proposal/checkpoint/sourcehashes and unchanged Devfreeze audited. '
            'Original10case independent B4prefix costs retained;three unmeasured cases explicitly unknown. '
            'Cached replay is not NN cost. No Test reselection,new fit,IRregeneration,extra seed,novelty or completed-goal claim.')
    if value['peak_process_RSS_GiB'] > 8:
        raise RuntimeError('Own independent CPU audit exceeded8GiB')
    output = args.output_directory
    output.mkdir(exist_ok=False)
    table = output / 'official_CSVs'
    table.mkdir()
    for name, path in csv_sources.items():
        (table / (name + '.csv')).write_bytes(path.read_bytes())
    (output / 'complete.json').write_text(json.dumps(value, indent=2) + '\n')
    lines = ['# Full-Train frozen family: independent complete Test audit', '', value['scope'], '',
        f'Frozen primary: {primary}; HOTA {primary_hota:.6f}.',
        f"Versus fixed original RGB29.529: {value['primary_minus_fixed_rounded_baseline_Test_HOTA']:+.6f}.",
        f"Versus frozen own-Dev-best RGB: {value['primary_minus_matched_RGB_Test_HOTA']:+.6f}.", '',
        '| Frozen case | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| ' + row['id'] + ' | ' + ' | '.join(f"{row['combined47'][k]:.6f}"
            for k in ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA']) + f" | {row['combined47']['IDSW']} |")
    lines += ['', '| Removed mechanism | Primary minus paired Test | Primary minus own-Dev-best removal Test |',
        '| --- | ---: | ---: |']
    for row in removals:
        lines.append(f"| {row['mechanism']} | {row['primary_minus_paired_Test_HOTA']:+.6f} | "
            f"{row['primary_minus_removal_own_Dev_best_Test_HOTA']:+.6f} |")
    lines += ['', 'All original negative cases retained. Three complete Train traversals per fit,not72full epochs. '
        'HistoricalTestandpretrainingDev exposure remain. Independent contribution count0;globalgoal remains active.']
    (output / 'analysis.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=value['status'], HOTA=primary_hota,
        versus_original=value['primary_minus_fixed_rounded_baseline_Test_HOTA'],
        versus_matched_RGB=value['primary_minus_matched_RGB_Test_HOTA'],
        numeric_threshold_met=value['numeric_threshold_met'], seconds=value['seconds'])), flush=True)


if __name__ == '__main__':
    main()
