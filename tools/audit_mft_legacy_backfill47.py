#!/usr/bin/env python3
"""Audit the nine original structural backfill cases without changing selection."""
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
from tools.neck72_runtime import require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import sha256
from tools.structural_legacy_backfill import (
    BASE, CASES, DOCS, PREDICTIONS, REGISTRY, SPEC, check_producer,
    context, read, verified_spec,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This original-file audit is CPU only')
    started = time.monotonic()
    plan, work, protocol = context()
    require_host_memory(8 * 2**30, maximum_workers=1)
    require_storage(work, growth_bytes=4 * 2**20)
    spec = verified_spec()
    statepath = work / 'structural_legacy_backfill_state.json'
    state = read(statepath)
    if (state['status'] != 'COMPLETE_ALL9_FORMAL_BACKFILL_CASES' or
            len(state['jobs']) != 9 or
            any(j['status'] != 'COMPLETE' or j['exit_code'] != 0 for j in state['jobs'].values()) or
            any(p['status'] != 'COMPLETE' for p in state['producers'].values()) or
            (Path('/proc') / str(state['controller_pid'])).exists()):
        raise ValueError('Wait for all nine official jobs, full export and actual controller exit')
    frozenpath = work / 'structural_legacy_backfill_selection_frozen.json'
    freeze = read(frozenpath)
    devpath = DOCS / 'PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_ANALYSIS.json'
    devstatepath = work / 'structural_relative_dev_state.json'
    dev, devstate = read(devpath), read(devstatepath)
    if (freeze['specification_sha256'] != sha256(SPEC) or
            freeze['registry_sha256'] != sha256(REGISTRY) or
            not freeze['complete_original59Dev_before_Test'] or
            not freeze['current_primary_unchanged'] or freeze['fixed_seed'] != 0 or
            state['specification_sha256'] != sha256(SPEC) or
            spec['original59Dev_analysis_sha256'] != sha256(devpath) or
            spec['original59Dev_state_sha256'] != sha256(devstatepath) or
            devstate['status'] != 'COMPLETE' or len(devstate['jobs']) != 59 or
            any(j['status'] != 'COMPLETE' for j in devstate['jobs'].values()) or
            state['source_hashes'] != spec['source_hashes']):
        raise ValueError('The original complete Dev freeze or source registration changed')
    schema, counts = official_schema(plan['evaluation']['TrackEval_root'])
    lengths = LENGTHS['test']
    if list(lengths) != protocol['historical_test'] or sum(lengths.values()) != 15985:
        raise ValueError('All seven original videos and frames are required')
    files = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('Original evidence changed: ' + str(path))
        if str(path) in files and files[str(path)] != digest:
            raise ValueError('Evidence changed during audit: ' + str(path))
        files[str(path)] = digest
        return digest

    for path in [ROOT / 'configs/perceptual_mft_20261122.json', ROOT / plan['protocol'],
                 BASE, SPEC, REGISTRY, statepath, frozenpath, devpath, devstatepath, Path(__file__)]:
        bind(path)
    for path, digest in spec['source_hashes'].items():
        bind(ROOT / path, digest)
    reportpath = DOCS / 'PERCEPTUAL_MFT_20261124_STRUCTURAL_LEGACY_BACKFILL_TEST.json'
    report = read(reportpath)
    bind(reportpath)
    if (report['status'] != 'COMPLETE_INITIAL_STRUCTURAL_FORMAL_ALL7_TEST_BACKFILL' or
            report['complete_state_sha256'] != sha256(statepath) or
            report['roles'] != spec['roles'] or len(report['rows']) != 9):
        raise ValueError('The complete original publication must precede this audit')
    for seq, producer in state['producers'].items():
        if (Path('/proc') / str(producer['pid'])).exists():
            raise ValueError('An original NN producer has not exited')
        folder = work / PREDICTIONS / 'test' / seq
        meta = check_producer(folder, spec, lengths[seq])
        bind(folder / 'complete.json', producer['complete_sha256'])
        if meta['physical_gpu'] not in ['4', '5']:
            raise ValueError('Unexpected physical GPU')
        for output in meta['outputs'].values():
            bind(output['path'], output['sha256'])
        check_producer(folder / 'source_check_8', spec, 8, source=True)
        bind(folder / 'source_check_8/complete.json')
    rows, csv_sources = [], {}
    exported = {r['id']: r for r in report['rows']}
    if set(exported) != set(state['jobs']) or set(exported) != {r['id'] for r in spec['cases']}:
        raise ValueError('A frozen negative or control case is missing')
    for case in spec['cases']:
        name, job = case['id'], state['jobs'][case['id']]
        if (Path('/proc') / str(job['pid'])).exists():
            raise ValueError('An original official worker has not exited')
        folder = work / 'structural_legacy_backfill_test' / name
        path = folder / 'metrics.json'
        value = read(path)
        manifest, config = value['manifest'], value['manifest']['config']
        if (value['status'] != 'COMPLETE' or manifest['split'] != 'test' or
                manifest['frames'] != 15985 or manifest['sequences'] != list(lengths) or
                config != read(job['config']) or config['seed'] != 0 or
                manifest['config_sha256'] != job['sha256'] or
                any(config.get(k) != v for k, v in case.items()) or
                manifest['plan_sha256'] != sha256(ROOT / 'configs/perceptual_mft_20261122.json') or
                manifest['replay_source_sha256'] != sha256(ROOT / 'tools/replay_structural_nocycle_test.py') or
                manifest['byte_tracker_source_sha256'] != sha256(ROOT / 'yolox/tracker/byte_tracker.py') or
                any(manifest[k] for k in ['GT_inference_input', 'future_frame_input', 'previous_outputs_rewritten']) or
                value['trackeval'] != exported[name]['metrics'] or
                {s: c['frames'] for s, c in manifest['coverage'].items()} != lengths):
            raise ValueError('Frozen recipe, inference contract or full coverage differs: ' + name)
        bind(path, exported[name]['metrics_sha256'])
        bind(job['config'], job['sha256'])
        bind(folder / 'inference_manifest.json')
        if read(folder / 'inference_manifest.json') != manifest:
            raise ValueError('The before-GT and final manifests differ')
        for seq, coverage in manifest['coverage'].items():
            track = folder / 'track_results' / (seq + '.txt')
            bind(track, coverage['output_sha256'])
            validate_mot(track, lengths[seq], coverage['output_rows'])
            source = manifest['sources'][seq]
            bind(source['path'], source['sha256'])
        if not config.get('baseline'):
            bind(config['checkpoint'], config['checkpoint_sha256'])
            if any(m['checkpoint_sha256'] != config['checkpoint_sha256'] or
                   m['checkpoint'] != config['checkpoint'] for m in manifest['model_sources'].values()):
                raise ValueError('A Test checkpoint differs from its frozen recipe')
        csv = folder / 'trackeval_output/pedestrian_detailed.csv'
        bind(csv, exported[name]['official_detailed_csv_sha256'])
        bind(ROOT / exported[name]['official_detailed_csv_file'], sha256(csv))
        rows.append(dict(id=name, config=config, metrics39=value['trackeval'],
            **official_csv47(csv, schema, 'test', value['trackeval']),
            coverage=manifest['coverage'], metrics_sha256=files[str(path.resolve())],
            replay_seconds=manifest['replay_seconds'], replay_FPS_scope=manifest['replay_FPS_scope']))
        csv_sources[name] = csv
    by_id, roles = {r['id']: r for r in rows}, spec['roles']
    off, fresh, anchor = [by_id[roles[k]] for k in
                          ['IR_unavailable', 'source_RGB_fresh', 'restored_cycle_IR_reference']]
    if any(off[k] != fresh[k] for k in ['combined47', 'per_sequence47', 'official_CSV_sha256']):
        raise ValueError('Unavailable IR is not identical to original fresh RGB')
    if any(off['coverage'][s]['output_sha256'] != fresh['coverage'][s]['output_sha256'] for s in lengths):
        raise ValueError('Unavailable IR changed a complete MOT file')
    parentpath = DOCS / 'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_TEST.json'
    bind(parentpath, spec['parent_Test_report_sha256'])
    parent = next(r for r in read(parentpath)['rows'] if r['id'] == spec['parent_reference_case'])
    parentdir = work / 'structural_nocycle_test' / spec['parent_reference_case']
    bind(parentdir / 'metrics.json', parent['metrics_sha256'])
    if (anchor['metrics39'] != parent['metrics'] or
            anchor['official_CSV_sha256'] != parent['official_detailed_csv_sha256']):
        raise ValueError('Restored-cycle anchor did not reproduce its original official table')
    for seq in lengths:
        expected = parent['manifest']['coverage'][seq]['output_sha256']
        bind(parentdir / 'track_results' / (seq + '.txt'), expected)
        if anchor['coverage'][seq]['output_sha256'] != expected:
            raise ValueError('Restored-cycle anchor did not reproduce its original MOT')
    hota = lambda role: by_id[roles[role]]['combined47']['HOTA']
    devrows = {r['role']: r['source_Dev_case'] for r in read(REGISTRY)['cases']}
    devrows['restored_cycle_IR_reference'] = dev['best_by_cue']['ir']['id']
    dev47 = {}
    for role, name in devrows.items():
        folder = work / 'structural_relative_dev' / name
        path, csv = folder / 'metrics.json', folder / 'trackeval_output/pedestrian_detailed.csv'
        value = read(path)
        if value['status'] != 'COMPLETE' or value['manifest']['frames'] != 4834:
            raise ValueError('An original Dev selection source is incomplete')
        bind(path)
        bind(csv)
        for old in read(REGISTRY)['cases']:
            if old['source_Dev_case'] == name:
                bind(path, old['source_Dev_metrics_sha256'])
        dev47[role] = dict(source_Dev_case=name, **official_csv47(csv, schema, 'dev', value['trackeval']))
    anchorh = hota('restored_cycle_IR_reference')
    comparisons = {role: anchorh - hota(role) for role in roles if role != 'restored_cycle_IR_reference'}
    dev_comparisons = {role: dev47['restored_cycle_IR_reference']['combined47']['HOTA'] -
                       row['combined47']['HOTA'] for role, row in dev47.items()
                       if role != 'restored_cycle_IR_reference'}
    if any(sha256(path) != digest for path, digest in files.items()):
        raise ValueError('Original evidence changed during independent audit')
    value = dict(status='COMPLETE_INDEPENDENT_LEGACY_ALL9_ALL7_TEST_OFFICIAL47_AUDIT',
        source_auditor_sha256=sha256(__file__), source_file_hashes=files, rows=rows, roles=roles,
        complete_cases=9, frames_each=15985, sequences=list(lengths), official_fields=47,
        official_CSV_columns=276, official_schema=schema, official_family_field_counts=counts,
        original_Dev_rows=dev47, restored_cycle_IR_minus_role_Test_HOTA=comparisons,
        restored_cycle_IR_minus_role_Dev_HOTA=dev_comparisons,
        unavailable_all47_CSV_MOT_exact_to_fresh_RGB=True,
        restored_cycle_IR_all39_CSV_MOT_exact_to_original_parent=True,
        original_controller_and_all_recorded_children_exited=True,
        official_worker_exit_codes={k: j['exit_code'] for k, j in state['jobs'].items()},
        producer_exit_evidence='Original registered controller asserted poll()==0; numeric exits not separately persisted.',
        original_Train_scope='Initial192cachedframes/video,1152Trainframes;not the later27251frame complete-Train study.',
        new_GPU_work=False, new_fit=False, fixed_seed=0, additional_seed_work=False,
        current_primary_unchanged=True, Test_primary_reselection=False,
        individual_method_NN_cost_measured_here=False, historical_Test_exposed=True,
        source_RGB_pretraining_saw_Dev=True, auxiliary_is_RGB_derived=True,
        goal_met=False, supported_innovations=0, seconds=time.monotonic()-started,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        scope='All9previously frozen original59Dev choices,including negative/control cases. '
              'Exact original276column CSVs,all47per-sequence/combined fields,full15985frame MOT/proposal/checkpoint/source '
              'provenance. Initial hard-cycle family,not the later full-Train S/C/R family. No Test reselection, '
              'new architecture,novelty,causal-estimand,NNcost or completed-goal claim.')
    if value['peak_process_RSS_GiB'] > 8:
        raise RuntimeError('CPU audit exceeded8GiB')
    output = args.output_directory
    output.mkdir(exist_ok=False)
    (output / 'official_CSVs').mkdir()
    for name, csv in csv_sources.items():
        (output / 'official_CSVs' / (name + '.csv')).write_bytes(csv.read_bytes())
    (output / 'complete.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status=value['status'], cases=9, seconds=value['seconds'],
                         peak_process_RSS_GiB=value['peak_process_RSS_GiB'],
                         restored_IR_HOTA=anchorh, comparisons=comparisons)))


if __name__ == '__main__':
    main()
