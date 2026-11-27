#!/usr/bin/env python3
"""Audit completed bridge tables/retained main rows and export small evidence."""
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_mft_trusted_shadow_bridge import registered, context
from tools.audit_mft_protected_peripheral import audit_projection
from tools.audit_mft_coverage_recall import parse_official, audit_validation_views
from tools.mft_protected_peripheral_experiment import cache
from tools.native_pyramid_experiment import read, frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.fit_mft_proposal_quality import utc
from tools.fusion_memory_observation import LENGTHS
from tools.export_mft_coverage_recall import compact


def main():
    r = registered(); plan, protocol, work, parent, base = context()
    e = read(base/'run_r1.execution.json'); state = read(base/'state.json')
    assert e['actual_child_exit_code'] == 0 and e['source_unchanged'] and identity(e['identity']['PID']) is None
    jobs = [j for phase in state['phases'].values() for j in phase.values()]
    assert all(j['status'] == 'COMPLETE' and type(j['exit_code']) is int and j['exit_code'] == 0
               and j['argv_identity_observed'] and identity(j['pid']) is None for j in jobs)
    views = audit_validation_views(r)
    fields = tables = files = main_rows = all_rows = cleanup_bytes = 0
    for split in ['trainval', 'dev', 'test']:
        if not (base/'streams'/split).exists():
            continue
        logical = 'train' if split == 'trainval' else split
        lengths = r['validation_lengths'] if split == 'trainval' else LENGTHS[split]
        for sequence in lengths:
            m = read(base/'streams'/split/'sequence_manifests'/(sequence+'.json'))
            meta, a = cache(logical, sequence)
            assert m['source_versions'] == r['source_hashes'] and m['source_frame_bank_sha256'] == meta['sha256']
            assert not m['GT_labels_read'] and not m['future_frames_supplied'] and not m['past_outputs_rewritten']
            assert m['all_past_frames_consumed'] == LENGTHS[logical][sequence]
            for key, v in m['streams'].items():
                assert sha256(v['track_path']) == v['track_sha256']
                n, total = audit_projection(Path(v['track_path']), a, m['validation_window_cut'],
                                            v['main_aliases'], meta['scored_reference_MOT_sha256'])
                assert total == v['track_rows']; main_rows += n; all_rows += total; files += 1
        for path in sorted((base/'evaluation'/split).glob('*/metrics.json')):
            row = read(path); csv = path.parent/'trackeval_output/pedestrian_detailed.csv'
            raw = parse_official(csv, row['official_schema'], lengths)
            assert sha256(csv) == row['official_CSV_sha256']
            assert {k:raw[-1][k] for k in row['combined47']} == row['combined47']
            assert all(all(v[k] == original[k] for k in original) for v, original in zip(row['per_sequence47'], raw))
            fields += 47*len(raw); tables += 1
            summary = (path.parent/'trackeval_output/pedestrian_summary.txt').read_text().splitlines()
            keys, values = summary[0].split(), list(map(float, summary[1].split()))
            assert len(keys) == len(values) == 39
            assert all(abs(row['combined47'][k]-v) <= .000501 for k,v in zip(keys,values))
            for v in read(path.parent/'derived_duplicate_cleanup.json')['rows']:
                assert not Path(v['path']).exists() and sha256(v['retained_canonical']) == v['sha256']
                cleanup_bytes += v['bytes']
    result = read(base/'complete_Train_Dev.json')
    val, dev = result['validation_rows'], result['Dev_rows']
    reference = next(v for v in val if v['case']['bridge_gap'] == 0)
    dev_ref = next(v for v in dev if v['case']['bridge_gap'] == 0)
    assert reference['combined47'] == r['parent_validation47'] and dev_ref['combined47'] == r['parent_Dev47']
    gains = {v['case']['id']:v['combined47']['HOTA']-reference['combined47']['HOTA'] for v in val}
    assert gains == result['validation_HOTA_gains_over_PP_pp']
    eligible = [v for v in dev if v['case']['bridge_gap'] > 0 and v['case']['bridge_agreement'] == 3
                and gains[v['case']['id']] >= r['specification']['minimum_validation_HOTA_gain_over_selected_parent_pp']
                and v['combined47']['HOTA'] >= dev_ref['combined47']['HOTA']-r['specification']['full_Dev_noninferiority_tolerance_pp']]
    chosen = min(eligible, key=lambda v:(-gains[v['case']['id']], *(-v['combined47'][k] for k in ['HOTA','AssA','IDF1']), v['case']['id'])) if eligible else None
    assert result['eligible_ids'] == [v['case']['id'] for v in eligible]
    assert result['selected_case'] == (None if chosen is None else chosen['case'])
    test = None
    if chosen is None:
        assert not (base/'Test_selection_frozen.json').exists() and not (base/'full_Test_results.json').exists()
    else:
        freeze = read(base/'Test_selection_frozen.json'); test = read(base/'full_Test_results.json')
        assert freeze['chosen_case'] == result['selected_case'] == test['selected_case']
        assert freeze['complete_Train_Dev_sha256'] == sha256(base/'complete_Train_Dev.json')
        selected = next(v for v in test['rows'] if v['case'] == test['selected_case'])
        assert test['original_RGB_baseline']['official_CSV_sha256'] == 'c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
        assert test['HOTA_gain_over_original_RGB_pp'] == selected['combined47']['HOTA']-29.528970687393972
        assert all(j['started_UTC'] > freeze['UTC'] for phase, js in state['phases'].items()
                   if phase.startswith('SHADOW_TEST_') for j in js.values())
    audit = dict(status='COMPLETE_BRIDGE_CSV_MAIN_PROJECTION_SELECTION_AUDIT', UTC=utc(),
        actual_controller_exit0=True, actual_related_jobs_exit0=len(jobs), official_tables_reparsed=tables,
        official47_fields_checked=fields, original_main_MOT_projections_checked=files,
        main_rows_preserved=main_rows, total_rows_checked=all_rows, cleanup_bytes_verified=cleanup_bytes,
        validation_views_rebuilt=views, selection_rebuilt=True, own_source_sha256=sha256(Path(__file__)),
        scope='Independent final MOT retention, CSV47/39, cleanup and selection; not full independent bridge-state or visual production.',
        goal_met=False, supported_innovations=0)
    frozen_json(base/'independent_results_audit.json', audit)
    doc = ROOT/'docs/fusion_research'; directory = doc/'perceptual_20261127_trusted_shadow_bridge'
    assert not directory.exists(); directory.mkdir()
    names = ['registered.json','unit_regressions.json','run_r1.supervisor.py','run_r1.execution.json','run_r1.stdout.txt',
             'state.json','engineering.json','complete_Train_Dev.json','independent_results_audit.json']
    names.extend(n for n in ['Test_selection_frozen.json','full_Test_results.json'] if (base/n).exists())
    for folder in ['streams','evaluation','logs']:
        for p in sorted((base/folder).rglob('*')):
            if not p.is_file(): continue
            if folder == 'streams' and 'sequence_manifests' not in p.parts: continue
            if folder == 'evaluation' and p.name not in ['metrics.json','pedestrian_detailed.csv','pedestrian_summary.txt','derived_duplicate_cleanup.json']: continue
            names.append(str(p.relative_to(base)))
    assert len(names) == len(set(names))
    paths = []
    for name in names:
        source, target = base/name, directory/name
        assert source.stat().st_size < 2*2**20 and source.suffix not in ['.npz','.pt','.pth'] and 'track_results' not in source.parts
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source,target)
        assert sha256(source) == sha256(target); paths.append(str(target.relative_to(ROOT)))
    report = dict(status='PUBLISHED_COMPLETE_TRUSTED_SHADOW_STUDY', UTC=utc(), specification=r['specification'],
        validation_rows=[compact(v) for v in val], Dev_rows=[compact(v) for v in dev], validation_HOTA_gains_over_PP_pp=gains,
        selected_case=result['selected_case'], Test_rows=[] if test is None else [compact(v) for v in test['rows']],
        Test_gain_over_original_RGB_pp=None if test is None else test['HOTA_gain_over_original_RGB_pp'],
        baseline_HOTA=29.528970687393972, target_HOTA=34.52897068739397, stretch_HOTA=39.52897068739397,
        independent_audit=audit, prior_Test_exposed=True, same_native_source_cache=True, new_GPU_work=False,
        no_new_IR_specific_component=True, goal_met=False, supported_innovations=0)
    root = doc/'PERCEPTUAL_MFT_20261127_TRUSTED_SHADOW_BRIDGE.json'; assert not root.exists()
    root.write_text(json.dumps(report,indent=2)+'\n'); paths.append(str(root.relative_to(ROOT)))
    lines = ['# Short measured continuation of trusted identities','',
        'Fixed November DiffV2IR/ProtocolV2, seed0. Original RGB-only YOLOX-X + ByteTrack baseline stays **29.5289706874 HOTA**; minimum **34.5289706874**, stretch **39.5289706874**. StrongestRGB/crossarchitecture sweeps stay deferred.','',
        'Repeated agreement on exactly the same current proposal binds an independent low-threshold shadow to an already published trusted identity. If the trusted output is missing, a current measured shadow may continue that identity for5 or25 frames after the last agreement. It must remain unassigned and less than half spatially explained, with score>0.1 and area>100. Current disagreement revokes the old binding. No two trusted IDs merge; no synthetic box or earlier output rewrite is used. Every original trusted geometry/score and constant injective alias remains checked.','',
        'Four state/identity regressions and six real-Train prefix checks passed. All27,251 Train frames warm each publisher; only the last6,814 are scored. Full Dev covers4,834 frames. Only agreement3 cases with Train gain>=0.1 over selected PP and no full Dev degradation qualify. Agreement1 is a fixed control. All hypotheses/cases/selection rules were registered before their new metrics. Historical Test, including prior PP results, was already exposed; this is not a fresh blind test.','',
        '| Case | Forward Train HOTA | Gain over PP | Dev HOTA |','|---|---:|---:|---:|']
    bydev = {v['case']['id']:v for v in dev}
    for v in val:
        key = v['case']['id']; lines.append(f"| {key} | {v['combined47']['HOTA']:.9f} | {gains[key]:+.9f} | {bydev[key]['combined47']['HOTA']:.9f} |")
    if test is None:
        lines.extend(['','No case qualified. No new Test run/selection occurred. The prior formal PP result remains **32.4955880280 HOTA**, **+2.9666173406** over original RGB, below+5.'])
    else:
        selected = next(v for v in test['rows'] if v['case'] == test['selected_case'])
        lines.extend(['',f"Frozen selection **{test['selected_case']['id']}**: complete Test7 HOTA **{selected['combined47']['HOTA']:.9f}**, original-RGB gain **{test['HOTA_gain_over_original_RGB_pp']:+.9f}**. Numeric+5: **{test['target_plus5_met']}**, +10: **{test['target_plus10_met']}**."])
    lines.extend(['',f"Independent checks reparse{tables} original276-column CSVs and{fields} official47 values; check{files} main projections and{main_rows} retained trusted rows; reconstruct GT windows, prospective selection and duplicate-copy cleanup. All{len(jobs)} jobs and controller have actualexit0. Scope excludes a second visual producer and complete independent bridge-state replay.",'',
        'The bridge uses geometry/IDs and inherits the prior paired need gate. It provides no new independent IR signal. Positive continuation gains cannot be claimed as multimodal novelty; short-term object permanence/continuation/handoff have established precedents. Legacy encoder Train identities, prior development-window inspection and older detector Dev exposure remain disclosed. Pseudo-IR is generated from RGB, not measured thermal.','',
        'Only lightweight code/configuration/official47 aggregate and per-sequence values/CSV/summary/logs/provenance are published. Canonical MOT, descriptors, images, weights and GT stay private. Cached publisher costs exclude neural inference/image IO/IR generation and are not end-to-end FPS. The new study has a256MiB private cap inside the64GiB work cap with40GiB free reserve. No new GPU use or seed.','',
        '**Goal remains active; independently supported new multimodal innovations:0.**',''])
    md = doc/'PERCEPTUAL_MFT_20261127_TRUSTED_SHADOW_BRIDGE.md'; assert not md.exists()
    md.write_text('\n'.join(lines)); paths.append(str(md.relative_to(ROOT)))
    manifest = directory/'publication_manifest.json'
    manifest.write_text(json.dumps(dict(paths=paths,file_SHA256={p:sha256(ROOT/p) for p in paths},
        total_bytes=sum((ROOT/p).stat().st_size for p in paths),large_assets_included=False),indent=2)+'\n')
    print(json.dumps(dict(status='EXPORTED_COMPLETE_TRUSTED_SHADOW',paths=len(paths)+1,actual_jobs_exit0=len(jobs),official_tables=tables)))


if __name__ == '__main__':
    main()
