#!/usr/bin/env python3
"""Assess all58 frozen native36 behaviors only after the final50 cases finish."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.native_remaining_test_backfill import verify, finished, signature
from tools.native_common_test_backfill import context, read
from tools.run_pseudo_ir_comparison import dump, sha256

DOCS = ROOT / 'docs/fusion_research'


def paired_signature(config):
    current = json.loads(signature(config))
    cue = current.pop('native_cue')
    assert cue in {'rgb', 'ir'} and current['native_seed'] == 0
    path = current['native_query_checkpoint']
    normalized = re.sub(r'_(rgb|ir)_s0/epoch[0-9]+\.pt$',
                        r'_{cue}_s0/epoch{Dev_selected}.pt', path)
    assert normalized != path
    current['native_query_checkpoint'] = normalized
    return json.dumps(current, sort_keys=True, separators=(',', ':')), cue


def off_signature(config):
    current = dict(config)
    current.pop('disable_native_increment', None)
    return signature(current, curve=True)


def main():
    plan, work, protocol = context()
    state_path = work / 'native_remaining_test_backfill_state.json'; state = read(state_path)
    assert state['status'] == 'COMPLETE_ALL50_REMAINING_NATIVE_TEST'
    assert finished(state['controller_pid']), 'Wait for final export and actual controller exit'
    for key in ['dev_jobs', 'test_jobs']:
        assert len(state[key]) == 50
        assert all(j['status'] == 'COMPLETE' and j.get('exit_code') in {None, 0}
                   and finished(j['pid']) for j in state[key].values())
    assert all(j['source_proof']['all39metrics_tracks_official_CSV_exact'] for j in state['dev_jobs'].values())
    specification = verify()
    report_path = DOCS / 'PERCEPTUAL_MFT_20261124_NATIVE_REMAINING_TEST_BACKFILL.json'
    report = read(report_path)
    assert report['status'] == 'COMPLETE_ORIGINAL58_NATIVE_BEHAVIORS_SEED0_ALL7_TEST'
    parent_path = DOCS / 'PERCEPTUAL_MFT_20261124_NATIVE_COMMON_TEST_BACKFILL.json'
    parent = read(parent_path)
    rows = {r['id']: r for r in parent['rows'] + report['rows']}
    parent_ids = {r['id'] for r in parent['rows']}
    grouped, by_group = {}, {}
    for original in specification['groups']:
        group = original['group_id']
        covered = report['original58_behavior_results'][group]
        row = rows[covered['case']]
        assert signature(row['config']) == signature(original['selected']['config'])
        assert row['metrics'] == covered['metrics']
        stage = 'native_common_test_backfill_test' if row['id'] in parent_ids else 'native_remaining_test_backfill_test'
        folder = work / stage / row['id']; value = read(folder / 'metrics.json')
        assert value['status'] == 'COMPLETE' and value['manifest']['frames'] == 15985
        assert value['manifest']['sequences'] == protocol['historical_test']
        assert signature(value['manifest']['config']) == signature(row['config'])
        assert value['trackeval'] == row['metrics'] and len(row['metrics']) == 39
        csv_path = ROOT / row['official_detailed_csv_file']
        assert sha256(csv_path) == row['official_detailed_csv_sha256']
        assert sha256(csv_path) == sha256(folder / 'trackeval_output/pedestrian_detailed.csv')
        literal = {seq: sha256(folder / 'track_results' / (seq+'.txt')) for seq in protocol['historical_test']}
        assert all(literal[seq] == value['manifest']['coverage'][seq]['output_sha256'] for seq in literal)
        record = dict(group=group, case=row['id'], config=row['config'],
            selected_epoch=original['selected']['epoch'], original_Dev_HOTA=original['selected']['HOTA'],
            metrics=row['metrics'], CSV_sha256=sha256(csv_path), literal_tracks_sha256=literal,
            official_CSV=str(csv_path.relative_to(ROOT)))
        by_group[group] = record
        key, cue = paired_signature(row['config'])
        assert cue not in grouped.setdefault(key, {})
        grouped[key][cue] = record
    assert len(by_group) == 58
    off_controls = {}
    for row in rows.values():
        if not row['config'].get('disable_native_increment'):
            continue
        stage = 'native_common_test_backfill_test' if row['id'] in parent_ids else 'native_remaining_test_backfill_test'
        folder = work / stage / row['id']; value = read(folder / 'metrics.json')
        assert value['status'] == 'COMPLETE' and value['manifest']['frames'] == 15985
        assert value['trackeval'] == row['metrics']
        csv = folder / 'trackeval_output/pedestrian_detailed.csv'
        assert sha256(csv) == row['official_detailed_csv_sha256']
        tracks = {seq: sha256(folder / 'track_results' / (seq+'.txt')) for seq in protocol['historical_test']}
        assert all(tracks[seq] == value['manifest']['coverage'][seq]['output_sha256'] for seq in tracks)
        key = off_signature(row['config'])
        record = dict(case=row['id'], metrics=row['metrics'], CSV_sha256=sha256(csv), tracks=tracks)
        if key in off_controls:
            previous = off_controls[key][0]
            assert previous['metrics'] == record['metrics'] and previous['CSV_sha256'] == record['CSV_sha256']
            assert previous['tracks'] == record['tracks']
        off_controls.setdefault(key, []).append(record)
    pairs, unmatched = [], []
    for key, members in sorted(grouped.items()):
        if set(members) != {'rgb', 'ir'}:
            unmatched.extend(r['group'] for r in members.values()); continue
        ir, rgb = members['ir'], members['rgb']
        delta = ir['metrics']['HOTA'] - rgb['metrics']['HOTA']
        references = off_controls.get(off_signature(ir['config']))
        own_off = None if not references else dict(cases=[r['case'] for r in references],
            Test_HOTA=references[0]['metrics']['HOTA'],
            IR_minus_off_HOTA_pp=ir['metrics']['HOTA']-references[0]['metrics']['HOTA'],
            IR_all39_CSV_tracks_exact_to_off=ir['metrics'] == references[0]['metrics']
                and ir['CSV_sha256'] == references[0]['CSV_sha256'] and ir['literal_tracks_sha256'] == references[0]['tracks'])
        pairs.append(dict(IR_group=ir['group'], RGB_group=rgb['group'],
            backend=ir['config']['backend'], frontend=ir['config']['detection_arm'],
            IR_Dev_selected_epoch=ir['selected_epoch'], RGB_Dev_selected_epoch=rgb['selected_epoch'],
            IR_Dev_HOTA=ir['original_Dev_HOTA'], RGB_Dev_HOTA=rgb['original_Dev_HOTA'],
            IR_Test_HOTA=ir['metrics']['HOTA'], RGB_Test_HOTA=rgb['metrics']['HOTA'],
            IR_minus_RGB_HOTA_pp=delta,
            IR_minus_RGB_IDF1_pp=ir['metrics']['IDF1']-rgb['metrics']['IDF1'],
            all39_metrics_exact=ir['metrics'] == rgb['metrics'],
            entire_official_CSV_exact=ir['CSV_sha256'] == rgb['CSV_sha256'],
            literal_tracks_exact=ir['literal_tracks_sha256'] == rgb['literal_tracks_sha256'],
            own_disabled_increment_reference=own_off,
            route=dict(backend=ir['config']['backend'],
                query_ranking=ir['config'].get('query_ranking', True),
                query_null=ir['config'].get('query_null', True),
                disable_native_increment=ir['config'].get('disable_native_increment', False),
                shuffle_native_cue=ir['config'].get('shuffle_native_cue', False)),
            source_pair_config=json.loads(key)))
    assert len(pairs) == 23 and len(unmatched) == 12
    statistics = dict(pairs=23, unmatched_IR_only_configurations=12,
        positive_HOTA_pairs=sum(p['IR_minus_RGB_HOTA_pp'] > 0 for p in pairs),
        negative_HOTA_pairs=sum(p['IR_minus_RGB_HOTA_pp'] < 0 for p in pairs),
        equal_rounded_HOTA_pairs=sum(p['IR_minus_RGB_HOTA_pp'] == 0 for p in pairs),
        exact39_CSV_and_tracks_pairs=sum(p['all39_metrics_exact'] and p['entire_official_CSV_exact']
                                        and p['literal_tracks_exact'] for p in pairs))
    statistics['IR_pairs_with_strict_own_disabled_reference'] = sum(p['own_disabled_increment_reference'] is not None for p in pairs)
    observed = [row['metrics']['HOTA'] for row in by_group.values()]
    largest_pair = max(pairs, key=lambda p: p['IR_minus_RGB_HOTA_pp'])
    observed_range = dict(minimum=min(observed), maximum=max(observed),
        maximum_minus_original_RGB_pp=max(observed)-29.529, at_or_above_goal34_529=sum(v >= 34.529 for v in observed),
        scope='Descriptive range of all58already-frozen Test configurations; not a new primary selection.')
    result = dict(status='COMPLETE_FROZEN_NATIVE58_SCOPED_ALL_TEST_ASSESSMENT',
        original_behavior_groups=58, new_full_Test_cases=50, parent_full_Test_cases=26,
        frames_per_Test=15985, matched_pair_statistics=statistics, matched_pairs=pairs,
        observed_Test_HOTA_range=observed_range,
        largest_observed_paired_delta=dict(IR_group=largest_pair['IR_group'], RGB_group=largest_pair['RGB_group'],
            IR_Test_HOTA=largest_pair['IR_Test_HOTA'], RGB_Test_HOTA=largest_pair['RGB_Test_HOTA'],
            delta_pp=largest_pair['IR_minus_RGB_HOTA_pp'], IR_minus_original_RGB_pp=largest_pair['IR_Test_HOTA']-29.529),
        unmatched_original_groups=unmatched, original58_results=by_group,
        report_sha256=sha256(report_path), parent_report_sha256=sha256(parent_path),
        source_sha256=sha256(__file__), fixed_seed=0, additional_seed_work=False,
        direct_child_exit_codes_saved_in_original_state=all('exit_code' in j
            for key in ['dev_jobs', 'test_jobs'] for j in state[key].values()),
        process_verification_scope='Original controller checks zero child exit before marking COMPLETE '
            'but did not save numeric exit codes. Missing codes remain unknown; independently check '
            'all complete artifacts, source versions, officialCSV, frame coverage, tracks and exited processes.',
        current_primary_unchanged=True, Test_based_selection=False, new_evaluation=False,
        interpretation='All58 pre-existing behaviors now have complete7Test coverage. '
            '23 cue pairs match every inference field except cue and independently Dev-selected head snapshot; '
            'all12unmatched original behaviors retained. Positive pair differences are not automatically '
            '2-3independent innovations, four architectures, or a>=5point result. No Test winner is promoted.',
        metric_scope='HOTA/IDF1 differences here use the original39 headline rounded summaries. '
            'Full precision and every threshold/sequence remain in the unchanged276-columnCSV/47fieldviews. '
            'Equal HOTA alone is not exact track equivalence.',
        runtime_scope='Cached CPU replay only; full detector/encoder/point/IR-generation costs excluded.',
        supported_innovations=0, goal_met=False, full_goal_coverage=False)
    stem = DOCS / 'PERCEPTUAL_MFT_20261124_NATIVE_REMAINING_ASSESSMENT'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    dump(stem.with_suffix('.json'), result)
    lines = ['# 原58种native行为的完整Test核对', '',
        f'全部58个冻结对象的HOTA范围为{min(observed):.3f}–{max(observed):.3f}，'
        f'观测最大值相对原RGB29.529为{max(observed)-29.529:+.3f}点，尚无对象达到34.529。'
        '这只是既有结果范围，未据Test改选主方法。', '',
        f'23组严格匹配的RGB/IR对照中，14组IR较高、9组较低。最大配对差值'
        f'{largest_pair["IR_minus_RGB_HOTA_pp"]:+.3f}对应IR{largest_pair["IR_Test_HOTA"]:.3f}'
        f'与RGB{largest_pair["RGB_Test_HOTA"]:.3f}；该IR相对原始29.529仅'
        f'{largest_pair["IR_Test_HOTA"]-29.529:+.3f}点，不能把较弱对照造成的差值当成+5或多项创新。', '',
        result['interpretation'], '',
        '| 原IR组 | 原RGB组 | IR Dev轮次 | RGB Dev轮次 | IR Test HOTA | RGB Test HOTA | IR−RGB | IR−关闭增量 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for pair in pairs:
        off = pair['own_disabled_increment_reference']
        off_text = '未严格配对' if off is None else f'{off["IR_minus_off_HOTA_pp"]:+.3f}'
        lines.append(f'| {pair["IR_group"]} | {pair["RGB_group"]} | {pair["IR_Dev_selected_epoch"]} | '
            f'{pair["RGB_Dev_selected_epoch"]} | {pair["IR_Test_HOTA"]:.3f} | {pair["RGB_Test_HOTA"]:.3f} | '
            f'{pair["IR_minus_RGB_HOTA_pp"]:+.3f} | {off_text} |')
    lines += ['', result['metric_scope'], '',
        '这58种是同一家族不同路线/记忆/读出配置，不是58种融合架构。'
        '所有旧配置由完整Dev选快照，Test只统一评估，固定主方法未改。'
        '12种IR专属去除/干预没有强行配成RGB对照，原结果仍完整保留。'
        '相同固定前端下比较新增向量，关闭增量不等于关闭new前端全部IR路径。', '',
        result['runtime_scope'], '',
        '旧状态未保存数值退出码；这些值保留未知。核验依据已完成产物、原始来源、逐帧覆盖、'
        '实际进程退出及控制器完成前的零退出检查，没有补造exit_code=0。', '',
        '历史Test已暴露、RGB预训练见过Dev、伪IR从RGB派生的限制保留。Goal未完成。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(result['status'], statistics, flush=True)


if __name__ == '__main__':
    main()
