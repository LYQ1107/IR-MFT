#!/usr/bin/env python3
"""Reconcile all23 frozen cue pairs with original full-precision official CSV.

No inference, fitting, Test selection or new primary. Missing controls stay
unknown. This is a necessary-evidence audit of a completed behavior family.
"""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.assess_native_remaining_test import off_signature
from tools.native_remaining_test_backfill import signature
from tools.run_pseudo_ir_comparison import dump, sha256

DOCS = ROOT/'docs/fusion_research'
ASSESSMENT = DOCS/'PERCEPTUAL_MFT_20261124_NATIVE_REMAINING_ASSESSMENT.json'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    plan = read(ROOT/'configs/perceptual_mft_20261122.json')
    protocol = read(ROOT/plan['protocol']); work = Path(plan['work_root'])
    assessment = read(ASSESSMENT)
    assert assessment['status'] == 'COMPLETE_FROZEN_NATIVE58_SCOPED_ALL_TEST_ASSESSMENT'
    assert len(assessment['matched_pairs']) == 23
    cache = {}

    def load_case(case, split, optional=False):
        key = case, split
        if key in cache: return cache[key]
        paths = [work/(stage+'_'+split)/case/'metrics.json' for stage in [
            'native_remaining_test_backfill', 'native_common_test_backfill']]
        paths = [p for p in paths if p.exists()]
        if not paths and optional: return None
        assert len(paths) == 1, (case, split, paths)
        path = paths[0]; value = read(path); manifest = value['manifest']
        expected = protocol['dev' if split == 'dev' else 'historical_test']
        assert value['status'] == 'COMPLETE' and len(value['trackeval']) == 39
        assert manifest['split'] == split and manifest['sequences'] == expected
        assert manifest['frames'] == {'dev':4834,'test':15985}[split]
        assert set(manifest['coverage']) == set(expected)
        assert sum(row['frames'] for row in manifest['coverage'].values()) == manifest['frames']
        assert manifest['config']['native_seed'] == 0
        table = path.parent/'trackeval_output/pedestrian_detailed.csv'
        with table.open() as stream:
            reader = csv.DictReader(stream); rows = list(reader)
            assert len(reader.fieldnames) == 276
        assert [r['seq'] for r in rows] == expected+['COMBINED']
        combined = rows[-1]; hota = 100*float(combined['HOTA___AUC'])
        assert abs(hota-value['trackeval']['HOTA']) <= .000501
        result = dict(HOTA_pp=hota, summary_HOTA=value['trackeval']['HOTA'],
            metrics_json=str(path), metrics_json_sha256=sha256(path),
            official_CSV=str(table), official_CSV_sha256=sha256(table),
            frames=manifest['frames'], sequences=expected,
            config=manifest['config'])
        cache[key] = result
        return result

    rows = []
    for pair in assessment['matched_pairs']:
        ir, rgb = [assessment['original58_results'][pair[k]] for k in ['IR_group','RGB_group']]
        values = {}
        for cue, record in [('IR',ir),('RGB',rgb)]:
            dev, test = [load_case(record['case'], split) for split in ['dev','test']]
            assert signature(dev['config']) == signature(test['config']) == signature(record['config'])
            assert abs(dev['HOTA_pp']-record['original_Dev_HOTA']) <= .000501
            assert test['official_CSV_sha256'] == record['CSV_sha256'] == sha256(ROOT/record['official_CSV'])
            values[cue] = dict(Dev=dev,Test=test)
        dev_delta = values['IR']['Dev']['HOTA_pp']-values['RGB']['Dev']['HOTA_pp']
        test_delta = values['IR']['Test']['HOTA_pp']-values['RGB']['Test']['HOTA_pp']
        reference = pair['own_disabled_increment_reference']; off = None
        if reference:
            off_test = load_case(reference['cases'][0], 'test')
            off_dev = load_case(reference['cases'][0], 'dev', optional=True)
            assert off_signature(off_test['config']) == off_signature(ir['config'])
            assert off_test['config']['disable_native_increment']
            if off_dev:
                assert signature(off_dev['config']) == signature(off_test['config'])
            off = dict(cases=reference['cases'],Test=off_test,Dev=off_dev,
                IR_minus_off_Dev_pp=None if off_dev is None else values['IR']['Dev']['HOTA_pp']-off_dev['HOTA_pp'],
                IR_minus_off_Test_pp=values['IR']['Test']['HOTA_pp']-off_test['HOTA_pp'])
        both = dev_delta > 1e-10 and test_delta > 1e-10
        off_both = (off is not None and off['IR_minus_off_Dev_pp'] is not None and
                    off['IR_minus_off_Dev_pp'] > 1e-10 and off['IR_minus_off_Test_pp'] > 1e-10)
        rows.append(dict(IR_group=pair['IR_group'],RGB_group=pair['RGB_group'],
            frontend=pair['frontend'],backend=pair['backend'],values=values,
            IR_minus_RGB_Dev_pp=dev_delta,IR_minus_RGB_Test_pp=test_delta,
            positive_against_RGB_on_both_splits=both,own_disabled_increment=off,
            positive_against_own_off_on_both_splits=off_both,
            all_four_necessary_deltas_positive=both and off_both))
    for row in cache.values(): row.pop('config')
    counts = dict(pairs=len(rows),positive_IR_vs_RGB_Dev=sum(r['IR_minus_RGB_Dev_pp']>1e-10 for r in rows),
        positive_IR_vs_RGB_Test=sum(r['IR_minus_RGB_Test_pp']>1e-10 for r in rows),
        positive_IR_vs_RGB_both_splits=sum(r['positive_against_RGB_on_both_splits'] for r in rows),
        strict_own_off_Test_available=sum(r['own_disabled_increment'] is not None for r in rows),
        strict_own_off_Dev_and_Test_available=sum(r['own_disabled_increment'] is not None and r['own_disabled_increment']['Dev'] is not None for r in rows),
        all_four_necessary_deltas_positive=sum(r['all_four_necessary_deltas_positive'] for r in rows))
    stem = DOCS/'PERCEPTUAL_MFT_20261124_NATIVE_DEV_TEST_CONCORDANCE'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    result = dict(status='COMPLETE_FULL_PRECISION_NATIVE23_DEV_TEST_CONCORDANCE',counts=counts,rows=rows,
        assessment_sha256=sha256(ASSESSMENT),source_sha256=sha256(Path(__file__)),
        metric_scope='Original276-column officialCSV COMBINED HOTA___AUC times100; full precision absolute percentage points. Rounded39headline values retained separately. Complete2Dev4834 and7Test15985 frames only.',
        interpretation='All previously frozen cue pairs retained. Test-positive alone is insufficient; Dev-positive and strict own-increment-off comparisons are required. Unknown off controls remain unknown. Even all four positive deltas would not prove2independent contributions or originality.',
        control_scope='RGB frontend: own-native-increment off removes this IR query increment. New dual-stream frontend: it does not remove every IR path in the detector. Same cue pairs differ only in cue and their independently Dev-selected head checkpoint.',
        selection_policy='No Test winner promoted, no primary changed, no new selection or fitting. This completed-family audit cannot authorize a new method from Test.',
        fixed_seed=0,additional_seed_work=False,GPU_work=False,new_inference=False,
        new_accuracy_evaluation=False,full_goal_coverage=False,goal_met=False,supported_innovations=0)
    dump(stem.with_suffix('.json'),result)
    lines=['# 原生身份分支：Dev/Test与自身关闭对照的一致性', '',
        '使用原始276列官方CSV的COMBINED HOTA___AUC×100，保留完整精度；没有重新推理或评估。', '',
        f"23组配对中，Dev胜过RGB有{counts['positive_IR_vs_RGB_Dev']}组，Test有{counts['positive_IR_vs_RGB_Test']}组，"
        f"两者同时胜过仅{counts['positive_IR_vs_RGB_both_splits']}组。再要求同时胜过自身增量关闭对照，满足四项必要增益的有{counts['all_four_necessary_deltas_positive']}组。", '',
        '这些是必要数值检查，不是创新性判定。缺失Dev关闭对照保留未知；不能用弱RGB对照替代自身模块贡献。', '',
        '| IR配置 | Dev相对RGB | Test相对RGB | Dev相对自身关闭 | Test相对自身关闭 |',
        '| --- | ---: | ---: | ---: | ---: |']
    def fmt(v): return '未知' if v is None else f'{v:+.6f}'
    for row in rows:
        off=row['own_disabled_increment']
        lines.append(f"| {row['IR_group']} | {fmt(row['IR_minus_RGB_Dev_pp'])} | {fmt(row['IR_minus_RGB_Test_pp'])} | "
            f"{fmt(None if off is None else off['IR_minus_off_Dev_pp'])} | {fmt(None if off is None else off['IR_minus_off_Test_pp'])} |")
    lines += ['', '所有数值都是HOTA绝对百分点。未据Test选择新主方法；历史Test暴露、RGB预训练见过Dev的限制继续适用。', '',
        '关闭new前端的native query增量不等于关闭检测器全部IR路径。本表不代表四种独立融合架构、2–3项独立创新或达到+5目标。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(result['status'],json.dumps(counts),flush=True)


if __name__ == '__main__': main()
