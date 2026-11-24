#!/usr/bin/env python3
"""Assess all26 completed frozen native36 cases without choosing a Test winner."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    source = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_COMMON_TEST_BACKFILL.json'
    data = json.loads(source.read_text())
    assert data['status'] == 'COMPLETE_FROZEN_NATIVE36_ALL26_TEST_BACKFILL'
    rows = {row['id']: row for row in data['rows']}
    assert len(rows) == 26 and len(data['Dev_reproduction_proofs']) == 8
    assert all(p['all39metrics_tracks_official_CSV_exact'] for p in data['Dev_reproduction_proofs'].values())
    sequences = ['BT-002', 'BT-004', 'MSK-003', 'PF-002', 'SN-009', 'SN-011', 'SN-014']
    assert all(r['manifest']['frames'] == 15985 and r['manifest']['sequences'] == sequences
               and set(r['per_sequence']) == set(sequences) and len(r['metrics']) == 39 for r in rows.values())
    for row in rows.values():
        assert sha256(ROOT / row['official_detailed_csv_file']) == row['official_detailed_csv_sha256']
    baseline = rows['NC36_rgb_ByteTrack']['metrics']
    assert baseline['HOTA'] == 29.529
    output = []
    for backend in ['consensus', 'robust']:
        for frontend in ['new', 'rgb']:
            prefix = 'NC36_'+backend+'_'+frontend
            normal = rows[prefix+'_ir_normal']
            matched = rows[prefix+'_rgb_normal']
            off = rows[prefix+'_ir_unavailable']
            wrong = rows[prefix+'_ir_mismatched']
            proofs = {}
            for name, intervention in [('native_increment_off', off), ('current_candidate_encoding_shuffle', wrong)]:
                proofs[name] = dict(all39_summary_exact=normal['metrics'] == intervention['metrics'],
                    official_CSV_exact=normal['official_detailed_csv_sha256'] == intervention['official_detailed_csv_sha256'],
                    all_tracks_exact=all(normal['manifest']['coverage'][s]['output_sha256'] ==
                        intervention['manifest']['coverage'][s]['output_sha256'] for s in sequences),
                    differing_summary_fields={k: [normal['metrics'][k], intervention['metrics'][k]]
                        for k in normal['metrics'] if normal['metrics'][k] != intervention['metrics'][k]})
            deltas = dict(IR_minus_fixed_RGB_HOTA=normal['metrics']['HOTA']-baseline['HOTA'],
                IR_minus_same_backend_RGB_HOTA=normal['metrics']['HOTA']-matched['metrics']['HOTA'],
                IR_minus_native_increment_off_HOTA=normal['metrics']['HOTA']-off['metrics']['HOTA'],
                IR_minus_candidate_encoding_shuffle_HOTA=normal['metrics']['HOTA']-wrong['metrics']['HOTA'])
            output.append(dict(backend=backend, frontend=frontend, IR=normal['metrics'],
                same_backend_RGB=matched['metrics'], native_increment_off=off['metrics'],
                current_candidate_encoding_shuffle=wrong['metrics'], deltas=deltas,
                exactness=proofs, per_sequence_deltas={s: dict(
                    IR_minus_matched_RGB_HOTA=normal['per_sequence'][s]['HOTA']-matched['per_sequence'][s]['HOTA'],
                    IR_minus_native_increment_off_HOTA=normal['per_sequence'][s]['HOTA']-off['per_sequence'][s]['HOTA'],
                    IR_minus_encoding_shuffle_HOTA=normal['per_sequence'][s]['HOTA']-wrong['per_sequence'][s]['HOTA']) for s in sequences}))
    stem = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_COMMON_TEST_ASSESSMENT'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    report = dict(status='COMPLETE_SCIENTIFIC_ASSESSMENT_OF_FROZEN_NATIVE36_ALL26_TEST',
        source_report_sha256=sha256(source), source_tool_sha256=sha256(__file__), rows=output,
        fixed_baseline=baseline, fixed_target_HOTA=34.529,
        native_IR_contribution_supported=False, supported_innovations=0, goal_met=False,
        current_primary_unchanged=True, Test_based_promotion=False,
        scope='All8 original Dev-selected normal models and16 frozen interventions plus2 source baselines, all7Test/15985frames each; seed0, no extra training/GPU.',
        cautions=[
            'Off disables only the learned native increment, preserving the learned RGB real/null prior, backend lifecycle and the chosen detector frontend. It does not disable every possible IR path in the new frontend.',
            'Mismatched intervention shuffles current candidate encodings; it is not a roll of actual IR image pixels.',
            'Equal rounded HOTA does not mean equal all39metrics, full detailed CSV or emitted tracks; exact flags are retained.',
            'Observed package-level improvement does not identify an individual causal component. No current native-increment benefit survives these interventions.',
            'Previously exposed historical Test and RGB pretraining on current Dev limit generalization claims. Cached replay FPS excludes encoding and is not end-to-end FPS.',
            'Remaining50 original native behaviors and other formal architectures are outside this26-case subset; their coverage is not inferred here.'
        ])
    dump(stem.with_suffix('.json'), report)
    lines = ['# Native36完整Test：收益与模态归因核查', '',
        '26项冻结条件均完成7序列/15985帧；八个原Dev选择的摘要、轨迹和官方CSV均已先精确复现。固定直接RGB为29.529，目标34.529。本轮不按Test更换主方法。', '',
        '| 后端 | 前端 | IR HOTA | 同后端RGB | native增量关闭 | 候选编码置换 | IR−直接RGB | IR−同后端RGB |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in output:
        lines.append(f"| {row['backend']} | {row['frontend']} | {row['IR']['HOTA']:.3f} | {row['same_backend_RGB']['HOTA']:.3f} | "
            f"{row['native_increment_off']['HOTA']:.3f} | {row['current_candidate_encoding_shuffle']['HOTA']:.3f} | "
            f"{row['deltas']['IR_minus_fixed_RGB_HOTA']:+.3f} | {row['deltas']['IR_minus_same_backend_RGB_HOTA']:+.3f} |")
    lines += ['',
        'new前端的consensus IR整体相对直接RGB高1.913点，但同后端RGB为31.443；关闭native增量或置换当前候选编码，HOTA仍为31.442。这里仅HOTA摘要到显示精度相等：normal与off还差1个CLR_TP、1个Dets等，完整CSV和轨迹也不相同，不能称为逐位无影响。rgb前端的consensus IR与这两个干预则同时满足39摘要、完整CSV、轨迹完全相等。', '',
        'new前端的robust IR比同后端RGB高0.091点，但比native增量关闭低0.332点、比置换编码低0.291点；rgb前端IR也未通过匹配RGB与干预比较。因此当前证据不支持该native增量的独立正贡献，已验证创新数仍为0。', '',
        '31.442的consensus组合IDF1为33.756，相对直接RGB33.072高0.684；IDSW为582，相对450增加132。整体HOTA提高伴随身份切换代价，且没有达到+5点。不能把前端、学习到的RGB先验、候选规则和后端的组合差值当成IR模块的单独贡献。', '',
        '关闭的是native增量，原学习到的RGB real/null先验、后端生命周期和所选前端均保留；尤其new前端仍是原新伪红外前端，不能称为完全关闭IR。候选编码置换也不同于新原型登记的实际IR图像roll。', '',
        '本批次复用原有seed0缓存，没有新增编码/GPU或种子。全部39摘要、逐序列数值、276列官方表及来源校验见父报告与表目录；缓存回放速度不能作为端到端速度。另50种原正式行为已自动进入Dev来源复现，尚不能据本批次宣布所有架构或候选补齐。历史Test已暴露、原RGB预训练见过当前Dev等限制继续适用。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'], supported_innovations=0, goal_met=False)), flush=True)


if __name__ == '__main__': main()
