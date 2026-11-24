#!/usr/bin/env python3
"""Verify the seven old, Dev-frozen seed0 ordinary controls without rerunning them.

This audits named formal roles, not every exploratory backend configuration.
No fitting, image forward, TrackEval invocation or accuracy-based choice occurs.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.export_perceptual_metrics47 import official_schema, read_table
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.evidence_adapter import EvidenceAdapter

DOCS = ROOT / 'docs/fusion_research'
EXPECTED_TEST_LENGTHS = dict(zip(
    ['BT-002', 'BT-004', 'MSK-003', 'PF-002', 'SN-009', 'SN-011', 'SN-014'],
    [493, 763, 629, 3000, 4500, 4500, 2100]))


def read(path):
    return json.loads(Path(path).read_text())


class Audit:
    def __init__(self):
        self.artifact_hashes = {}

    def pin(self, path, expected=None):
        path = Path(path)
        key = str(path.resolve())
        if key not in self.artifact_hashes:
            self.artifact_hashes[key] = sha256(path)
        value = self.artifact_hashes[key]
        if expected is not None:
            assert value == expected, path
        return value

    def dev(self, path, protocol, expected=None):
        self.pin(path, expected)
        result = read(path)
        manifest = result['manifest']
        assert result['status'] == 'COMPLETE' and len(result['trackeval']) == 39
        assert manifest['split'] == 'dev' and manifest['sequences'] == protocol['dev']
        assert manifest['frames'] == 4834 and not manifest['GT_inference_input']
        assert not manifest['future_frame_input'] and not manifest['previous_outputs_rewritten']
        self.pin(manifest['config_path'], manifest['config_sha256'])
        assert read(manifest['config_path']) == manifest['config']
        return result

    def result(self, work, protocol, schema, family, name, cue, epoch, old_row):
        stage, key = (('metric_selected_test', 'evidence_adapter') if family == 'metric'
                      else ('pair_selected_test', 'association_checkpoint'))
        config_path = ROOT / ('configs/perceptual_metric_selected_20261122' if family == 'metric'
                              else 'configs/perceptual_pair_selected_20261123') / (name + '.json')
        config = read(config_path)
        assert config['id'] == name and (family != 'association_metric' or config['calibration_cue'] == cue)
        assert set(k for k in config if k.endswith('checkpoint') or k == 'evidence_adapter') == {key}
        checkpoint_path = Path(config[key])
        assert checkpoint_path.parent == work / family / (cue + '_s0')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['seed'] == 0 and checkpoint['cue'] == cue and checkpoint['epoch'] == epoch
        assert checkpoint['fit_sequences'] == protocol['train']
        model = (EvidenceAdapter if family == 'metric' else AssociationCalibrator)(**checkpoint['architecture'])
        model.load_state_dict(checkpoint['model'], strict=True)
        assert all(torch.isfinite(value).all() for value in model.state_dict().values())
        complete_path = checkpoint_path.parent / 'complete.json'
        complete = read(complete_path)
        assert complete['status'] == 'COMPLETE_FIT_STAGE' and complete['seed'] == 0 and complete['cue'] == cue
        assert complete['epochs'] == (24 if family == 'metric' else 36)
        assert [r['epoch'] for r in complete['history']] == list(range(1, complete['epochs'] + 1))
        fit_config = ROOT / ('configs/perceptual_metric_20261122.json' if family == 'metric'
                             else 'configs/perceptual_pair_calibration_20261123.json')
        self.pin(fit_config, checkpoint['config_sha256'])
        if family == 'metric':
            assert complete['Dev_fitting_rows'] == complete['Test_fitting_rows'] == 0
            assert complete['train_bank_sha256'] == checkpoint['train_bank_sha256']
            bank_meta = work / 'train_bank/complete.json'
            self.pin(bank_meta)
            bank = read(bank_meta)
            assert bank['status'] == 'COMPLETE_TRAIN_ONLY' and bank['sequences'] == protocol['train']
            assert bank['Dev_rows_used'] == bank['Test_rows_used'] == 0
            assert bank['feature_bank_sha256'] == checkpoint['train_bank_sha256']
            self.pin(work / 'train_bank/train.npz', checkpoint['train_bank_sha256'])
        else:
            assert complete['Dev_fit_rows'] == complete['Test_fit_rows'] == 0
            assert checkpoint['Dev_fit_rows'] == checkpoint['Test_fit_rows'] == 0
            assert checkpoint['sources'] == complete['sources'] and set(checkpoint['sources']) == set(protocol['train'])
            for sequence, digest in checkpoint['sources'].items():
                bank_meta = work / 'association_train' / sequence / 'complete.json'
                self.pin(bank_meta)
                bank = read(bank_meta)
                assert bank['status'] == 'COMPLETE_TRAIN_ONLY' and bank['sequence'] == sequence
                assert bank['Dev_rows'] == bank['Test_rows'] == 0
                assert bank['config_sha256'] == checkpoint['config_sha256'] and bank['bank_sha256'] == digest
                self.pin(bank_meta.parent / 'pairs.npz', digest)
        self.pin(complete_path)
        folder = work / stage / name
        metrics_path = folder / 'metrics.json'
        result = read(metrics_path)
        manifest = result['manifest']
        assert result['status'] == 'COMPLETE' and len(result['trackeval']) == 39
        assert manifest['config'] == config
        self.pin(config_path, manifest['config_sha256'])
        self.pin(metrics_path, old_row['metrics_sha256'])
        assert old_row['metrics'] == result['trackeval']
        assert manifest['plan_sha256'] == sha256(ROOT / 'configs/perceptual_mft_20261122.json')
        assert manifest['split'] == 'test' and manifest['sequences'] == protocol['historical_test']
        assert manifest['frames'] == 15985 and set(manifest['coverage']) == set(EXPECTED_TEST_LENGTHS)
        assert not any(manifest[k] for k in ['GT_inference_input', 'future_frame_input', 'previous_outputs_rewritten'])
        expected_hash = (manifest['evidence_adapter']['checkpoint_sha256'] if family == 'metric'
                         else manifest['association_checkpoint_sha256'])
        self.pin(checkpoint_path, expected_hash)
        lines = (folder / 'trackeval_output/pedestrian_summary.txt').read_text().splitlines()
        summary = dict(zip(lines[0].split(), map(float, lines[1].split())))
        assert len(summary) == 39 and summary == result['trackeval']
        self.pin(folder / 'trackeval_output/pedestrian_summary.txt')
        published = DOCS / 'perceptual_20261122_official_tables' / stage / (name + '.csv')
        self.pin(published, sha256(folder / 'trackeval_output/pedestrian_detailed.csv'))
        values, _ = read_table(published, schema, protocol)
        assert len(values) == 8 and {v['seq'] for v in values} == set(EXPECTED_TEST_LENGTHS) | {'COMBINED'}
        for sequence, count in EXPECTED_TEST_LENGTHS.items():
            coverage = manifest['coverage'][sequence]
            assert coverage['frames'] == count
            value = next(v for v in values if v['seq'] == sequence)
            assert value['CLR_Frames'] == count
            tracks = folder / 'track_results' / (sequence + '.txt')
            self.pin(tracks, coverage['output_sha256'])
            rows = tracks.read_text().splitlines()
            assert len(rows) == coverage['output_rows']
            assert all(1 <= int(row.split(',', 1)[0]) <= count for row in rows)
            source = manifest['sources'][sequence]
            self.pin(source['path'], source['sha256'])
            with np.load(source['path']) as array:
                np.testing.assert_array_equal(array['frame_ids'], np.arange(1, count + 1))
                assert array['offsets'][-1] == coverage['detections']
            if 'foreground_transport' in source:
                point = source['foreground_transport']
                self.pin(point['path'], point['sha256'])
        return dict(id=name, family=family, cue=cue, seed=0, frontend=config['detection_arm'],
            executed_fit_epochs=complete['epochs'], selected_epoch=epoch,
            checkpoint_sha256=self.pin(checkpoint_path), complete_fit_sha256=self.pin(complete_path),
            trainable_head_parameters=sum(p.numel() for p in model.parameters()),
            strict_model_load_and_finite_state=True, complete_Test_sequences=7, complete_Test_frames=15985,
            metrics39=result['trackeval'], official47_by_sequence={v['seq']: {
                k: x for k, x in v.items() if k not in ['candidate', 'split', 'seq']} for v in values},
            official_CSV=str(published.relative_to(ROOT)), official_CSV_sha256=self.pin(published),
            metrics_path=str(metrics_path), metrics_sha256=self.pin(metrics_path),
            original_replay_source_sha256=manifest['replay_source_sha256'],
            original_inference_git_commit=manifest['git_commit'],
            cached_replay_seconds=manifest['replay_seconds'], cached_replay_FPS=manifest['replay_FPS'],
            replay_cost_scope='Existing cached CPU replay, not detector/encoder/point/IR-generation-inclusive FPS.',
            last_recorded_fit_invocation_seconds=complete['seconds'],
            fit_seconds_scope='Final resumed fitting invocation only; not total cumulative training cost.',
            existing_artifacts_verified=True, new_inference=False, provenance_not_new_source_reproduction=True)


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    source_files = ['tools/audit_perceptual_control_coverage.py', 'tools/export_perceptual_metrics47.py',
        'tools/run_pseudo_ir_comparison.py', 'yolox/tracker/evidence_adapter.py',
        'yolox/tracker/association_calibrator.py', 'configs/perceptual_mft_20261122.json',
        'tools/train_perceptual_metric.py', 'tools/train_association_calibrator.py',
        'configs/perceptual_metric_20261122.json', 'configs/perceptual_pair_calibration_20261123.json',
        'configs/cfp/protocol_v2.json', 'configs/perceptual_single_seed_policy_20261123.json']
    before = {p: sha256(ROOT / p) for p in source_files}
    assert before['configs/perceptual_mft_20261122.json'] == '388e636d45840f3da8eae44cde2efc4731eeceadc7c169a39f19bf30488ecc03'
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    protocol = read(ROOT / plan['protocol'])
    assert set(protocol['historical_test']) == set(EXPECTED_TEST_LENGTHS)
    work = Path(plan['work_root'])
    audit = Audit()
    metric_selection_path = work / 'metric_epoch24_decision.json'
    pair_selection_path = DOCS / 'PERCEPTUAL_MFT_20261123_PAIR_SELECTION.json'
    metric_report_path = DOCS / 'PERCEPTUAL_MFT_20261123_FIXED_METRIC_TEST.json'
    pair_report_path = DOCS / 'PERCEPTUAL_MFT_20261123_FIXED_PAIR_TEST.json'
    for p in [metric_selection_path, pair_selection_path, metric_report_path, pair_report_path]:
        audit.pin(p)
    metric_selection, pair_selection = read(metric_selection_path), read(pair_selection_path)
    metric_report, pair_report = read(metric_report_path), read(pair_report_path)
    assert metric_selection['status'] == 'COMMON_STOP_AT_24' and not metric_selection['Test_selection']
    assert pair_selection['status'] == 'FROZEN_BEFORE_FORMAL_ORDINARY_TEST' and not pair_selection['Test_selection']
    assert metric_report['selection_decision'] == metric_selection
    assert pair_report['selection_sha256'] == sha256(pair_selection_path)
    schema, _ = official_schema(plan['evaluation']['TrackEval_root'])
    for name in ['hota.py', 'clear.py', 'identity.py', 'count.py', '_base_metric.py']:
        audit.pin(Path(plan['evaluation']['TrackEval_root']) / 'trackeval/metrics' / name)
    rows = []
    for cue in ['rgb', 'ir']:
        choice = next(r for r in metric_selection['rows'] if r['cue'] == cue and r['seed'] == 0)
        assert [r['epoch'] for r in choice['points']] == [6, 12, 18, 24]
        for point in choice['points']:
            result = audit.dev(point['path'], protocol)
            assert result['trackeval']['HOTA'] == point['HOTA']
        selected = min(choice['points'], key=lambda p: (-p['HOTA'], p['epoch']))
        assert selected['epoch'] == choice['selected_epoch']
        for frontend in (['new', 'rgb'] if cue == 'rgb' else ['new']):
            name = f'M_{frontend}_{cue}_s0_selected'
            old_row = next(r for r in metric_report['rows'] if r['id'] == name)
            rows.append(audit.result(work, protocol, schema, 'metric', name, cue, selected['epoch'], old_row))
    for choice in pair_selection['rows']:
        if choice['seed'] != 0:
            continue
        assert [r['epoch'] for r in choice['curve']] == [6, 12, 18, 24, 30, 36]
        for point in choice['curve']:
            result = audit.dev(point['Dev_metrics'], protocol, point['Dev_metrics_sha256'])
            audit.pin(point['configuration'], result['manifest']['config_sha256'])
            assert result['trackeval']['HOTA'] == point['HOTA']
        selected = min(choice['curve'], key=lambda p: (-p['HOTA'], p['epoch']))
        assert selected == choice['selected']
        audit.pin(choice['configuration'], choice['configuration_sha256'])
        audit.pin(choice['checkpoint'], choice['checkpoint_sha256'])
        name = f'K_{choice["frontend"]}_{choice["cue"]}_s0'
        old_row = next(r for r in pair_report['rows'] if r['id'] == name)
        rows.append(audit.result(work, protocol, schema, 'association_metric', name,
                                 choice['cue'], selected['epoch'], old_row))
    assert len(rows) == 7 and all(sha256(ROOT / p) == h for p, h in before.items())
    assert not torch.cuda.is_initialized()
    report = dict(status='COMPLETE_SCOPED_SEED0_ORDINARY_FORMAL_ROLES_ARTIFACT_AUDIT', audit_revision=1,
        rows=rows, named_formal_roles_verified=7, source_hashes=before,
        artifact_hashes=audit.artifact_hashes, GPU_work=False, new_fit_or_inference=False,
        new_TrackEval=False, additional_seed_work=False, Test_accuracy_used_for_selection=False,
        original_metric_selection_pending_flag_is_historical=True,
        scope='Only the three fixed24-epoch metric roles and four fixed36-epoch pair roles, '
            'all seed0; original Train/Dev choices and all7Test/15985frames verified. '
            'Old source hashes are preserved; this is not a replay of old results with current source.',
        not_covered=['Exploratory correlation/priority/trajectory backends and missing fullTest sources.',
            'neck_metric72 downstream heads and7Test.', 'Mixed-seed ROI configurations.',
            'Other current learned fusion pipelines, full runtime/memory/gate requirements, '
            'four architecture count and2-3positive multimodal contributions.'],
        current_primary_unchanged=True, full_formal_candidate_coverage=False,
        supported_innovations=0, goal_met=False)
    stem = DOCS / 'PERCEPTUAL_MFT_20261124_ORDINARY_CONTROL_COVERAGE'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    dump(stem.with_suffix('.json'), report)
    lines = ['# 七个既有普通对照的正式覆盖核验', '',
        '仅核验已经存在的seed0正式角色，没有新训练、图像推理、TrackEval或Test选点。', '',
        '| 角色 | 实际训练轮次 | 原Dev选中轮次 | 头参数量 | 七段Test帧数 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["id"]} | {row["executed_fit_epochs"]} | {row["selected_epoch"]} | '
                     f'{row["trainable_head_parameters"]} | {row["complete_Test_frames"]} |')
    lines += ['', '逐项核对原完整Dev曲线和冻结配置、全部训练历史及Train-only输入银行的真实SHA、'
        '真实检查点的严格加载与有限值、'
        '七段实际轨迹SHA/行数/帧范围、输入特征SHA及帧索引、39项官方摘要和未经改写的276列CSV。'
        'JSON保留七段及COMBINED的全部47指标，未用序列平均替换官方合并。', '',
        '度量头RGB选点沿用原配对策略：在原RGB前端Dev曲线选中同一RGB头，分别应用于RGB与new前端；'
        '没有另按new前端重选。普通关联头各前端/输入沿用其原六个Dev快照的选择。'
        '旧metric决策中的formal_Test_pending是真实历史快照，后续完整结果证明已完成，没有回写旧记录。', '',
        '这里验证既有来源，保留原执行代码SHA；没有声称当前代码重新复现了旧轨迹。'
        '既有replay_FPS仅为缓存CPU回放，排除检测器、编码器、点生成和伪IR生成；'
        '不能当完整系统FPS。complete.json的拟合秒数仅记录最后一次续训调用，'
        '没有冒充24/36轮累计训练成本。这些轮次是旧银行上的拟合，不能替代新六组的三遍全部27251帧。'
        '历史Test暴露与RGB预训练见过Dev的限制继续适用。', '',
        '这七个正常对照角色已有完整Test，不再因旧清单的“未核实”重复安排训练/评估。'
        '其余关联交互、neck72下游、混合种子ROI、四类融合架构、运行成本和多模态贡献仍须分别处理。'
        '普通控制的覆盖不证明新IR贡献，Goal仍未完成。']
    stem.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], roles=len(rows), new_evaluation=False,
                          verified_artifact_files=len(audit.artifact_hashes)), ensure_ascii=False))


if __name__ == '__main__':
    main()
