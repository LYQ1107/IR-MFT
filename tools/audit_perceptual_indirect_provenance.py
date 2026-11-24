#!/usr/bin/env python3
"""Trace existing encoder use and whole-config seed boundaries, read only."""
import collections
import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    work = Path(plan['work_root'])
    source = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_COMMON_TEST_BACKFILL.json'
    completed = json.loads(source.read_text())
    assert completed['status'] == 'COMPLETE_FROZEN_NATIVE36_ALL26_TEST_BACKFILL'
    assert len(completed['rows']) == 26
    cached = {}
    def digest(path):
        path = str(Path(path))
        if path not in cached: cached[path] = sha256(path)
        return cached[path]
    normal = [r for r in completed['rows'] if r['id'].endswith('_normal')]
    assert len(normal) == 8
    encoder_rows = []
    for family, fit_family, train_family, eval_family in [
        ('native36', 'native_cue', 'native_train_features', 'native_eval_features'),
        ('neck_metric72', 'native_neck_metric', 'native_neck_train_features', 'native_neck_eval_features')]:
        for cue in ['rgb', 'ir']:
            complete_path = work / fit_family / (cue+'_s0') / 'complete.json'
            fit = json.loads(complete_path.read_text())
            assert fit['seed'] == 0 and fit['Dev_fit_rows'] == fit['Test_fit_rows'] == 0
            trained_path = work / train_family / (cue+'_s0') / 'complete.json'
            trained = json.loads(trained_path.read_text())
            checkpoint = Path(trained['checkpoint'])
            assert digest(checkpoint) == trained['checkpoint_sha256']
            assert trained['cue'] == cue and trained['seed'] == 0
            assert trained['observations'] == 11967 and trained['Dev_fit_rows'] == trained['Test_fit_rows'] == 0
            if family == 'neck_metric72':
                assert fit['cumulative_native_epochs'] == 72
                endpoint = work / 'native_cue' / (cue+'_s0') / 'epoch36_fp32.pt'
                assert fit['initial_checkpoint_sha256'] == digest(endpoint)
            actual_uses = []
            for row in normal:
                config, manifest = row['config'], row['manifest']
                if config.get('native_feature_family', 'native36') != family or config['native_cue'] != cue:
                    continue
                assert config['native_seed'] == 0 and not config.get('disable_native_increment', False)
                assert manifest['frames'] == 15985 and manifest['sequences'] == protocol['historical_test']
                streams = {}
                for seq in protocol['historical_test']:
                    # The loader consumes native_s0 from this cue. Merely listing
                    # s1/s2 checkpoints in a shared packet is not use of them.
                    native = manifest['sources'][seq]['native_identity'][cue]
                    assert native['checkpoints']['native_s0']['sha256'] == digest(checkpoint)
                    path = Path(native['path'])
                    assert eval_family in path.parts and digest(path) == native['sha256']
                    producer_path = path.parent / 'complete.json'
                    producer = json.loads(producer_path.read_text())
                    assert producer['checkpoints']['native_s0']['sha256'] == digest(checkpoint)
                    assert not producer['GT_read'] and not producer['future_frame_input']
                    assert producer['split'] == 'test' and producer['sequence'] == seq
                    streams[seq] = dict(packet=str(path), packet_sha256=digest(path),
                        producer_sha256=digest(producer_path), selected_array='native_s0',
                        frames=producer['frames'])
                actual_uses.append(dict(case=row['id'], metrics_sha256=row['metrics_sha256'], streams=streams,
                    official_CSV_sha256=row['official_detailed_csv_sha256']))
            dev, test = [], []
            for split, target in [('dev', dev), ('test', test)]:
                for seq in protocol['dev' if split == 'dev' else 'historical_test']:
                    base = work / eval_family / split / seq / cue
                    path = base / 'seed_0/complete.json' if family == 'neck_metric72' else base / 'complete.json'
                    if path.exists():
                        info = json.loads(path.read_text())
                        assert info['checkpoints']['native_s0']['sha256'] == digest(checkpoint)
                        target.append(dict(sequence=seq, frames=info['frames'], manifest=str(path), manifest_sha256=digest(path)))
            encoder_rows.append(dict(family=family, cue=cue, seed=0,
                encoder_complete_sha256=digest(complete_path), Train_feature_manifest_sha256=digest(trained_path),
                checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                encoder_executed_epochs=fit.get('cumulative_native_epochs', fit['epochs']),
                Dev_manifests=dev, Test_manifests=test,
                complete_Test_uses_verified=actual_uses,
                status='INDIRECT_ALL7_TEST_DEPENDENCY_VERIFIED' if actual_uses else 'DOWNSTREAM_TRACKER_AND_ALL7_TEST_NOT_COMPLETED',
                independent_multimodal_benefit_established=False))
    roi = []
    for path in sorted((work / 'roi_confirmed_recovery_dev').glob('*/metrics.json')):
        value = json.loads(path.read_text())
        manifest, config = value['manifest'], value['manifest']['config']
        assert value['status'] == 'COMPLETE' and manifest['frames'] == 4834 and manifest['split'] == 'dev'
        match = re.search(r'_s([0-9]+)$', Path(config['association_checkpoint']).parent.name)
        assert match is not None
        primary, auxiliary = int(match.group(1)), config['auxiliary_seed']
        roi.append(dict(id=config['id'], primary_seed=primary, auxiliary_seed=auxiliary,
            disabled_ROI=config.get('disable_ROI_confirmation', False),
            all_loaded_models_seed0=(primary == auxiliary == 0),
            config_sha256=manifest['config_sha256'], metrics_sha256=digest(path),
            primary_checkpoint=config['association_checkpoint'], auxiliary_checkpoint=config['auxiliary_checkpoint']))
    assert len(roi) == 32 and not any(r['all_loaded_models_seed0'] for r in roi)
    unfinished = {}
    for name in ['native_neck_robust_query', 'native_neck_clean_query']:
        unfinished[name] = [str(p) for p in (work / name).glob('*/complete.json')]
    assert not any(unfinished.values())
    stem = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_INDIRECT_ENCODER_AND_SEED_AUDIT'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    report = dict(status='COMPLETE_SCOPED_INDIRECT_ENCODER_AND_WHOLE_CONFIG_SEED_AUDIT',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), encoder_rows=encoder_rows,
        source_complete26_report_sha256=digest(source), ROI_original_Dev_rows=roi,
        missing_neck_head_completions=unfinished, source_hashes={name: digest(ROOT / name) for name in [
            'tools/audit_perceptual_indirect_provenance.py', 'tools/native_query_replay.py',
            'yolox/tracker/native_query_tracker.py', 'yolox/tracker/roi_confirmed_recovery.py',
            'configs/perceptual_single_seed_policy_20261123.json']},
        source_artifact_hashes=cached, GPU_work=False, new_fit_or_inference=False,
        additional_seed_work=False, Test_accuracy_used_for_selection=False,
        full_formal_candidate_coverage=False, goal_met=False, supported_innovations=0,
        interpretation='native36RGB/IR encoder endpoints are now verified indirect dependencies of existing full Test. '
            'The separate neck72 repair remains unfinished downstream; it is not silently covered by native36. '
            'ROI configurations load cyclic nonzero-seed auxiliary models even when filenames say s0. '
            'Disabled-ROI cases may never call those auxiliaries, but this audit does not assume or claim source-equivalent full Test. '
            'The latest policy forbids new seed1/2 inference; preserve old mixed-seed results without relabelling them strictseed0.')
    dump(stem.with_suffix('.json'), report)
    lines = ['# 间接编码器来源与完整配置种子核查', '',
        '本项只读既有权重、缓存来源和已完成报告，没有新训练、推理、GPU或准确率选择。不能据此宣布正式候选已经全覆盖。', '',
        '| 表征 | 输入 | 累积编码训练轮次 | 完整Dev特征 | 完整Test特征 | 验证到的正常完整Test使用 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in encoder_rows:
        lines.append(f"| {row['family']} | {row['cue']} | {row['encoder_executed_epochs']} | {len(row['Dev_manifests'])} | "
                     f"{len(row['Test_manifests'])} | {len(row['complete_Test_uses_verified'])} |")
    lines += ['',
        'native36的RGB与IR各在四个已冻结正常Test条件中实际使用，均覆盖7序列/15985帧。核对训练端点SHA、特征生产记录、回放实际消费的native_s0和完整结果来源。共享NPZ记录含旧s1/s2不表示本轮调用了它们，关闭native增量的行也不用于证明辅助编码器已被消费。', '',
        'neck_metric72是另一套修复端点，包含原36轮加后续36轮。它的两个seed0模型只有两段Dev的特征，尚无七段Test特征，也没有neck clean/robust跟踪头完成记录；不能用已完成native36结果替代。原训练和负的Dev身份辨别诊断均保留，这条下游路线继续列为未完成。', '',
        '原ROI确认共有32项完整Dev。12项primary_s0中，8项正常与4项关闭ROI配置都加载auxiliary_s1；其余primary_s1/2同样使用循环种子。整套配置没有一项所有加载模型均为seed0。关闭ROI时辅助可能不做forward，这里不擅自推断对应Test源等价。现行用户策略禁止新增seed1/2推理，因此不把这些历史混合种子路线伪装为当前纯seed0补测，也不据负Dev把它们改叫诊断。', '',
        '下一步仍须完成其他正式路线的逐项登记核查，以及neck72下游范围的明确处理。当前全量Train和新原生金字塔队列保持不变；既有失败、历史Test暴露、RGB预训练见过Dev的限制继续适用。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'], indirect_encoders_with_Test=sum(bool(r['complete_Test_uses_verified'])for r in encoder_rows),
        neck72_downstream_incomplete=True, ROI_strict_seed0_configs=0, goal_met=False)), flush=True)


if __name__ == '__main__': main()
