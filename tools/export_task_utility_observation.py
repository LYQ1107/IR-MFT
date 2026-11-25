#!/usr/bin/env python3
"""Publish small component evidence and the preserved R1 failure after actual exits."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fusion_memory_evaluation import read,sha
from tools.fusion_memory_observation import write_json
from tools.neck72_runtime_study import exited


def main():
    spec_path=ROOT/'configs/perceptual_task_utility_canonical_check_20261125.json';spec=read(spec_path)
    base=Path(spec['output_directory']);result_path=base/'complete.json';result=read(result_path)
    execution_path=base.with_suffix('.execution.json');execution=read(execution_path)
    assert execution['exit_code']==0 and not execution['RSS_limit_exceeded']
    assert sha(execution['log'])==execution['log_sha256']
    assert result['identity']==execution['identity'] and exited(execution['pid'],identity=execution['identity'])
    assert result['status']=='COMPLETE_TASK_UTILITY_CANONICAL_CPU_CHECK_WITH_R1_FAILURE_PRESERVED'
    assert result['specification_sha256']==execution['specification_sha256']==sha(spec_path)
    assert result['source_hashes']==spec['source_hashes'] and all(sha(ROOT/p)==s for p,s in spec['source_hashes'].items())
    assert result['actual_new_NN_inference_frames']==result['actual_new_optimizer_updates']==0
    assert result['reused_R1_Train_RGB_prefix_frames']==12 and result['actual_new_official_Dev_evaluated_frames']==4834
    assert result['all39_headlines_47_persequence_combined_and276_CSV_exact'] and result['R1_failure_preserved']
    assert result['peak_process_RSS_GiB']<=8 and result['actual_device']=='cpu'
    assert not any(result[k] for k in ['formal_GPU_eligibility','new_GPU_work','actual_TaskUtility_fitted_checkpoint_loaded',
        'new_TaskUtility_Dev_Test_accuracy','full_workflow_registered','goal_met'])
    for p,s in spec['preserved_R1_artifact_hashes'].items():assert sha(p)==s
    first=read(spec['preserved_R1_execution']);assert first['exit_code']==1 and exited(first['pid'],identity=first['identity'])
    assert sha(first['log'])==first['log_sha256'] and not first['RSS_limit_exceeded']
    report=result['original_vs_replayed_tracks']
    assert report['BT-005']['literal_files_exact'] and not report['SN-013']['literal_files_exact']
    assert report['SN-013']['original_minus_replayed_ID_offsets']==[22]
    assert all(v['all_non_ID_fields_literal_exact'] and v['whole_video_ID_bijection'] and
        not v['original_or_replayed_rows_rewritten'] for v in report.values())
    stem=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_OBSERVATION_COMPONENTS'
    targets=[stem.with_suffix('.json'),stem.with_suffix('.md'),Path(str(stem)+'_EXECUTION.json'),
        Path(str(stem)+'_CPU_LOG.txt'),Path(str(stem)+'_R1_FAILURE.json'),Path(str(stem)+'_R1_FAILURE_LOG.txt'),
        Path(str(stem)+'_OFFICIAL_DEV_276.csv')]
    assert not any(p.exists() for p in targets)
    csv=base/'official_Dev/trackeval_output/pedestrian_detailed.csv'
    assert sha(csv)==result['baseline_official_reproduction']['official_CSV_sha256']
    for source,target in [(result_path,targets[0]),(Path(execution['log']),targets[3]),(Path(first['log']),targets[5]),(csv,targets[6])]:
        assert source.stat().st_size<2**20;target.write_bytes(source.read_bytes())
    write_json(targets[2],dict(actual_execution=execution,actual_execution_sha256=sha(execution_path),
        public_result_sha256=sha(targets[0]),public_official_CSV_sha256=sha(targets[6]),exporter_sha256=sha(__file__),
        R1_failed_execution_preserved=True,no_new_method_accuracy=True,goal_met=False,supported_innovations=0),exclusive=True)
    write_json(targets[4],dict(status='PRESERVED_R1_LITERAL_MOT_FILE_ASSERTION_FAILURE_AFTER_PARTIAL_OUTPUTS',
        actual_execution=first,original_state=read(spec['preserved_R1_state']),
        specification_sha256=sha(ROOT/'configs/perceptual_task_utility_observation_components_20261125.json'),
        diagnosis=report,all_preserved_original_file_hashes=spec['preserved_R1_artifact_hashes'],
        explanation='R1incorrectly required literal global-ID MOT files to equal sequence-local-ID streams. '
            'Both complete videos match every non-ID field with a fixed video-wide ID bijection; '
            'SN-013original IDs have constant+22offset. R2independent official39/47/276CSV exact. '
            'No old or new row renumbered; R1still exit1; prior unrelated CPU numerical anomaly remains unknown.'),exclusive=True)
    units=[('OBSERVATION',ROOT/'configs/perceptual_task_utility_observation_components_20261125.json'),
           ('CANONICAL_CHECK',spec_path)]
    for label,config in units:
        evidence=read(config);receipt=Path(evidence['pytest_receipt']);unit=read(receipt)
        assert unit['exit_code']==0 and unit['all_sources_unchanged'] and sha(unit['log'])==unit['log_sha256']
        for source,extension in [(receipt,'.json'),(Path(unit['log']),'.txt')]:
            dest=Path(str(stem)+'_'+label+'_PYTEST'+extension);assert not dest.exists();dest.write_bytes(source.read_bytes())
    selection_receipt=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122/task_utility_selection_pytest_1795568625081322354.json')
    unit=read(selection_receipt);assert unit['exit_code']==0 and unit['all_sources_unchanged']
    assert all(sha(ROOT/p)==s for p,s in unit['source_hashes'].items()) and sha(unit['log'])==unit['log_sha256']
    for source,extension in [(selection_receipt,'.json'),(Path(unit['log']),'.txt')]:
        dest=Path(str(stem)+'_SELECTION_PYTEST'+extension);assert not dest.exists();dest.write_bytes(source.read_bytes())
    text=('# 任务效用候选：完整观测与评测组件\n\n'
        '新增逐帧生产、原RGB包重放、分离CPU官方评估、完整Dev阶段与Test前冻结组件。'
        '11项观测／配方、10项选择、6项ID比较人工检查通过；这些不代表真实正式训练或GPU资格。'
        '完整控制器、独立成本、总导出和集成资格仍待补齐，TaskUtility完整流程尚未登记。'
        '没有新方法Dev/Test准确率，独立贡献仍0。\n\n'
        '同容量12条件保留全部较差Dev模型；每个epoch的正常与四组IR缺失／原图roll均需完整4834帧。'
        '共同6／9／12遍预算按全部正常Dev曲线决定，RGB改善同样延长所有条件；'
        '不能在已触发停止后追加预算。实际完整Dev文件、原276列CSV、47项逐序列与合并、'
        '因果观测、MOT及原模型快照均重新绑定，再产生26项完整Test配方。'
        '三项门控去除和固定原RGB几何是推理干预，不冒充重新训练消融。'
        '目前真实TaskUtility Dev／Test／冻结文件均不存在，选择组件只有人工规则／文件测试。\n\n'
        '真实R1在BT-003前12帧完成当前RGB原图B1检测与新writer，原RGB权重／BN／梯度保持，'
        'MOT摘要精确匹配先前独立直接RGB前向。IR文件仅读字节验证来源12帧、实际解码0；'
        '不能据此声称学习IR分支已经接入本组件。先前IR worker的实际训练与多路NN证据另列。\n\n'
        'R1随后完整重放两段Dev的4834帧原冻结RGB检测包，却在第二段错误要求整个MOT文件字节一致而退出1。'
        '逐行诊断：BT-005的17931行完全相同；SN-013的27209行只因历史跨视频累计ID而比新逐视频ID大22，'
        '52个输出身份的全视频映射一一对应，所有帧号、框、分数及其他字段逐字符精确。'
        '没有重命名、覆盖或修补任一轨迹文件，完整失败源码／配置／日志／退出码与成本保留。\n\n'
        f"R1实际失败调用用时{first['seconds']:.3f}秒，采样RSS峰值{first['peak_sampled_RSS_GiB']:.3f}GiB；"
        '12帧阶段的169.038秒含源核验、加载和输出，不是独立模型FPS。'
        'R2直接复用保留的部分成功产物，未重复NN、训练或重放，也不将R1改记成功。'
        f"R2实际内部用时{result['seconds']:.3f}秒、RSS峰值{result['peak_process_RSS_GiB']:.3f}GiB，"
        f"单独官方评估{result['separate_official_evaluation_seconds']:.3f}秒。"
        '原39项摘要、47项逐序列／合并值与整份276列官方CSV全部精确复现；'
        'Dev HOTA47.579162182711485是既有RGB基线重现，不是新方法收益，也不等同Test HOTA29.529。\n\n'
        '本轮新增GPU工作0、正式TaskUtility快照加载0，旧CPU数值异常根因仍未知。'
        '九份旧登记与F4的384项动态来源保持；没有复制图像、FPN或新增权重。'
        '原六组完整Train已结束，50项Dev继续按原队列推进；仅物理4／5卡、固定seed0、'
        '工作盘64GiB上限保持。历史Test暴露、原RGB预训练见过Dev及RGB派生伪IR限制继续披露；'
        'Goal active，至少+5HOTA和2–3项独立贡献均未达到。\n')
    targets[1].write_text(text)
    print(json.dumps(dict(status='EXPORTED_TASK_UTILITY_OBSERVATION_COMPONENTS_WITH_R1_FAILURE',
        artificial_tests=27,R1_exit_code=1,R2_exit_code=0,new_method_accuracy=False,files=len(targets)+6)),flush=True)


if __name__=='__main__':main()
