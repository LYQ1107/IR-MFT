#!/usr/bin/env python3
"""Publish actual one-arm worker CPU evidence after verified exit; no gain claim."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fusion_memory_evaluation import read,sha
from tools.fusion_memory_observation import write_json
from tools.neck72_runtime_study import exited


def main():
    spec_path=ROOT/'configs/perceptual_task_utility_worker_components_20261125.json';spec=read(spec_path)
    base=Path(spec['output_directory']);result_path=base/'complete.json';result=read(result_path)
    execution_path=base.parent/'task_utility_worker_CPU_engineering_20261125.execution.json';execution=read(execution_path)
    assert execution['exit_code']==0 and not execution['RSS_limit_exceeded']
    assert sha(execution['log'])==execution['log_sha256']
    assert result['identity']==execution['identity'] and exited(execution['pid'],identity=execution['identity'])
    assert result['status']=='COMPLETE_ONE_IR_TASK_UTILITY_WORKER_TRAIN_B1_CPU_NOT_GPU_ELIGIBILITY'
    assert result['specification_sha256']==execution['specification_sha256']==sha(spec_path)
    assert result['source_hashes']==spec['source_hashes'] and all(sha(ROOT/p)==s for p,s in spec['source_hashes'].items())
    assert result['fit_id']=='TU_foreground_utility_labels_ir' and result['actual_optimizer_updates']==3
    assert result['actual_Train_frames']==result['actual_B1_inference_frames']==12 and result['Train_domains']==1
    assert result['complete_Train_epochs']==0 and result['original_RGB_weights_BN_gradients_unchanged']
    assert len(result['image_content_records'])==24 and len(result['Train_GT_hashes'])==6
    assert not any(result[k] for k in ['new_GPU_work','formal_GPU_eligibility','full_workflow_registered',
        'saved_weights','new_image_FPN_disk_cache','new_Dev_Test_accuracy','goal_met'])
    assert result['peak_process_RSS_GiB']<=8 and result['actual_device']=='cpu'
    value=result['actual_record'];rows=value['optimizer_rows'];online=value['inference'];checks=online['checks']
    assert len(rows)==3 and value['actual_updates']==3 and value['complete_Train_epochs']==0
    assert rows[0]['initial_full_RGB_exact'] and rows[0]['initial_proposal_full_RGB_exact']
    assert rows[0]['utility_label_mass']==rows[0]['utility_supervision_direct_gradient_L1']==0
    assert sum(r['foreground_supervision_direct_gradient_L1'] for r in rows)>0
    assert sum(r['utility_supervision_direct_gradient_L1'] for r in rows)>0
    assert value['existing_AdamW_moments_unchanged'] and not value['all_missing_training']['optimizer_updated']
    assert online['B1_frames']==len(checks)==12 and not online['GT_read'] and not online['future_frame_input']
    assert [r['frame'] for r in checks]==list(range(1,13))
    for row in checks:
        assert row['sequence']=='BT-003' and row['direct_zero_missing_raw_head_and_NMS_exact']
        assert row['direct_zero_missing_MOT_ID_KF_exact'] and row['geometry_packet_matches_independent_coordinate_intervention']
        assert row['IR_roll']['actual_input_and_latent_changed'] and not row['GT_read'] and not row['future_frame_input']
    hashes=online['MOT_prefix_sha256'];assert len({hashes[k] for k in ['RGB','zero','missing','direct']})==1
    assert online['frozen_inference_states_exact']
    unit_path=Path(result['pytest_receipt']);unit=read(unit_path)
    assert sha(unit_path)==result['pytest_receipt_sha256'] and unit['exit_code']==0 and unit['all_sources_unchanged']
    assert sha(unit['log'])==unit['log_sha256']
    prior=read(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_T4_CPU_ENGINEERING.json')
    previous=next(v for v in prior['conditions'] if v['id']=='foreground_utility_labels_ir')['steps']
    scalar_differences=[]
    for index,(old,new) in enumerate(zip(previous,rows)):
        for field in sorted(set(old)&set(new)):
            if old[field]!=new[field]:scalar_differences.append(dict(position=index,field=field,previous=old[field],current=new[field]))
    summary=dict(optimizer_updates=3,B1_frames=12,independent_direct_zero_missing_raw_NMS_MOT_ID_KF_exact=True,
        actual_IR_input_and_latent_roll_changed_frames=12,
        raw_head_roll_changed_frames=sum(r['IR_roll']['raw_head_changed'] for r in checks),
        detection_packet_roll_changed_frames=sum(r['IR_roll']['packet_changed'] for r in checks),
        geometry_intervention_exact_frames=12,previous_T4_common_scalar_rows_exact=not scalar_differences,
        previous_T4_scalar_differences=scalar_differences,original_CPU_numerical_anomaly_cause_still_unknown=True)
    stem=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_WORKER_CPU_ENGINEERING'
    paths=[stem.with_suffix('.json'),stem.with_suffix('.md'),Path(str(stem)+'_EXECUTION.json'),Path(str(stem)+'_CPU_LOG.txt')]
    assert not any(p.exists() for p in paths)
    raw=result_path.read_bytes();log=Path(execution['log']).read_bytes();assert len(raw)<2**20 and len(log)<2**20
    paths[0].write_bytes(raw);paths[3].write_bytes(log)
    write_json(paths[2],dict(actual_execution=execution,execution_sha256=sha(execution_path),result_sha256=sha(result_path),
        exporter_sha256=sha(__file__),summary=summary,formal_GPU_eligibility=False,new_tracking_accuracy=False,
        supported_innovations=0,goal_met=False),exclusive=True)
    text=('# 任务效用候选：真实训练worker与逐帧CPU路径\n\n'
        '一个IR联合监督条件已实际完成三次四帧Train更新，然后连续12帧B1原图检测／跟踪检查。'
        '仅BT-003第1–12帧、一个Train域，完整Train epoch为0；没有正式拟合、GPU资格、'
        'Dev/Test或HOTA提升，已验证独立贡献0。\n\n'
        '本次调用新worker的实际canary函数，Train保持480×864及500步warmup，'
        'B1保持800×1440。前景／效用直接监督梯度均非零、首步中性效用权重为0；'
        '全缺失Train片段跳过AdamW，已有矩状态及辅助权重不变。\n\n'
        '逐帧路径不接收GT或provider目标：正常、零修正、学习后缺失、IR原图roll、'
        '固定原RGB坐标干预及RGB参照均使用当前观测。另用独立的完整RGB前向作参照；'
        '12帧的RGB／零／缺失完整原始head、NMS包、MOT、局部ID计数器和KF状态逐值相等，'
        '固定坐标结果也匹配独立组合再NMS。没有改写以前轨迹。\n\n'
        f"IR实际输入与至少一层原生特征在12帧均因原图roll改变；原始head有{summary['raw_head_roll_changed_frames']}帧变化，"
        f"检测包有{summary['detection_packet_roll_changed_frames']}帧变化。"
        '这些只是输入进入计算路径的检查，不能证明框更正确、语义互补或跟踪改善。'
        '门控／前景／效用和修正仅保存当前有效网格的标量统计；缺失帧不虚构统计。\n\n'
        f"同一IR条件的三次更新与上一轮独立T4检查共同字段完全相等：{not scalar_differences}。"
        '逐项差异列表完整保存；这不排除其他CPU调用上下文的数值差异，旧CPU及F4异常根因仍未知。\n\n'
        f"实际CPU用时{result['seconds']:.3f}秒，峰值RSS{result['peak_process_RSS_GiB']:.3f}GiB。"
        '含来源核对、加载、Train和多路逐帧复核，不能作为独立系统FPS。'
        '64MiB编码图像RAM，24个原图文件与完成IR111拟合的内容SHA一致，六份TrainGT核验；'
        '原RGB权重／BN／梯度不变，不保存新权重、图像或FPN缓存。\n\n'
        '新增研究配置保留四监督组合×三cue的12个计划正常拟合与26项完整Test配方。'
        '当前完整pipeline登记缺失、F4前序仍未释放；正式GPU必须等待全部原队列导出和'
        '实际进程退出，再完成12个实际GPU资格检查。CPU结果写在独立目录，不生成GPU资格文件。'
        '生产／评估、独立成本、完整控制器和导出仍待接通；不能把本轮标为完整正式流程。\n\n'
        '模型文件加载组件只有人工文件／张量测试；尚无真实正式TaskUtility快照被加载。'
        '本次临时学习权重直接留在内存完成B1检查，不能冒充已完成训练模型或F4学习融合记忆转移。'
        '历史Test暴露、原RGB预训练见过Dev、伪IR由RGB生成以及先验工作边界保持；Goal active。\n')
    paths[1].write_text(text)
    print(json.dumps(dict(status='EXPORTED_ONE_REAL_IR_WORKER_CPU_COMPONENT_NO_FORMAL_GAIN',summary=summary,
        seconds=result['seconds'],peak_process_RSS_GiB=result['peak_process_RSS_GiB'])),flush=True)


if __name__=='__main__':main()
