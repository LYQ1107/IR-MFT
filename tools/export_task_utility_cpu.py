#!/usr/bin/env python3
"""Publish complete engineering evidence, never a fitted model or tracking gain."""
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fusion_memory_evaluation import read,sha
from tools.fusion_memory_observation import write_json
from tools.neck72_runtime_study import exited


def main():
    spec_path=ROOT/'configs/perceptual_task_utility_prototype_20261125.json';spec=read(spec_path)
    base=Path(spec['output_directory']);value=read(base/'complete.json')
    execution_path=base.parent/'task_utility_CPU_engineering_R2_20261125.execution.json';execution=read(execution_path)
    assert execution['exit_code']==0 and sha(execution['log'])==execution['log_sha256']
    assert value['status']=='COMPLETE_ACTUAL_TASK_UTILITY_CPU_ENGINEERING_NOT_FORMAL_FIT_OR_GAIN'
    assert value['identity']==execution['identity'] and exited(execution['pid'],identity=execution['identity'])
    assert value['specification_sha256']==execution['specification_sha256']==sha(spec_path)
    assert value['source_hashes']==spec['source_hashes'] and all(sha(ROOT/p)==s for p,s in spec['source_hashes'].items())
    assert value['actual_Train_frames']==value['Train_domains']==6 and value['optimizer_updates_total']==72
    assert len(value['image_content_records'])==12 and len(value['Train_GT_hashes'])==6
    assert not any(value[k] for k in ['new_GPU_work','saved_weights','new_image_FPN_ROI_disk_cache','formal_launch_ready','new_Dev_Test_accuracy','goal_met'])
    assert value['original_RGB_weights_BN_and_gradients_unchanged'] and value['peak_process_RSS_GiB']<=4
    expected={v['id']+'_'+c for v in spec['variants'] for c in spec['matched_cues']}
    assert {r['id'] for r in value['actual_conditions']}==expected and len(expected)==12
    summaries=[]
    for row in value['actual_conditions']:
        steps=row['steps'];assert len(steps)==6 and [s['address'] for s in steps]==spec['CPU']['addresses']
        assert row['initial_raw_RGB_exact'] and len(row['missing_checks'])==6 and all(s['missing_raw_head_bit_exact'] for s in row['missing_checks'])
        variant=row['variant'];assert variant in spec['variants']
        fg=sum(s['weighted_foreground_direct_gradient_L1'] for s in steps)
        utility=sum(s['weighted_utility_direct_gradient_L1'] for s in steps)
        assert (fg>0)==variant['supervise_foreground'] and (utility>0)==variant['supervise_utility']
        assert steps[0]['weighted_utility_direct_gradient_L1']==0
        if row['cue']=='ir':assert row['actual_IR_image_roll_changes_raw_head'] is True
        summaries.append(dict(condition=row['id'],foreground_direct_gradient_L1_sum=fg,utility_direct_gradient_L1_sum=utility,
            maximum_decoded_box_change=max(s['decoded_box_max_change'] for s in steps),
            helpful_teacher_labels=sum(x['helpful_cells'] for s in steps for x in s['local_utility_labels']),
            harmful_teacher_labels=sum(x['harmful_cells'] for s in steps for x in s['local_utility_labels']),
            cached_reference_max_abs=max(s['cached_RGB_reference_max_abs'] for s in steps),
            cached_reference_total_different_values=sum(s['cached_RGB_reference_different_values'] for s in steps),
            initial_zero_current_RGB_exact=True,missing_frames_exact=6,IR_image_roll_raw_change=row['actual_IR_image_roll_changes_raw_head']))
    docs=ROOT/'docs/fusion_research';stem=docs/'PERCEPTUAL_MFT_20261125_TASK_UTILITY_CPU_ENGINEERING'
    outputs=[stem.with_suffix('.json'),stem.with_suffix('.md'),Path(str(stem)+'_EXECUTION.json'),Path(str(stem)+'_CPU_LOG.txt')]
    assert not any(p.exists() for p in outputs)
    raw=(base/'complete.json').read_bytes();assert len(raw)<2**20;outputs[0].write_bytes(raw)
    write_json(outputs[2],dict(actual_execution=execution,execution_sha256=sha(execution_path),result_sha256=sha(base/'complete.json'),
        exporter_sha256=sha(__file__),summary=summaries,formal_fit=False,new_tracking_accuracy=False,supported_innovations=0,goal_met=False),exclusive=True)
    log=Path(execution['log']).read_bytes();assert len(log)<2**20;outputs[3].write_bytes(log)
    lines=['# 任务相关空间监督：真实CPU工程检查','',
        '12个条件全部完成，共六个Train视频／六帧、每条件六次更新、总72次工程更新。'
        '这不是完整Train拟合、GPU资格、收敛、Dev/Test或HOTA提升。独立创新仍为0。','',
        '四种条件始终保留同一推理结构和553705参数，仅开关Train前景／效用监督。'
        '每种均有重复RGB、灰度、新版IR；相同seed0、初始化权重和样本顺序。'
        '额外门控容量相同，不能把监督效果混为新增容量。','',
        '前景标签是Train框覆盖的网格中心，不是真实分割。效用标签用同一冻结RGB的SimOTA分配，'
        '比较原RGB与固定0.5门控、已detach提案的逐锚点原YOLOX损失；包括目标性、前景分类及5倍IoU项。'
        '差值／温度0.1经sigmoid转为软目标，绝对差值决定标签权重，目标和权重均停止梯度。'
        '零效应对应零标签权重，不能用零初始化伪造正负监督。','',
        '主损失仍使用原YOLOX动态标签分配与几何回归，另保留可靠RGB项。'
        '推理只读取当前RGB和当前辅助像素，无GT、过去或未来帧；前景／效用logit加入质量logit，'
        '残差仍有RMS界和RGB保护，缺失模态返回原RGB。效用不是校准概率或因果效应。','',
        '| 条件 | 前景监督直接梯度和 | 效用监督直接梯度和 | 框最大变化 | 有用局部标签 | 有害局部标签 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summaries:
        lines.append(f"| {row['condition']} | {row['foreground_direct_gradient_L1_sum']:.6f} | {row['utility_direct_gradient_L1_sum']:.6f} | {row['maximum_decoded_box_change']:.6f} | {row['helpful_teacher_labels']} | {row['harmful_teacher_labels']} |")
    lines+=['','这些梯度及框变化证明计算路径被实际使用，不能证明框更正确、FG预测有效或跟踪收益。'
        '原始每步损失、各层标签、门控变化、72项缺失检查和四项实际IR图像roll响应均完整保存。'
        '初始零修正检查覆盖每条件首个Train帧；缺失检查覆盖全部六个工程帧，不能冒充完整序列。','',
        'R1在首次零修正处失败、更新0次，源码与日志保留在提交60d6093及R1_FAILURE记录。'
        '诊断中相同FPN和权重的CPU head也曾相对旧参考产生宽高差异；其他小诊断没有复现，根因仍未知。'
        'R2按现有生产训练器为每个优化输入新算RGB head，全部比较继续用torch.equal。'
        '这不证明CPU跨调用上下文数值已普遍稳定，也不修复或放行旧F4 CPU测速异常。','',
        f"CPU用时{value['seconds']:.3f}秒，峰值RSS{value['peak_process_RSS_GiB']:.3f}GiB。"
        '严格加载原RGB源，原权重／BN／梯度不变；12个实际原图文件与完成拟合的内容摘要匹配，六份TrainGT摘要核对。'
        '不保存新权重、图像或FPN缓存，原GPU训练和全部已登记队列未改。','',
        '前景指导及模态选择损失已有[GAFF先例](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.pdf)；'
        '任务驱动融合也有[TarDAL先例](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Target-Aware_Dual_Adversarial_Learning_and_a_Multi-Scenario_Multi-Modality_Benchmark_To_CVPR_2022_paper.html)。'
        '本轮只核对主站可索引正文／摘要，完整PDF抓取失败，没有声称完整重现或复制这两个项目的代码。'
        '固定分配局部效用也未证明原创性。','',
        '若后续正式登记，仍需相同完整Train预算、完整Dev冻结、全部正常／消融／错配／缺失完整七段Test、'
        '47字段／原276列CSV与独立成本。CPU工程通过不会自动启动正式拟合或增加四方向完成数。'
        '历史Test暴露、RGB预训练见过Dev、伪IR由RGB派生等限制不变；Goal继续active。']
    outputs[1].write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='EXPORTED_ALL12_TASK_UTILITY_CPU_ENGINEERING_NO_FORMAL_GAIN',summary=summaries,
        seconds=value['seconds'],peak_process_RSS_GiB=value['peak_process_RSS_GiB'])),flush=True)


if __name__=='__main__':main()
