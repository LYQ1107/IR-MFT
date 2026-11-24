#!/usr/bin/env python3
"""Export every real T4 CPU condition after verified exit, without efficacy claims."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fusion_memory_evaluation import read,sha
from tools.fusion_memory_observation import write_json
from tools.neck72_runtime_study import exited


def main():
    spec_path=ROOT/'configs/perceptual_task_utility_training_components_20261125.json'
    spec=read(spec_path);base=Path(spec['output_directory'])
    result=read(base/'complete.json')
    execution_path=base.parent/'task_utility_T4_CPU_engineering_R2_20261125.execution.json'
    execution=read(execution_path)
    assert execution['exit_code']==0 and not execution['RSS_limit_exceeded']
    assert sha(execution['log'])==execution['log_sha256']
    assert result['identity']==execution['identity'] and exited(execution['pid'],identity=execution['identity'])
    assert result['specification_sha256']==execution['specification_sha256']==sha(spec_path)
    assert result['source_hashes']==spec['source_hashes']
    assert all(sha(ROOT/p)==s for p,s in spec['source_hashes'].items())
    assert result['status']=='COMPLETE_ACTUAL_T4_TASK_SUPERVISION_API_CPU_NOT_FORMAL_FIT_OR_GAIN'
    assert result['optimizer_updates_total']==36 and result['missing_clip_optimizer_calls']==12
    assert result['actual_unique_Train_frames']==12 and result['Train_domains']==1
    assert result['actual_Train_frame_visits']==144 and result['all_missing_frame_rechecks']==48
    assert result['complete_Train_epochs']==0 and result['parameters_each']==553705
    assert result['source_RGB_weights_BN_and_gradients_unchanged'] and result['peak_process_RSS_GiB']<=8
    assert len(result['image_content_records'])==24 and len(result['Train_GT_hashes'])==6
    assert result['actual_device']=='cpu' and not any(result[k] for k in [
        'new_GPU_work','formal_launch_ready','new_Dev_Test_accuracy','saved_weights','new_image_FPN_disk_cache','goal_met'])
    prototype=read(ROOT/spec['prototype']['path'])
    expected=[v['id']+'_'+c for v in prototype['variants'] for c in prototype['matched_cues']]
    assert [v['id'] for v in result['conditions']]==expected and len(expected)==12
    unit_path=Path(result['pytest_receipt']);unit=read(unit_path)
    assert sha(unit_path)==result['pytest_receipt_sha256'] and unit['exit_code']==0
    assert unit['all_sources_unchanged'] and sha(unit['log'])==unit['log_sha256']
    assert all(spec['source_hashes'][p]==s for p,s in unit['source_hashes'].items())
    summaries=[];prefixes=set()
    for condition in result['conditions']:
        steps=condition['steps'];supervision=condition['supervision']
        assert len(steps)==3 and condition['delivered_Train_clips']==3
        assert [dict(sequence=s['sequence'],frames=s['frame_indices']) for s in steps]==spec['CPU']['ordered_clips']
        assert all(s['frames']==s['available_frames']==4 and s['missing_frames']==0 and s['optimizer_updated'] for s in steps)
        assert steps[0]['initial_full_RGB_exact'] and steps[0]['initial_proposal_full_RGB_exact']
        assert steps[0]['utility_label_mass']==steps[0]['utility_supervision_direct_gradient_L1']==0
        for position,step in enumerate(steps):
            assert step['learning_rate']==spec['training']['fit']['learning_rate']*((position+1)/500)
            loss=step['full_YOLOX']+.02*step['reliable_RGB']
            loss+=.1*supervision['foreground']*step['foreground']+.1*supervision['utility']*step['utility']
            assert abs(loss-step['loss'])<=2e-6*max(1.,abs(loss))
        foreground=sum(s['foreground_supervision_direct_gradient_L1'] for s in steps)
        utility=sum(s['utility_supervision_direct_gradient_L1'] for s in steps)
        assert (foreground>0)==supervision['foreground'] and (utility>0)==supervision['utility']
        missing=condition['all_missing_clip']
        assert condition['all_missing_skips_AdamW_with_existing_moments_exact']
        assert missing['initial_full_RGB_exact'] and missing['initial_proposal_full_RGB_exact']
        assert not missing['optimizer_updated'] and missing['frames']==missing['missing_frames']==4
        assert missing['valid_task_cells']==missing['utility_label_mass']==missing['decoded_box_max_change']==0
        assert not condition['formal_fit'] and condition['complete_Train_epochs']==0
        prefixes.add(condition['actual_address_prefix_sha256'])
        summaries.append(dict(condition=condition['id'],foreground_direct_gradient_L1_sum=foreground,
            utility_direct_gradient_L1_sum=utility,maximum_decoded_box_change=max(s['decoded_box_max_change'] for s in steps),
            helpful_teacher_cells=sum(s['helpful_teacher_cells'] for s in steps),
            harmful_teacher_cells=sum(s['harmful_teacher_cells'] for s in steps),
            utility_label_mass=sum(s['utility_label_mass'] for s in steps),
            initial_raw_RGB_exact=True,all_missing_raw_RGB_exact=True,all_missing_AdamW_unchanged=True))
    assert len(prefixes)==1
    stem=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_T4_CPU_ENGINEERING'
    outputs=[stem.with_suffix('.json'),stem.with_suffix('.md'),Path(str(stem)+'_EXECUTION.json'),Path(str(stem)+'_CPU_LOG.txt')]
    assert not any(p.exists() for p in outputs)
    payload=(base/'complete.json').read_bytes();log=Path(execution['log']).read_bytes()
    assert len(payload)<2**20 and len(log)<2**20
    outputs[0].write_bytes(payload);outputs[3].write_bytes(log)
    write_json(outputs[2],dict(actual_execution=execution,execution_sha256=sha(execution_path),
        result_sha256=sha(base/'complete.json'),exporter_sha256=sha(__file__),summary=summaries,
        every_missing_raw_head_exact=True,formal_fit=False,new_tracking_accuracy=False,supported_innovations=0,goal_met=False),exclusive=True)
    lines=['# 任务效用监督：真实四帧CPU训练接口检查','',
        '12个条件全部实际完成，每条件三个连续四帧片段，共36次优化更新；'
        '另有12次全缺失检查调用、均跳过AdamW。12个唯一Train帧来自BT-003一个视频，'
        '正常交付144帧次，缺失复核另外48帧次；完整Train epoch为0。'
        '没有GPU资格、正式拟合、Dev/Test或HOTA提升，独立创新仍为0。','',
        '这次使用新optimizer_step接口，每次都从真实像素调用原RGB编码器／检测头，'
        '输入480×864、四帧批次；没有用预存FPN替代。保持正式计划的500步warmup及'
        '12遍固定学习率周期，前三步学习率分别6e−7、1.2e−6、1.8e−6。'
        '冻结原RGB权重、BN及梯度，相同seed0初始化、553705参数和地址顺序；'
        '四种条件保持相同推理结构，仅改变前景／效用监督开关。','',
        '每步总损失核对为原YOLOX＋0.02可靠RGB保持项＋各0.1的启用监督。'
        '首步完整原始head输出及提案与新算RGB逐值相等，零效应对应零效用标签权重总量。'
        '全缺失检查覆盖四帧的完整原始head、辅助权重和已有AdamW状态，逐值核对。'
        '这里的原始head检查没有包括NMS、ByteTrack或完整在线推理。','',
        '| 条件 | 前景监督直接梯度和 | 效用监督直接梯度和 | 框最大变化 | 有用局部标签 | 有害局部标签 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:
        lines.append(f"| {s['condition']} | {s['foreground_direct_gradient_L1_sum']:.9g} | {s['utility_direct_gradient_L1_sum']:.9g} | {s['maximum_decoded_box_change']:.9g} | {s['helpful_teacher_cells']} | {s['harmful_teacher_cells']} |")
    lines+=['','非零监督梯度与框变化证明路径被使用；它们不证明预测更正确或跟踪收益。'
        '局部效用使用冻结RGB同一SimOTA分配下的原检测损失差，包含几何，'
        '是停止梯度的训练教师信号，不是因果效应或校准可靠性。','',
        '首次R1在第一步优化返回后因日志字段重名退出；实际完成1次调用，'
        '状态计数0、数值行没有恢复。提交60e781f及R1_FAILURE完整保留。'
        'R2仅修正地址字段并另存目录，数据、模型、损失、学习率和严格断言不变。'
        '此前CPU解码上下文差异及旧F4异常的原因仍未知，不能因本次通过就称已解决。','',
        f"实际CPU进程用时{result['seconds']:.3f}秒，峰值RSS{result['peak_process_RSS_GiB']:.3f}GiB；"
        '编码图像RAM上限64MiB。24个原图文件与已完成IR111完整拟合的内容SHA匹配，'
        '六份TrainGT校验；不保存权重、图像或FPN缓存，没有新GPU工作。','',
        '独立断点循环的六项人工检查已通过，但本次只执行真实训练步，'
        '没有完成该循环的真实全数据交付或恢复。正式GPU检查、研究控制器、'
        '匹配完整Train、完整Dev冻结、全部正式对象完整七段Test／47字段／原276列CSV'
        '及独立成本仍需完成。原两组训练、已登记队列和来源保持不变。'
        '历史Test暴露、原RGB预训练见过Dev、伪IR由RGB派生的限制不变；Goal active。']
    outputs[1].write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='EXPORTED_ALL12_ACTUAL_T4_CPU_CHECKS_NO_FORMAL_GAIN',
        conditions=len(summaries),seconds=result['seconds'],peak_process_RSS_GiB=result['peak_process_RSS_GiB'])),flush=True)


if __name__=='__main__':main()
