#!/usr/bin/env python3
"""Publish actual runtime-component CPU evidence, not formal method cost/gain."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.native_pyramid_experiment import read
from tools.neck72_runtime_study import exited
from tools.run_pseudo_ir_comparison import sha256


def main():
    config=ROOT/'configs/perceptual_task_utility_runtime_components_20261125.json';spec=read(config)
    base=Path(spec['output_directory']);result=read(base/'complete.json');execution_path=base.with_suffix('.execution.json')
    execution=read(execution_path)
    assert execution['exit_code']==0 and not execution['RSS_limit_exceeded']
    assert result['identity']==execution['identity'] and exited(execution['pid'],identity=execution['identity'])
    assert execution['log_sha256']==sha256(execution['log'])
    assert result['specification_sha256']==execution['specification_sha256']==sha256(config)
    assert result['source_hashes']==spec['source_hashes'] and all(sha256(ROOT/p)==s for p,s in spec['source_hashes'].items())
    assert result['status']=='COMPLETE_TASK_UTILITY_ACTUAL_TRAIN_B4_CPU_RUNTIME_COMPONENT_NOT_GPU_COST'
    assert result['actual_unique_Train_frames']==8 and result['actual_NN_batches']==6 and result['actual_NN_frame_visits']==24
    assert result['actual_optimizer_updates']==0 and result['IR_bytes_read_or_decoded']==0
    assert result['independent_legacy_preprocessing_and_lookup_inputs_bit_exact']
    assert result['independent_raw_B4outputs_bit_exact_in_all_three_passes'] and result['independent_NMS_packets_and_literal_MOT_exact']
    assert result['original_RGB_weights_BN_gradients_unchanged'] and result['hooks_removed']
    assert result['peak_process_RSS_GiB']<=8 and result['actual_device']=='cpu'
    assert not any(result[k] for k in ['new_GPU_work','formal_GPU_eligibility','full_workflow_registered',
        'historical_GPU_B4_packet_equivalence_established','TaskUtility_fitted_snapshot_loaded','new_Dev_Test_accuracy',
        'new_image_FPN_disk_cache','saved_weights','goal_met','formal_method_FPS_reported'])
    unit=read(spec['pytest_receipt']);assert unit['exit_code']==0 and unit['all_sources_unchanged']
    assert sha256(unit['log'])==unit['log_sha256'] and sha256(spec['pytest_receipt'])==spec['pytest_receipt_sha256']
    bad=read(spec['preserved_first_failed_unit_receipt']);assert bad['exit_code']==1 and bad['all_sources_unchanged']
    assert sha256(bad['log'])==bad['log_sha256']==spec['preserved_first_failed_unit_log_sha256']
    assert sha256(spec['preserved_first_failed_unit_source'])==bad['source_hashes']['tests/test_task_utility_runtime.py']
    stem=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_RUNTIME_COMPONENTS'
    copies=[(base/'complete.json',stem.with_suffix('.json')),
        (Path(execution['log']),Path(str(stem)+'_CPU_LOG.txt')),
        (Path(spec['pytest_receipt']),Path(str(stem)+'_PYTEST.json')),
        (Path(unit['log']),Path(str(stem)+'_PYTEST.txt')),
        (Path(spec['preserved_first_failed_unit_source']),Path(str(stem)+'_R1_TEST_SOURCE.txt')),
        (Path(bad['log']),Path(str(stem)+'_R1_TEST_LOG.txt'))]
    for source,destination in copies:
        assert not destination.exists() and source.stat().st_size<2**20;destination.write_bytes(source.read_bytes())
    failure=Path(str(stem)+'_R1_TEST_FAILURE.json');assert not failure.exists()
    failure.write_text(json.dumps(dict(status='PRESERVED_FIRST_ARTIFICIAL_RUNTIME_TEST_FAILURE',actual_execution=bad,
        original_test_source_sha256=sha256(spec['preserved_first_failed_unit_source']),
        explanation='Artificial fakeYOLOXreturnedonly1decodedanchor,triggering original NMSscalar-mask shape behavior. '
            'Real800x1440YOLOXreturns23625anchors. Corrected the testfixture to2anchors with1eligible detection; '
            'no legacyNMS/model/accuracy/source-policy changed. Added independent full7video file-binding checks. '
            'Firstfailedreceipt/source/log retained,not relabeled successful.',formal_GPU_eligibility=False,new_accuracy=False),indent=2)+'\n')
    summary=Path(str(stem)+'_EXECUTION.json');assert not summary.exists()
    summary.write_text(json.dumps(dict(actual_execution=execution,actual_execution_sha256=sha256(execution_path),
        public_result_sha256=sha256(stem.with_suffix('.json')),exporter_sha256=sha256(__file__),
        first_artificial_test_failure_preserved=True,formal_method_cost=False,new_accuracy=False,
        goal_met=False,supported_innovations=0),indent=2)+'\n')
    text=('# 任务效用候选：独立测速组件与真实CPU路径\n\n'
        '新增单条件测速入口、来源／前缀核对与B1/B4独立计时组件，17项人工检查通过。'
        '逐项验证真实批量大小、预热状态、单模型限制、全部七段原文件绑定、输入／输出变化即停和实际延迟统计。'
        '首次人工fixture仅一个原始锚点，触发旧NMS的标量mask形状错误；改为两个锚点但一个候选框，'
        '原NMS、模型及真实23625锚点路径不变。失败源码、日志和退出码完整保留。\n\n'
        '原研究写26项同B1路径，但其中历史原RGB明确来自B4，不能与新鲜B1或检测包重放混同。'
        '在正式TaskUtility拟合前另行冻结执行澄清：25项真实B1路径加1项真实四观测B4原RGB参照；'
        '26项准确率配方、主方法、预算与Dev选择规则不变。每项都先独立核对，再计时七段各32帧，'
        '前8帧预热、后24帧计时，共168帧；B4记录42个批次延迟，不能把摊销每图成本称作单帧延迟。\n\n'
        '正式入口必须已登记完整流程、前序真正导出退出、完整Test来源和原Dev冻结后才能调用CUDA。'
        'B4另用原FP16／原数值策略核对原B4检测包和逐视频MOT；不能替换成B1、四份当前帧复制或放宽一致性。'
        '历史四worker loader与全Test吞吐不冒充本次同步前缀成本。读取／解码／所需模态字节hash、'
        '预处理＋NN＋NMS＋回主存、ByteTrack的墙时分开记录，包含少量clock／同步开销；'
        '来源核对、输出摘要、资源核对在计时外。验证可能预热OS页缓存，不声称冷盘或性能方差。\n\n'
        '实际CPU检查只使用Train BT-003第1–8帧、一个域：独立原float64归一化再量化输入，与lookup输入逐值精确；'
        '冻结RGB模型实际完成六次B4前向、24次帧访问，参考／验证／计时三次路径的原始head、NMS包与字面MOT均一致。'
        '原RGB权重／BN／梯度不变，前向诊断hook移除，无优化更新、IR字节读取、TaskUtility快照加载或新Dev/Test。'
        f"实际CPU总用时{result['seconds']:.3f}秒，峰值RSS{result['peak_process_RSS_GiB']:.3f}GiB。"
        'CPU为FP32恢复的原FP16权重，并含输入／输出hash hook，八帧／四帧预热仅为工程范围，'
        '不能作为原FP16 GPU/B4一致性、正式方法FPS、融合贡献或完整研究执行资格。\n\n'
        '实际测速CLI预检因完整TaskUtility登记缺失在加载模型或调用CUDA前停止；F4前序仍未释放。'
        '新的正式GPU工作为0，完整控制器、总结果导出及集成检查仍待补齐，原CPU数值异常原因也未解决。'
        '本轮不增加独立创新计数；主结构方案13项完整Test继续在4／5卡执行。'
        '固定新版IR、seed0、最多两卡、64GiB工作盘、40GiB独立空闲限制保持；代码与轻量结果入Git，Goal active。\n')
    path=stem.with_suffix('.md');assert not path.exists();path.write_text(text)
    print(json.dumps(dict(status='EXPORTED_TASK_UTILITY_RUNTIME_COMPONENTS_CPU_NOT_FORMAL_COST',
        seconds=result['seconds'],peak_process_RSS_GiB=result['peak_process_RSS_GiB'],new_GPU_work=False,new_accuracy=False)),flush=True)


if __name__=='__main__':main()
