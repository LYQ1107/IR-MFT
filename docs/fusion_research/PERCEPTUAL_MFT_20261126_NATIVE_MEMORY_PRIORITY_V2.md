# 记忆流程V2：修正汇总行核验，继续等待当前完整Test

当前伪红外主模型、RGB基线、固定PNG、三权重0／0.05／0.1和seed0均保持。原始RGB完整Test HOTA29.528964，最低目标34.528964、进一步39.528964。此修订没有产生新准确率，也不增加创新计数。

预执行复核发现V1写错了一项结果布局判断：官方CSV解析结果与原Dev报告的per_sequence47都包含BT-005、SN-013和COMBINED三行，V1却删掉解析结果最后一行后再与原三行比较。若不修正，会在正式Dev评估已完成后错误拒绝合法结果。

旧CPU等待控制器在任何GPU工程／Dev／Test任务前已按实际身份停止，实际子进程SIGTERM退出-15、包装工具退出1。没有重启当前Test，没有失败的神经实验被删除。旧源码`c9ace54dc04fa7b9592ae1f238e372f7bb4f6bf73525d6c7bb4ee28b5bc2e47c`、锁定登记和退场原件保留。V2另用新的入口、工作／日志目录及登记，避免改写旧版本历史。

实际原始276列Dev CSV回归核对141个独立字段值，完整三行比较通过；故意删去COMBINED或改动视频HOTA的记录夹具均被拒绝。原记录中的错误截断比较也被实际重现为False。此检查是已完成旧Dev数据的格式回归，不是新模型评估。12个涉及加载、来源、GPU释放、工程、生产及官方评估的函数AST与V1完全相同，三个正常条件、视觉对象和科学配置也完全相同；原四个不同Train图像的CPU神经检查按字节复用，没有额外CNN调用。

V2已锁定430项来源，当前CPU等待控制器PID2458665、启动ticks97465484及原始命令均验证。正式CUDA入口在当前主Test仍活跃时再次正确拒绝，CUDA未初始化、GPU子任务0。必须等当前完整Test实际结束后，才按既有计划在4/5卡各做12帧GPU检查并执行完整Dev。没有增加RGB网络结构、随机种子或阈值搜索，也没有修改原广范围F4的完成条件。

另已准备并实际启动只读Test交付审计，PID2466611、启动ticks97505123，CUDA_VISIBLE_DEVICES为空。实际预检确认当前原控制器仍在，全部七段／三条件尚未齐备时没有创建结果发布目录。该审计要等原控制器和所有原worker实际退出0，核对三组各八行的47字段、21份MOT原件、规范RGB的原始完整CSV、冻结检查点及冻结时间早于Test启动，之后才发布。1128字段是即将执行的检查范围，不冒称当前已通过。它读取同一官方CSV，不重算TrackEval或运行神经模型，不改变选择。

截至2026-11-26T15:50:00.985785+00:00当前七段Test仍在运行，尚无完整本轮Test结论。新审计与V2均不占用GPU；4/5两卡仍归原始baseline优先Test所有。权重、图像、MOT和特征银行不推送Git。历史Test已暴露、RGB预训练见Dev、伪IR来自RGB及单seed0限制保持。

[完整修订记录](PERCEPTUAL_MFT_20261126_NATIVE_MEMORY_PRIORITY_V2.json)、[原件与登记](perceptual_20261126_native_memory_priority_v2/pipeline_registered.json)、[实际CSV回归](perceptual_20261126_native_memory_priority_v2/real_original_Dev_CSV_layout_regression_r1.json)、[V2执行器](../../tools/run_mft_native_memory_priority_v2.py)、[Test只读审计代码](../../tools/audit_mft_original_baseline_priority_test.py)。原[真实Train CPU图像检查](PERCEPTUAL_MFT_20261126_NATIVE_MEMORY_PRIORITY_CPU_READY.json)继续有效，其旧等待身份仅为当时快照；当前运行以本页为准。
