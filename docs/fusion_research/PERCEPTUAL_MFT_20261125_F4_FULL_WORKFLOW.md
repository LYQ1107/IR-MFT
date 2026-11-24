# F4完整流程登记与实际CPU等待

本轮完成F4调度、现有组件的完整CPU集成和实际CPU控制器启动。没有新的
HOTA、GPU执行或学习IR融合特征转移，独立贡献仍为0，Goal继续active。

入口为`tools/launch_fusion_memory.py`。先核验前序空间研究完整导出、原
控制器及全部子进程实际退出，再从两家族完整Dev文件确定视觉资格；没有
合格父模型时仅导出资格证据，不能称F4已经评估或被排除。有合格父模型时
执行真实GPU工程检查、完整正常Dev、正常选择冻结、附加Dev干预、最终
Dev冻结、完整Test、各配置独立成本及最终导出。选择只读取Dev。

GPU工程检查还要求实际CUDA上的已知坐标平面ROI池化结果精确为[24,12]；
CPU结果不能替代CUDA。所有正常配置及主IR缺失／错位配置仍须经过独立
检测器、原ByteTrack零路径、当前ROI包和保留记忆的实际核验。此前CPU
检测摘要异常原因仍未知，后续成功的CPU组件核对不等于该异常已解决。

95项检查实际执行通过，耗时21.07秒。其中完整169任务图使用人工数据、
人工执行器和完整长度人工Dev文件，验证阶段次序、冻结和拒绝路径；另有
真实标准库CPU子进程，检查最多两任务、实际身份／退出、失败时保留健康
兄弟任务、停止新任务和资源等待。现有评估、工作流、工程比较器和测速
组件检查同时执行。它们均不能证明真实GPU、模型资格或准确率。

此前92.194秒的真实RGB八帧CPU比较器，只为本轮仍精确不变的六个实际
执行核心提供证据；旧GPU入口未执行的版本不被声明为当前验证。原始GPU
组件登记字节另存`perceptual_f4_gpu_engineering_components_20261125_initial.json`，
当前设备坐标检查变化由新的完整登记锁定。

实际CPU预检在普通CPU模块导入后给主入口设置张量／CUDA与文件读取陷阱，
确认没有触发torch.load、CUDA、图像／张量／CSV／GT读取；记录15个不同
普通文件路径。五个完整入口齐全、登记存在；六个父模型完成标记仍缺，
资格未知，正式放行false。

完整登记锁定384项来源和实际CPU证明／前序依赖；CPU控制器PID488189、
start_ticks82242612，命令摘要见原始记录。审计状态为
`WAITING_FOR_COMPLETE_SPATIAL_EXPORT_AND_ACTUAL_EXITS`，phases为空。
它没有抢占前序GPU或新建拟合；物理4／5最多两卡、seed0、工作盘64GiB、
独立空闲40GiB、主机余量40GiB及新增worker RSS8GiB限制继续执行。
不追加图像／FPN／ROI盘缓存；编译缓存限制在工作盘内。

真实Popen句柄、退出码、启动ticks和命令摘要决定任务完成；观察超时不会
重启任务。失败产物保留，已运行的健康兄弟任务可以结束；失败后不继续
启动待办。所有工程通过的正式配置保留完整Test、47字段、原276列CSV和
独立168计时帧，不因负Dev删除。最多78个最终条件属于同一家族的控制与
权重组合，不是78项创新。

原两组IR任务仍为PID158984／159108，审计时均完成45/72宏循环，正在46；
其他四组完整三遍Train已完成。九份旧登记所有来源和依赖精确，空间323项
来源／11项依赖和测速137动态来源集合精确。工作文件58.236595GiB，独立
空闲2737.282GiB；未删除原缓存、未重启训练。

现有完整七段Test最高观测31.504，对原RGB29.529为+1.975，但对应RGB
31.440，IR差值仅+0.064；最高观测没有被晋升为按Test选择的主方法。
旧Dev固定结构主方法Test29.142，相对原RGB为−0.387。质量、空间、因果
时序与轨迹记忆的独立贡献仍待完整验证，目标34.529及2–3项贡献未达到。
历史Test暴露、原RGB预训练见过当前Dev、伪IR由RGB派生等限制保持公开。

原始轻量证据：[CPU集成证明](PERCEPTUAL_MFT_20261125_F4_FULL_WORKFLOW_CPU_PROOF.json)、
[实际pytest记录](PERCEPTUAL_MFT_20261125_F4_FULL_WORKFLOW_PYTEST.json)、
[pytest原日志](PERCEPTUAL_MFT_20261125_F4_FULL_WORKFLOW_PYTEST.txt)、
[完整登记](PERCEPTUAL_MFT_20261125_F4_FULL_WORKFLOW_REGISTERED.json)、
[CPU预检](PERCEPTUAL_MFT_20261125_F4_FULL_WORKFLOW_PREFLIGHT.json)、
[控制器启动记录](PERCEPTUAL_MFT_20261125_F4_FULL_WORKFLOW_CONTROLLER.json)、
[来源与资源审计](PERCEPTUAL_MFT_20261125_F4_FULL_WORKFLOW_SOURCE_AUDIT.json)。
