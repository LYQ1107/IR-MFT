# 任务效用候选：配置、worker与模型加载组件

新增固定来源研究配置、独立训练worker、只读诊断和CPU张量加载组件。
10项worker人工检查及4项人工文件／张量检查通过；一个真实IR联合监督
条件完成三次四帧更新和12帧B1路径。完整记录见
[WORKER_CPU_ENGINEERING](PERCEPTUAL_MFT_20261125_TASK_UTILITY_WORKER_CPU_ENGINEERING.md)。
没有正式全量拟合、GPU资格或新Dev/Test；独立创新计数仍0。

研究配置保持四种监督组合×三种cue，共12个正常条件，同一553705参数
推理结构，正常条件仅切换前景／效用训练监督。计算标识包含两个监督
开关、权重和温度、原RGB、图像内容参考、划分、学习率和完整交付计划。
复制配置不共享可变字典，不能把不同监督条件误当成同一计算。

拟采用原完整27251帧Train、自然时长的6814个T≤4片段，至少6遍，按
完整Dev曲线共同决定9／12遍延长；固定seed0、12遍学习率周期，源RGB
encoder／head冻结。每个已执行epoch的12条正常Dev曲线以及四个IR的
错位／缺失完整Dev均保留，选择只用Dev，全26项Test配方不筛去负对象。
这些仍是待实际运行的计划，不是已完成的12组新拟合或26项新Test。

worker在CUDA前要求完整登记、所有原队列（含F4）完整导出及控制器／
子进程实际退出，再做12个真实GPU Train＋B1检查后才能正式拟合。
F4若因无合格父模型结束，可以释放计算资源，但不据此声称F4已实验或
被证据排除。当前真实preflight记录为完整登记缺失、前序未就绪；CPU
结果写在独立目录，不生成正式GPU资格文件。

新断点封装必须包含task_utility_v1、八项损失、八项计数、精确完整
Train地址证明和来源元数据；旧循环、监督不符、漏帧或异常值拒绝。
CPU加载器核对原路径、文件SHA、原登记／arm／历史／监督来源，再严格
加载全部CPU FP32辅助张量。前景／效用／两者门控去除保留所有参数与
权重；固定坐标、输入干预不能与门控去除静默混合。实际正式TaskUtility
快照尚未生成，加载器只在标明人工的文件上验证，不能当作真实拟合证明。

真实逐帧部分仅接受当前RGB／IR与标识，无GT或未来帧接口；RGB、零修正、
学习后缺失三路与独立直接RGB的完整head、NMS、MOT、局部ID计数及KF
状态逐值一致。实际IR原图roll改变输入、原生特征及12帧head／检测包；
坐标固定结果匹配独立组合再NMS。仅12个Train帧，不能解释成准确率、
完整序列、强泛化、CPU／GPU数值普遍稳定或融合记忆已转移。

人工worker检查见[WORKER_PYTEST](PERCEPTUAL_MFT_20261125_TASK_UTILITY_WORKER_PYTEST.json)，
加载检查见[LOADING_PYTEST](PERCEPTUAL_MFT_20261125_TASK_UTILITY_LOADING_PYTEST.json)。
worker检查还覆盖原前序失败、身份伪造、控制器／导出／任一子进程未退出
均不释放，以及完整登记缺失时不进入GPU设置。人工历史中的27251等值
只检验规则，不是实际交付证据；真实交付仍要由正式训练循环实际产生。

后续仍缺生产／官方评估、独立计时、完整调度和导出，以及所有真实GPU
资格与正式Train/Dev/Test。本轮没有重启、替换或修改旧健康训练与队列，
来源与空间审计另见[WORKER_SOURCE_AUDIT](PERCEPTUAL_MFT_20261125_TASK_UTILITY_WORKER_SOURCE_AUDIT.json)。
Goal active；固定新版PNG、历史Test暴露、RGB预训练见过Dev及先验工作
边界均保持，不按实现数量或工程通过增加创新计数。
