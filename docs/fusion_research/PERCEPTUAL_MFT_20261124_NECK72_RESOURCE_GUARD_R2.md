# neck72内存准入修订2

修正尚未进入任何实际阶段的neck72等待队列，避免在子进程中再次预留已由同伴占用的内存。原先每个子进程按4×8GiB预留，要求MemAvailable至少72GiB；完整控制器实际最多运行两个生产者，且CPU/GPU阶段有屏障，原估算会拒绝仍能保留40GiB的运行条件。

现在控制器按两个8GiB生产者检查，每个新子进程再核对自身的8GiB。MemAvailable已经扣除实际驻留进程的内存；40GiB余量、单进程8GiB上界、64GiB工作目录上限和40GiB独立磁盘空闲均保留。

| 检查位置 | 预留的新增内存 | 最低MemAvailable |
| --- | ---: | ---: |
| 完整控制器 | 2×8GiB | 56GiB |
| 裁剪核验、GPU数据/拟合、银行编译子进程 | 1×8GiB | 48GiB |

只停止了经启动时间、命令摘要及无子进程核实的旧CPU等待控制器4041967。当时状态为WAITING_FOR_COMPLETE_CGR_AND_NATIVE50_PROCESSES，phases为空，没有开始neck72 GPU或正式拟合。旧状态、登记、CPU证明、日志和全部修改前源码均归档；原有两卡IR训练和其他等待队列继续运行。公开的初版证明与登记未被覆盖。

科学设计文件`configs/perceptual_neck72_study_20261124.json`保持原SHA。其资源字段中的4个生产者仍是原登记上界，实际实现收紧为2个；拟合预算、8个匹配头、输入、划分、seed0、原图、选点、主方法、所有Dev/Test角色及干预均未修改。原设计所指初版worker登记SHA和修订2实际worker登记SHA在新的完整登记中分别保留。

修订2的worker检查10项通过，包括56/48GiB边界及低于40GiB余量拒绝；8份既有完整Dev视图×3模式的数组逐元素一致，4幅实际Train图像的CPU前向及冻结状态通过。完整流程检查另10项通过，包含控制器确实按两个生产者预留的回归检查；8组既有seed0模型32帧前缀的轨迹、非空查询后验及只读统计结果一致。没有增加GPU、随机种子、正式拟合或新的准确率。

当前worker30个、完整流程94个源码/配置重新锁定。此前完整Train102个、输入恢复4个、原生金字塔102个、CGR87个及native50的61个来源版本均未修改。native50已完成；恢复后的CPU等待控制器仍须等待CGR及其前置全部导出和实际进程退出。

对应修订2的WORKER_CPU_PROOF、WORKER_REGISTRATION、STUDY_CPU_PROOF、STUDY_REGISTRATION及RESOURCE_GUARD_R2.json保留完整来源和新控制器身份。完整系统FPS仍需实现并实测，+5 HOTA和2–3项独立贡献仍未证明，整个Goal保持未完成。
