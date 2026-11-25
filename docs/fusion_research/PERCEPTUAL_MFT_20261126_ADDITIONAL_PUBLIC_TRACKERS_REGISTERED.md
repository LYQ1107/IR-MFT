# 追加BoT-SORT／HybridSORT：同一冻结检测输入的CPU比较登记

2026-11-26（北京时间）。正式CPU控制器已启动，先完成8项完整Dev，再冻结同一参数执行全部8项完整七段Test；本登记包不发布新准确率或创新点。原GPU4／5训练继续，同一Dev选定的ByteTrack主方案保持，HOTA29.834177与目标34.529均未改变。

两种关联器均复用已发布SORT／OC-SORT比较使用的四路完整检测输出：源RGB、同预算RGB、灰度和新版伪IR。每路沿用原完整Dev选定的同一快照与priority／raw策略，包坐标已还原至原图。不重新运行检测器、融合网络或ReID，不新增训练／种子，也不按新Test改选方案。Dev覆盖2段4834帧，Test覆盖全部7段15985帧，全部正式条件包括负Dev均保留，逐项导出官方47字段及原276列CSV。

| 方法 | 固定实现与预设 | 保留组件 |
| --- | --- | --- |
| BoT-SORT无ReID | 官方2519854提交demo关联参数；高／低阈值0.6／0.1、新轨迹0.7、match0.8、buffer30 | 原ORB相机运动补偿、原Kalman、分阶段关联；逐帧读取真实RGB，不跳过空检测帧 |
| HybridSORT无ReID | 官方396f8d3提交MOT17 ablation关联预设；阈值0.6、IoU0.25、inertia0.05、max_age30 | 原HMIoU、TCM两阶段、低分关联与原分数Kalman预测 |

BoT-SORT使用与原ByteTrack输入审计一致的25fps参数；统一MFT输出面积>100，不应用person宽高比过滤。HybridSORT其余参数来自原utils/args.py默认值。这是固定MFT适配的无ReID比较，不声称复现其person-ReID完整论文结果。已有旧V3＋鱼类RGB ReID成绩继续单独保留，不混为本次同输入结果。

23项官方参考源码／预设／说明文件绑定原提交及逐文件SHA，源码文件未编辑。BoT-SORT未使用的FastReID导入以拒绝实例化的哨兵隔离，with_reid固定False；这不替换其关联或ORB实现。NumPy旧np.float名称仅映射到原Python float类型。

HybridSORT在当前NumPy环境中的单元素分数返回布局需要兼容：原predict执行全部状态和协方差更新，适配层只把Kalman分数转为标量、simple分数转为单元素数组，使原update使用既有分支。克隆状态的逐值核对证明位置、KF状态、协方差及两种分数相同；逐次还断言两种分数均仅含一个值。该运行时布局适配完整公开，不把它称作新关联公式或创新点。

所有8项接口检查均实际通过，包括每路真实Dev前8帧、空帧交付与跨视频计数重置；工程调用及登记调用均实际退出0。前三次失败分别为缺lap、缺cython_bbox和上述分数布局，均在正式评测前发生，原代码、配置、执行和日志按字节归档。所有跟踪参数、输入包、参考源码、seed0和输出过滤保持一致。

依赖采用独立CPU目录：lap0.5.14轮子SHA经PyPI核对；cython-bbox0.1.5原sdist在本地编译并核对导入／数值接口。实际NumPy1.24.3，扩展也使用该版本头文件；编译退出0。两个安装模块合计约2.60MiB，不修改训练环境，不复制网络权重、图像或FPN，不把二进制依赖推送Git。安装元数据、源码包SHA和编译日志保留。

完整原登记包含大量逐视频重复模型metadata，本包发布去重摘要及其原完整741925字节JSON的SHA；原件在工作目录保持不可变。源码／输入包／生产记录SHA、四份模型来源和全部逐段帧数继续保留，已发布原ByteTrack／SORT／OC-SORT结果不变。

CPU计时将区分tracker.update与完整缓存重放：前者包含BoT-SORT的ORB，后者另含真实RGB读图／解码，两者均不含检测／融合／IR生成，不能作为端到端FPS；共享主机负载未隔离。单CPU worker，GPU仅原4／5卡，seed0，工作目录64GiB及空闲40GiB约束保持。

- [登记摘要](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_REGISTERED.json)
- [完整工程检查](perceptual_20261126_additional_public_tracker_registration/ENGINEERING.json)
- [实际工程执行](perceptual_20261126_additional_public_tracker_registration/ENGINEERING_EXECUTION.json)
- [第一次失败](perceptual_20261126_additional_public_tracker_registration/R1_engineering_log.txt)
- [第二次失败](perceptual_20261126_additional_public_tracker_registration/R2_engineering_log.txt)
- [第三次失败](perceptual_20261126_additional_public_tracker_registration/R3_engineering_log.txt)
- [固定配置](../../configs/audit_mft_additional_public_trackers_20261126.json)
- [执行源码](../../tools/audit_mft_additional_public_trackers.py)

历史Test已暴露、RGB预训练见Dev及伪IR由RGB派生的限制继续披露。新增公开后端不增加独立多模态贡献计数，原生CNN／时序的完整Dev／Test及整个Goal仍待完成。
