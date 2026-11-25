# 当前冻结检测输出上的公开跟踪方法对照

SORT、OC-SORT 各四组输出，均先完成两段4834帧Dev，再冻结相同参数完成七段15985帧Test；16个新CPU worker均实际退出0。ByteTrack四组是先前已发布结果，未重新训练或替换。

| 跟踪器 | 冻结检测输出 | Test HOTA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bytetrack | source_RGB_fresh | 29.528964 | 18.967559 | 33.072407 | 50.897096 | 450 |
| bytetrack | matched_rgb | 29.467182 | 18.864496 | 32.337002 | 50.877774 | 456 |
| bytetrack | matched_gray | 29.421963 | 18.917890 | 32.165328 | 51.120680 | 454 |
| bytetrack | primary_IR | 29.834177 | 19.367217 | 33.058532 | 51.041322 | 459 |
| sort | source_RGB_fresh | 22.795803 | 11.167695 | 22.838042 | 52.772717 | 1081 |
| sort | matched_rgb | 22.779991 | 11.142440 | 22.841148 | 52.780308 | 1094 |
| sort | matched_gray | 21.966831 | 10.435600 | 21.812721 | 52.595368 | 1098 |
| sort | primary_IR | 22.581972 | 10.979934 | 22.594224 | 52.540162 | 1100 |
| ocsort | source_RGB_fresh | 27.708995 | 16.508536 | 30.306473 | 52.263443 | 631 |
| ocsort | matched_rgb | 27.634146 | 16.404966 | 30.106026 | 52.253092 | 648 |
| ocsort | matched_gray | 26.274320 | 14.976681 | 28.893293 | 52.075743 | 638 |
| ocsort | primary_IR | 27.723771 | 16.589769 | 30.230437 | 52.052280 | 638 |

每个跟踪器内，固定IR相对同预算RGB的HOTA差值：

- bytetrack：+0.366995点。
- sort：-0.198019点。
- ocsort：+0.089625点。

原Dev选定主方法仍为ByteTrack＋IR，HOTA 29.834177；本表不按Test切换跟踪器或主模型，目标34.529及2–3项独立贡献尚未达成。

两种参考实现均从官方固定提交下载，保留原字节与许可通知，未复制进Git；[配置](../../configs/audit_mft_public_trackers_20261125.json)记录URL、SHA和全部参数。SORT使用max_age=1、OC-SORT使用max_age=30，均min_hits=3、IoU=0.3；输入置信阈值0.6，最小框面积100。没有外观ReID或测试集参数搜索。

第一次工程检查过严，未接受原始缓存已有的anchor_indices/rgb_reference_scores；修正为与原重放器一致仅读取四项检测字段。第一次正式CPU调用生成两段完整轨迹后，因SORT选择TkAgg而在TrackEval绘图阶段退出1；保留全部失败资产，改用Agg后，这两段重跑轨迹与失败调用逐字节相同。第二次CPU父调度在首个worker成功退出0后，因日志字段重名退出1；修复日志后，该完整Dev官方CSV逐字节相同。两个父进程的实际退出与原始worker日志均保留。没有更改跟踪算法、检测输出或种子。

原始276列CSV、逐序列及COMBINED的47字段、实际进程/退出码、来源SHA和CPU计时见[完整JSON](PERCEPTUAL_MFT_20261125_PUBLIC_TRACKERS.json)、[47字段CSV](PERCEPTUAL_MFT_20261125_PUBLIC_TRACKERS.csv)。新原始CSV位于[本次目录](perceptual_20261125_public_tracker_comparison/)。CPU计时只覆盖缓存跟踪；不包含检测器、融合网络或伪IR生成，不能作为端到端FPS。

执行代码：[CPU完整Dev/Test驱动](../../tools/audit_mft_public_trackers.py)、[独立结果核对及导出](../../tools/audit_mft_public_tracker_results.py)。JSON保存每个实际worker的完整argv、PID、启动tick、退出码、日志SHA和结果SHA；调用固定CUDA_VISIBLE_DEVICES为空、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=2、MKL_NUM_THREADS=2。其中1份启动瞬间的proc argv为空，保留其真实空字节SHA，不把它称为期望argv核验通过。进程启动命令、实际worker成功退出及独立活跃进程抽查另行保留。

旧分支已有BoT-SORT c0 HOTA30.310、HybridSORT c2 HOTA30.677，它们使用旧V3检测输出及鱼类RGB ReID，输入与本轮不同。[原文快照](perceptual_20261125_public_tracker_comparison/historical_public_methods_619cf27.md)固定于619cf27，不能当成本轮相同检测输入的对照。

历史Test暴露、原RGB预训练见Dev、伪IR由RGB派生的限制保持。本次补齐公开方法比较，不增加已验证创新点计数。
