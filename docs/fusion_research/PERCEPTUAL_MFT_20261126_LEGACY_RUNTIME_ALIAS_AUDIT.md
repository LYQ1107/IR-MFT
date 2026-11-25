# 初版IR同检查点的历史运行成本引用

初版九项完整Test中的LB_restored_cycle_IR_reference_raw与旧Full_cycle_IR_e06_raw为同一检查点、同一full硬循环模型及raw策略。两段Dev各32帧的四个检测数组逐值一致，原模型metadata、校准／源RGB／结构配置及11项共同推理源码均一致；这一个历史profile可作为该别名的成本参考。没有执行新的GPU或NN测速。

| 历史实测输入模式 | B4 FPS | 独立分配显存GiB | 预留显存GiB |
| --- | ---: | ---: | ---: |
| 含读图 | 2.808830 | 1.030273 | 1.414062 |
| 预读图像RAM | 4.555138 | 1.030273 | 1.414062 |

基干参数98995378，适配器参数145600，总计99140978；检查点SHA为`6bfc537c63f6aec71c9a76b33ce1f7e4a81c65ba30bebf1be15a758a3d2f7c52`。每模式为两段Dev、每段8帧预热＋24帧计时，共48计时帧；B4吞吐不能作为B1延迟或整段Test FPS。含输入准备、传输、CNN、NMS和ByteTrack，排除IR生成、模型加载、写盘及gate观察器；内存模式另排除读图／预读。共享负载和文件缓存未隔离。

此次只核对这两份旧NoCycle成本档案与九项正式别名的严格连接。其余八项在这两份档案中没有建立相同权重／cue／变体／轮次／策略的连接；这不等于整个仓库没有任何相关基线测速。不能仅因检查点相同而把正常IR成本赋给错位／缺失IR，也不能把no_cycle或第6轮的成本直接赋给full／第24轮。源RGB两个控制也没有用null检查点键自动匹配。

| 正式初版Test条件 | 本次历史档案连接 |
| --- | --- |
| LB_legacy_matched_RGB_raw | 在这两份档案中未严格匹配 |
| LB_legacy_matched_gray_priority | 在这两份档案中未严格匹配 |
| LB_legacy_no_centering_paired6_raw | 在这两份档案中未严格匹配 |
| LB_legacy_no_centering_Dev_best_raw | 在这两份档案中未严格匹配 |
| LB_restored_cycle_IR_reference_raw | 同检查点历史参考：Full_cycle_IR_e06_raw |
| LB_IR_image_roll_raw | 在这两份档案中未严格匹配 |
| LB_IR_unavailable_raw | 在这两份档案中未严格匹配 |
| LB_source_RGB_fresh | 在这两份档案中未严格匹配 |
| LB_source_RGB_canonical | 在这两份档案中未严格匹配 |

旧调度器记录两种成本条件完成，但没有单独保存旧benchmark worker的数值退出码；这里保留null，不回填0。本次CPU核对实际退出0，约2.335秒、峰值RSS0.037GiB。检测数组和旧源码的核对是已有产物审计，不是新的64帧NN调用，也不构成初版九项成本全覆盖。

[完整来源及原值](PERCEPTUAL_MFT_20261126_LEGACY_RUNTIME_ALIAS_AUDIT.json)、[实际CPU调用](PERCEPTUAL_MFT_20261126_LEGACY_RUNTIME_ALIAS_AUDIT_EXECUTION.json)、[CPU原日志](PERCEPTUAL_MFT_20261126_LEGACY_RUNTIME_ALIAS_AUDIT_CPU_LOG.txt)。核对代码见[只读审计工具](../../tools/audit_mft_legacy_runtime_alias.py)。旧profile文件及原九项47字段／MOT／CSV均未改写。

这份历史参考属于145600参数的初版结构模型，不能赋给459171参数的原生quality_ir或最新全量S/C/R家族。本次没有新Dev／Test准确率，不改变主方法、训练、创新计数或Goal状态。物理4／5卡上的原时序灰度／IR训练继续，目标34.529及2–3项独立贡献仍未达。
