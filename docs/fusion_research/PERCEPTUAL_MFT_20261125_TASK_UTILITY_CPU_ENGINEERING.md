# 任务相关空间监督：真实CPU工程检查

12个条件全部完成，共六个Train视频／六帧、每条件六次更新、总72次工程更新。这不是完整Train拟合、GPU资格、收敛、Dev/Test或HOTA提升。独立创新仍为0。

四种条件始终保留同一推理结构和553705参数，仅开关Train前景／效用监督。每种均有重复RGB、灰度、新版IR；相同seed0、初始化权重和样本顺序。额外门控容量相同，不能把监督效果混为新增容量。

前景标签是Train框覆盖的网格中心，不是真实分割。效用标签用同一冻结RGB的SimOTA分配，比较原RGB与固定0.5门控、已detach提案的逐锚点原YOLOX损失；包括目标性、前景分类及5倍IoU项。差值／温度0.1经sigmoid转为软目标，绝对差值决定标签权重，目标和权重均停止梯度。零效应对应零标签权重，不能用零初始化伪造正负监督。

主损失仍使用原YOLOX动态标签分配与几何回归，另保留可靠RGB项。推理只读取当前RGB和当前辅助像素，无GT、过去或未来帧；前景／效用logit加入质量logit，残差仍有RMS界和RGB保护，缺失模态返回原RGB。效用不是校准概率或因果效应。

| 条件 | 前景监督直接梯度和 | 效用监督直接梯度和 | 框最大变化 | 有用局部标签 | 有害局部标签 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_aux_labels_rgb | 0.000000 | 0.000000 | 5.104645 | 7955 | 1854 |
| no_aux_labels_gray | 0.000000 | 0.000000 | 5.638733 | 7437 | 1707 |
| no_aux_labels_ir | 0.000000 | 0.000000 | 4.107513 | 7554 | 1511 |
| foreground_labels_rgb | 0.126181 | 0.000000 | 5.101685 | 7954 | 1854 |
| foreground_labels_gray | 0.123375 | 0.000000 | 5.635986 | 7437 | 1706 |
| foreground_labels_ir | 0.121330 | 0.000000 | 4.108185 | 7555 | 1511 |
| utility_labels_rgb | 0.000000 | 0.022585 | 5.100739 | 7954 | 1854 |
| utility_labels_gray | 0.000000 | 0.021376 | 5.634369 | 7437 | 1707 |
| utility_labels_ir | 0.000000 | 0.018177 | 4.113037 | 7554 | 1511 |
| foreground_utility_labels_rgb | 0.126238 | 0.022812 | 5.097626 | 7955 | 1854 |
| foreground_utility_labels_gray | 0.123563 | 0.021448 | 5.631531 | 7437 | 1706 |
| foreground_utility_labels_ir | 0.121427 | 0.018534 | 4.114166 | 7555 | 1511 |

这些梯度及框变化证明计算路径被实际使用，不能证明框更正确、FG预测有效或跟踪收益。原始每步损失、各层标签、门控变化、72项缺失检查和四项实际IR图像roll响应均完整保存。初始零修正检查覆盖每条件首个Train帧；缺失检查覆盖全部六个工程帧，不能冒充完整序列。

R1在首次零修正处失败、更新0次，源码与日志保留在提交60d6093及R1_FAILURE记录。诊断中相同FPN和权重的CPU head也曾相对旧参考产生宽高差异；其他小诊断没有复现，根因仍未知。R2按现有生产训练器为每个优化输入新算RGB head，全部比较继续用torch.equal。这不证明CPU跨调用上下文数值已普遍稳定，也不修复或放行旧F4 CPU测速异常。

本次R2记录的缓存参考与每步新算参考差异也全部为0，没有再次复现R1。
不能把R2通过归因于某个已确定的数值问题修复；跨调用上下文差异的根因仍未知。
这不改变逐步新算参考与现有训练器一致的实施选择，也不放行正式GPU或F4资格。

这些比较使用原训练head辅助函数的raw输出，没有执行完整检测、NMS或ByteTrack，
不能替代完整在线源等价或GPU资格检查。

CPU用时465.956秒，峰值RSS1.850GiB。严格加载原RGB源，原权重／BN／梯度不变；12个实际原图文件与完成拟合的内容摘要匹配，六份TrainGT摘要核对。不保存新权重、图像或FPN缓存，原GPU训练和全部已登记队列未改。

前景指导及模态选择损失已有[GAFF先例](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.pdf)；任务驱动融合也有[TarDAL先例](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Target-Aware_Dual_Adversarial_Learning_and_a_Multi-Scenario_Multi-Modality_Benchmark_To_CVPR_2022_paper.html)。本轮只核对主站可索引正文／摘要，完整PDF抓取失败，没有声称完整重现或复制这两个项目的代码。固定分配局部效用也未证明原创性。

若后续正式登记，仍需相同完整Train预算、完整Dev冻结、全部正常／消融／错配／缺失完整七段Test、47字段／原276列CSV与独立成本。CPU工程通过不会自动启动正式拟合或增加四方向完成数。历史Test暴露、RGB预训练见过Dev、伪IR由RGB派生等限制不变；Goal继续active。
