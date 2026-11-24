# 任务效用候选的先验边界与后续归因

核对日期：2026-11-25，北京时间。当前已验证独立贡献仍为0，本文没有
新增训练或Dev/Test结果，也不构成正式GPU执行登记。

CLDyN（CVPR2026）已比较语义补偿前后的下游任务损失，使用原任务项
加相对初始输出变差时的hinge惩罚。故任务反馈、补偿前后比较和防止有害
融合均已有先例。[原论文3.3节、式6–7](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Customized_Fusion_A_Closed-Loop_Dynamic_Network_for_Adaptive_Multi-Task-Aware_Infrared-Visible_CVPR_2026_paper.pdf)

MetaFusion（CVPR2023）已利用检测语义与元特征嵌入促进融合／检测联合
学习，不能把检测任务指导融合本身当作原创。
[官方摘要](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_MetaFusion_Infrared_and_Visible_Image_Fusion_via_Meta-Feature_Embedding_From_CVPR_2023_paper.html)

GateFusion（WACV2026）的主动说话人检测使用分层门控注入，并加入
掩码对齐与过度正激活惩罚。因此门控残差和抑制辅助错误的泛化表述也
不足以证明原创；其任务是音视频主动说话人检测，不能称为我们的RGB／
伪IR跟踪复现。[官方摘要](https://openaccess.thecvf.com/content/WACV2026/html/Wang_GateFusion_Hierarchical_Gated_Cross-Modal_Fusion_for_Active_Speaker_Detection_WACV_2026_paper.html)

本轮CVF直接PDF抓取均返回403；CLDyN式6–7来自官方PDF可索引的3.3节
正文，另两项仅核对官方摘要。没有声称通读全部论文、完整重现、确认
等价或完成穷尽检索；没有复制第三方代码、下载权重或补造本任务指标。
之前GAFF／TarDAL的边界仍保留于CPU原型报告，不能因本次新增检索覆盖。

本仓库候选的具体计算是：只在Train用同一冻结RGB SimOTA分配，比较
原RGB与停止梯度的固定0.5门控提案在相同锚点上的原YOLOX损失；包含
几何、目标性和分类，局部差值产生停止梯度的软目标与权重。前景分支
只使用Train框中心占据标签；推理只看当前RGB／辅助图像与RGB分数。
标签权重不代表标注置信度；零效应标签权重为0；效用logit不是校准
概率或因果效应。这些是实现事实，不证明上述具体组合已经原创。

后续要检验的区别是“固定分配的局部教师能否识别当前伪IR何处真正
有用”，不能只展示梯度非零、门控变化、框移动或训练损失下降。
正式比较仍需相同完整Train预算的四种监督组合×RGB／灰度／IR，保持
相同553705参数推理结构，比较无监督、仅前景、仅效用、联合监督。
各模型只由完整Dev选择快照，主方法在Test前冻结。

前景与效用各自的条件增量，以及联合相对单项的增量都需报告。还需
比较这些增量在IR、重复RGB及灰度上的差别；若IR没有超过公平强对照，
就不能把通用容量或监督收益解释成伪IR互补。强对照的选择同样按Dev
冻结，不按Test挑选；使用未四舍五入的官方指标，不用表面小差值确立贡献。
正确IR需结合真实图像错位／缺失及去门控等对照判断；副作用和负结果
完整保留，所有正式工程通过候选仍按原要求完成七段Test及独立成本。

这些是有待满足的归因要求。新候选正式控制器尚未完成，不能把本说明、
CPU原型或接口检查当作正式拟合、四方向验证完成或2–3项创新。
固定新版PNG、原Protocol V2、seed0、最多物理4／5两卡及存储上限均保持。
