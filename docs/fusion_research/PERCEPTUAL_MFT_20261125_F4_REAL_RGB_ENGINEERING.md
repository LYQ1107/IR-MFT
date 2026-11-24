# F4：真实RGB P3进入非零权重记忆

普通RGB记忆对照的数据路径已在真实Train特征上检查：五个独立连续片段、
共20帧，268个ROI向量。权重0.05与0.1各有47次可靠记忆更新、34次正外观
惩罚；零权重的MOT字节、每帧tracked/lost/removed及Kalman数值状态均与
同输入的原ByteTrack严格一致。**两个非零权重的MOT输出在这些短片段上
也仍与原ByteTrack相同，没有显示跟踪收益。**

[预登记](../../configs/perceptual_f4_real_rgb_engineering_20261125.json)
先于本次执行。样本由BT-003原有192帧Train缓存中最早五段不重叠、自然
连续四帧确定，没有读取GT或按输出选择。原始帧为1–4、6–9、11–14、
20–23、25–28。每段显式重置为独立序列和局部1–4帧，不跳过记忆序列内部
帧，也不把它们冒充20帧连续视频或完整Dev。

## 实际执行的数据路径

使用原RGB检测器产生的实际P3/P4/P5缓存，P3为320通道；用原冻结YOLOX
head在CPU FP32、480×864执行。indexed NMS与原postprocess逐帧精确一致，
固定conf0.1／NMS0.7，经实际缩放比例恢复原图框。当前RGB P3由捕获接口
交给3×3 aligned ROIAlign，然后进入权重0／0.05／0.1的三个tracker。

没有使用独立裁剪编码器或描述向量替代P3，也没有运行新的图像编码。
每帧ROI输出后释放所持P3引用；检测包不改，实际记忆向量有限且为单位
向量，原head参数、BN及梯度未变。普通RGB＋固定权重1对照的辅助、融合、
质量和可用性字段置为不可读取对象；实际运行没有访问这些字段。

缓存完整SHA256、RGB检测器SHA256及20张当前RGB原始字节摘要均核对；
后者匹配已完成全量IR拟合的输入清单。只加载原RGB checkpoint；GT、
targets.npy与Test数据路径设读取拒绝保护，实际尝试计数为0。

[完整CPU记录](PERCEPTUAL_MFT_20261125_F4_REAL_RGB_CPU_PROOF.json)
由[probe_f4_real_rgb.py](../../tools/probe_f4_real_rgb.py)实际执行，退出0；
CPU用时25.211秒、峰值RSS0.968GiB。这个时间包含源码／缓存摘要检查、
权重加载及共享三个tracker的检查，不是单方法系统FPS。

## 尚未覆盖

这补齐真实普通RGB非零记忆控制的工程证据，不是学习到的融合特征已经
转移或F4有效性的证据。没有加载融合checkpoint、读取真实IR、访问Dev或
Test、更新参数、占用GPU或产生新官方指标。800×1440实际在线编码与
GPU ROI、完整序列、学习融合父模型资格和正式流程仍需分别完成。
当前独立创新贡献0，整体Goal保持active。

复现：固定CPU环境运行 `tools/probe_f4_real_rgb.py`；已有结果拒绝覆盖。
