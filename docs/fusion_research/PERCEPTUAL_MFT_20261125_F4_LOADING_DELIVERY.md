# F4：严格模型加载与序列源图像交付

模型加载和序列观测交付两个组件已实现，20项张量检查与8项交付检查通过。
真实入口因六个父模型完成标记缺失而在读取融合张量前返回等待；单独的
原RGB加载及八帧Train RGB交付检查均退出0。没有新的HOTA、已合格的IR
父模型或独立创新，完整F4执行流程仍未完成。

## 模型加载

[加载器](../../tools/fusion_memory_model_loading.py)实际入口先运行原完整
Dev文件验证，并要求已有IR主模型胜过全部冻结强对照／干预；只接收
原计划准确视觉对象。CPU快照按同一文件描述符在反序列化前后校验字节，
再检查当前路径没有被替换；完整张量key、shape、dtype、有限性、快照
metadata与原progress、所选epoch全部history均须对应。模块保持CPU、eval、
无梯度，不同正常／roll／缺失路由使用独立模块实例，不按分数换模型。

[20项检查](PERCEPTUAL_MFT_20261125_F4_MODEL_LOADING_PYTEST.txt)使用临时
未训练模块及人工历史，覆盖质量／时序／空间三类真实代码、源文件
变化、内部历史与架构不符、缺少／多余／错形状／错类型／非有限张量、
RNG保持、整数buffer及实例隔离。这不是实际完整Train模型或Dev资格。

[真实CPU记录](PERCEPTUAL_MFT_20261125_F4_MODEL_LOADING_CPU_PROOF.json)中，
实际父模型入口在任何张量加载前返回六个标记待齐，资格未知。之后仅
单独加载原始compactFP16 RGB源，完整98995378个参数按原语义转CPU FP32，
整数buffer保留；没有图像、标签或新推理。耗时6.578秒、峰值RSS1.547GiB，
全部拒绝尝试计数为0。尚未加载任何真实融合checkpoint，也没有CUDA。

## 完整声明范围的源图像交付

[交付组件](../../tools/fusion_memory_observation.py)按固定Protocol V2序列
及帧数执行；Dev/Test不允许截短。按顺序读取当前RGB与固定November IR
字节，IR只在视觉路径需要时解码。未使用的IR字节仍参与来源摘要，
不参与RGB候选、质量或关联。完整图像交付摘要须匹配固定源锚点，才能
产生完成记录；保留前32帧源／检测包／MOT摘要供后续独立成本检查。

失败或字节变化会留下已写MOT前缀和失败记录，不会生成完整记录，也
不会回改先前轨迹。组件不授权GPU、模型选择或官方评估；完整生产入口
仍须绑定登记、实际父模型与GPU释放条件。

[8项检查](PERCEPTUAL_MFT_20261125_F4_SEQUENCE_DELIVERY_PYTEST.txt)使用临时
人工图像和tiny教师，包括Dev/Test截短拒绝、顺序／重复、IR实际解码条件、
未使用IR字节变化、失败前缀以及拒绝覆盖。

[真实交付检查](PERCEPTUAL_MFT_20261125_F4_SEQUENCE_DELIVERY_CPU_PROOF.json)
固定BT-003 Train前八帧，原序列243帧，记录明确标为工程前缀。严格加载
原RGB后，480×864 B1在线编码，零／0.05／0.1三个配置的完整检测包及MOT
均精确匹配上轮独立参照，原RGB参数／BN／梯度未变。8张RGB解码，8个
固定IR文件只校验原字节，IR解码或算法消费为0；没有IR融合模型或标签。
CPU40.596秒、峰值RSS1.545GiB，非独立系统FPS。未新增准确率或正收益；
上轮这三个短片段输出本就相同，本次不把重复核对算作效果实验。

两份预登记分别见[加载](../../configs/perceptual_f4_model_loading_20261125.json)
与[交付](../../configs/perceptual_f4_sequence_delivery_20261125.json)。检查后
临时人工张量／图像夹具已清除；工程MOT、权重或图像不进入Git。

## 当前状态

[来源审计](PERCEPTUAL_MFT_20261125_F4_LOADING_DELIVERY_SOURCE_AUDIT.json)
确认九份既有登记、空间323来源／11依赖及测速137动态来源均保持一致。
IR101／110原进程在4／5卡，各到23/72宏循环、step128/288；24个宏循环
才是一遍全部27251帧。空间CPU等待器身份及空阶段不变。工作58.235GiB、
独立盘空闲2745.621GiB；没有新GPU、NN拟合、图像或FPN盘缓存。

实际视觉父模型资格／融合张量加载尚待原研究完成，正式GPU生产入口、
完整Dev/Test、47项官方值、独立成本和调度仍待接通。没有把本组件的
完成视为方向完成。历史Test暴露与原RGB预训练见过当前Dev的限制不变，
已验证独立多模态贡献0、目标34.529未达，Goal保持active。

复现：CPU环境分别运行`tests/test_fusion_memory_model_loading.py`与
`tests/test_fusion_memory_observation.py`，保存源码摘要测试记录，分别传给
`tools/probe_f4_model_loading.py`和`tools/probe_f4_sequence_delivery.py`的
`--pytest-receipt`。已有实际工程输出拒绝覆盖。
