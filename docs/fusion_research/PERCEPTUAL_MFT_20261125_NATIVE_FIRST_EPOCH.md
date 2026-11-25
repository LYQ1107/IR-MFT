# 原生质量融合：两个正式对照的首遍Train核对通过

原生质量模式quality_rgb／quality_gray已各实际完成一遍全部27251帧、6814个片段，没有padding重复帧；各6814次实际优化更新。两组交付地址摘要相同，源RGB权重／BN的逐遍检查保持不变。此后原进程继续第二遍，没有停止、重启或复制新训练进程。最低六遍共同预算仍在执行，其他四个正常条件按原登记队列接续。

| 条件 | 物理GPU | 第一遍实际更新 | 第一遍完整loss均值 | 原生模块参数已变化 | 检查点SHA256 |
| --- | ---: | ---: | ---: | --- | --- |
| quality_rgb | 4 | 6814 | 2.57960204 | True | 98ab4c5af00e3319b4d364cb70a7a2e2941bbacaa0b21325d917d9327d4c7057 |
| quality_gray | 5 | 6814 | 2.57888883 | True | 032327cce9ddd7c77f1b927d92e229c0f44d54a9e6de3e39c30ab768c13d4d2f |

实际源顺序为BT-003、SN-001、PF-001、BT-001、MSK-002、SN-015；每段完整帧数依次243、684、15000、3000、1754、6570，总27251。独立重新计算完整日程，并核对每段帧数、顺序摘要和无重复标记；6814个片段保留短尾，未补帧。快照的元数据、全部历史、完成遍数及SHA与原日志一致；原生张量均为有限值，module哈希与各自初始化不同。没有用loss或参数变化代替Dev/Test准确率。

两组是匹配的RGB与灰度容量对照，正式输入没有使用IR；目前不能声称IR完成一遍训练或有新的多模态收益。它们的来源绑定由实际执行的PinnedTrainImageReader在每次读图时核对SHA，再由完整交付日程保障覆盖。引用的是此前四个成功IR全量训练一致的原始图像清单，预期集合在这两组中仅包含27251个RGB条目。当前原生epoch日志不另存完整逐图消费列表，本核对没有拿参考清单冒充新的实际交付清单，也没有回读图像重造历史。

GPU进程的PID、启动tick和argv哈希与此前正式启动记录相同。独立核对只在CPU读取已保存原生小快照和文本，未加载完整RGB模型或做神经前向／优化；源RGB按引用保留，权重未推Git。工具复用了原verify_fit(stage=1)，只确认已完成前缀，不改变原through_epoch=6授权或完整Dev／Test规则。

实际CPU核对退出0，完整调用5.642秒、观测峰值RSS0.581GiB，PID 1001690，argv已实测匹配。两份原progress字节快照见[首遍原始日志](perceptual_20261125_native_first_epoch)，完整核对与checkpoint摘要见[JSON](PERCEPTUAL_MFT_20261125_NATIVE_FIRST_EPOCH.json)。这些快照是当时的完成前缀，实际训练仍继续推进。

本机复核命令（使用已配置的研究环境与新的输出目录）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
MFT26IR_ROOT=/data3/liuyeqiang/datasets/mft26-ir \
PYTHONPATH=/data3/liuyeqiang/IR-MFT:/data3/liuyeqiang/IR-MFT/exps:/data3/liuyeqiang/.codex/pydeps:/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages:/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT \
  /data3/liuyeqiang/.venvs/trackocd-a100/bin/python /data3/liuyeqiang/IR-MFT/tools/audit_mft_native_epoch_prefix.py \
  --fit-id quality_rgb --fit-id quality_gray --through-epoch 1 \
  --output-directory /var/tmp/ir_mft_native_epoch1_reproduced
```

原登记、静态快照、当前同一训练进程与资源余量的再次核对见[来源记录](PERCEPTUAL_MFT_20261125_NATIVE_FIRST_EPOCH_SOURCE_AUDIT.json)。尚无原生候选的新Dev/Test HOTA，完整六组训练及后续预算判断、消融、完整七段Test和成本仍待完成。原冻结S+C+R主方法仍29.834177，目标34.529及2–3项独立贡献未达。独立创新计数0，Goal保持active。
