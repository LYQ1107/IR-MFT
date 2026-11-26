# 最低六轮完整Dev图表入口：等待真实结果

当前新一轮完整Dev尚未结束，没有生成新精度图。新增 `tools/audit_mft_native_dev_figures.py` 只读取实际完成的最低六轮独立Dev审计、预先发布的62份配置和按字节保存的官方CSV。它不重新推理或评估，不读取Test结果，不修改模型、训练预算或选择。

结果齐全后保留三张独立SVG及本地PNG预览：

1. `all_six_normal_Dev_curves`：quality／temporal各自RGB、灰度、IR的全部36个正常检查点，并保留新鲜B1和规范B4 RGB参考线。
2. `all_IR_image_interventions`：两个IR模式每个轮次的真实IR、图像空间roll和辅助缺失，共36项IR条件；缺失结果须与新鲜RGB精确一致。
3. `matched_control_effects_by_Dev_video`：每个轮次、同模式的IR−RGB及IR−灰度，分别展示BT-005、SN-013和官方COMBINED。COMBINED直接取官方CSV，不平均视频HOTA。

所有62份官方276列CSV的47字段和逐序列值都先与独立审计逐值核对；图表输入表保留186项case×sequence的六项主要指标与配置／CSV SHA。图只显示CSV未舍入值，原统一6／9／12轮预算决定仍使用既定官方39摘要及排序。视频和检查点不作为独立重复，不画伪造的置信区间，不把同轮架构差值当单模块消融或成熟创新。

输入门禁已实际验证：`--check-ready`退出0并明确报告等待；在真实报告尚不存在时运行`--export`按预期退出1，SVG和预览目录都没有创建。短CPU调用未独立观察PID，保留未知。该检查不是完整绘图成功、原生精度、收敛或Test完成证明。

正式完整Dev审计完成后，在已有CPU环境运行：

```bash
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
  /data3/liuyeqiang/.venvs/trackocd-a100/bin/python \
  tools/audit_mft_native_dev_figures.py --export \
  --output-directory docs/fusion_research/perceptual_20261126_native_minimum6_dev_figures \
  --preview-directory /var/tmp/liuyeqiang_ir_mft_perceptual_20261122/native_minimum6_dev_figure_previews
```

若本地matplotlib不在所用Python环境中，沿用研究的CPU `PYTHONPATH`（cfp-fishmot site-packages和codex pydeps）；不需要新装GPU依赖。已有输出目录会被拒绝覆盖。Git只发布小型SVG、CSV、来源JSON及文档，PNG预览留在工作目录；不上传权重或图像数据集。

历史Test暴露、RGB预训练见Dev和伪IR由RGB生成的限制保持。当前原主方案Test HOTA29.834177，对固定RGB29.529约+0.305177；目标34.529与2–3项独立贡献仍未达，Goal继续active。

[输入门禁实测](PERCEPTUAL_MFT_20261126_NATIVE_DEV_FIGURES_INPUT_CHECK.json) · [绘图源代码](../../tools/audit_mft_native_dev_figures.py)
