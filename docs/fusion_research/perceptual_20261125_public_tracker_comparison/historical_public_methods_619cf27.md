# Historical-test public-method audit

评估集包含 `7` 条序列、`15985` 帧；使用 TrackEval `12c8791b303e0a0b50f753af204249e622d0281a` 和 protocol_v2 派生 GT。完整 47 个字段及逐序列结果在 JSON/TSV 中。

| 方法 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | CLR_FP | CLR_FN | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical_official | 31.091 | 49.575 | 19.619 | 35.044 | 52.325 | 560 | 31038 | 37489 | 2560 |
| bytetrack_v3_exact | 30.907 | 49.580 | 19.371 | 34.684 | 52.289 | 568 | 31034 | 37537 | 2558 |
| bot_sort_c0_rgb_seed0 | 30.310 | 49.730 | 18.584 | 32.156 | 52.824 | 618 | 30413 | 37333 | 2530 |
| hybrid_sort_c2_rgb_seed0 | 30.677 | 50.030 | 18.920 | 31.419 | 51.757 | 813 | 34084 | 35013 | 3342 |

方法范围：原始 BYTETracker、冻结历史轨迹、BoT-SORT c0、HybridSORT c2。两个外部关联器共用同一 V3 proposal 和同一 RGB 鱼类特征缓存；没有声称复现其原生 person-ReID 权重。

历史测试划分过去曾参与调参，因此表格用于跨序列冻结审计，不作为新的盲测结论。
