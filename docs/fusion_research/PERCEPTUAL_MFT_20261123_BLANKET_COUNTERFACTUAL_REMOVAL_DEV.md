# 完整开发集：去掉错帧前景有效性约束

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All8 fullDev policies after unchanged24E fitting. Keep recoverability curriculum; remove only the shared-foreground validity exclusion from wrong-auxiliary consistency. Compare strongest complete and removed curves, not a hand-picked epoch. No new Test.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CGR_ir_blanket_counterfactual_s0_e06 | 46.965 | 68.965 | 32.016 | 54.690 | 90.168 | 91 | 0 | 0 |
| CGR_ir_blanket_counterfactual_s0_e06_raw | 45.931 | 68.698 | 30.744 | 53.100 | 89.987 | 98 | 0 | 0 |
| CGR_ir_blanket_counterfactual_s0_e12 | 46.126 | 68.787 | 30.968 | 53.030 | 90.078 | 96 | 0 | 0 |
| CGR_ir_blanket_counterfactual_s0_e12_raw | 45.536 | 68.767 | 30.189 | 51.622 | 89.989 | 104 | 0 | 0 |
| CGR_ir_blanket_counterfactual_s0_e18 | 46.231 | 68.812 | 31.098 | 53.359 | 90.110 | 91 | 0 | 0 |
| CGR_ir_blanket_counterfactual_s0_e18_raw | 48.487 | 68.855 | 34.172 | 55.373 | 90.108 | 104 | 0 | 0 |
| CGR_ir_blanket_counterfactual_s0_e24 | 46.317 | 68.715 | 31.256 | 53.371 | 90.108 | 92 | 0 | 0 |
| CGR_ir_blanket_counterfactual_s0_e24_raw | 46.156 | 68.752 | 31.022 | 52.870 | 90.078 | 108 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.
