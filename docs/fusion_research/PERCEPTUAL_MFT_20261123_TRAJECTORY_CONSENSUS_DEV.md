# 完整开发集：多历史锚点与竞争保护

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All24 fullDev cases complete after all6 zero-module controls reproduce all39 metrics, both track files and detailedCSV exactly. New-IR observation plus RGB point heads averages50.314 over3 association seeds, above its ByteTrack parent49.771; IR point heads49.885 trail the same-backend RGB heads and strongest gray plain control49.965. Detector adapter remains seed0. Classical multi-anchor/mutual gating is an integration repair, not standalone multimodal novelty. No new Test.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TR_gray_ir_s0_full | 48.917 | 68.210 | 35.123 | 56.524 | 89.376 | 100 | 0 | 0 |
| TR_gray_ir_s0_zero | 45.267 | 67.955 | 30.196 | 51.607 | 89.105 | 122 | 0 | 0 |
| TR_gray_ir_s1_full | 50.073 | 68.173 | 36.823 | 58.918 | 89.402 | 97 | 0 | 0 |
| TR_gray_ir_s2_full | 50.074 | 68.174 | 36.823 | 58.920 | 89.398 | 97 | 0 | 0 |
| TR_gray_rgb_s0_full | 48.219 | 68.166 | 34.157 | 56.471 | 89.382 | 100 | 0 | 0 |
| TR_gray_rgb_s0_zero | 46.198 | 68.041 | 31.402 | 52.968 | 89.156 | 122 | 0 | 0 |
| TR_gray_rgb_s1_full | 48.780 | 68.099 | 34.980 | 57.787 | 89.380 | 99 | 0 | 0 |
| TR_gray_rgb_s2_full | 50.074 | 68.174 | 36.823 | 58.920 | 89.398 | 97 | 0 | 0 |
| TR_ir_ir_s0_full | 49.334 | 68.377 | 35.628 | 59.693 | 89.860 | 89 | 0 | 0 |
| TR_ir_ir_s0_zero | 45.781 | 68.350 | 30.690 | 53.464 | 89.807 | 117 | 0 | 0 |
| TR_ir_ir_s1_full | 50.010 | 68.450 | 36.570 | 60.174 | 89.873 | 88 | 0 | 0 |
| TR_ir_ir_s2_full | 50.312 | 68.455 | 37.007 | 60.532 | 89.886 | 85 | 0 | 0 |
| TR_ir_rgb_s0_full | 50.312 | 68.453 | 37.008 | 60.532 | 89.884 | 86 | 0 | 0 |
| TR_ir_rgb_s0_zero | 50.482 | 68.361 | 37.312 | 60.349 | 89.865 | 102 | 0 | 0 |
| TR_ir_rgb_s1_full | 50.319 | 68.458 | 37.016 | 60.540 | 89.901 | 86 | 0 | 0 |
| TR_ir_rgb_s2_full | 50.312 | 68.455 | 37.007 | 60.532 | 89.888 | 86 | 0 | 0 |
| TR_rgb_ir_s0_full | 49.964 | 68.632 | 36.401 | 58.259 | 90.116 | 92 | 0 | 0 |
| TR_rgb_ir_s0_zero | 46.827 | 68.588 | 32.008 | 53.571 | 89.964 | 114 | 0 | 0 |
| TR_rgb_ir_s1_full | 50.018 | 68.620 | 36.487 | 58.255 | 90.104 | 90 | 0 | 0 |
| TR_rgb_ir_s2_full | 49.963 | 68.628 | 36.401 | 58.258 | 90.104 | 93 | 0 | 0 |
| TR_rgb_rgb_s0_full | 50.018 | 68.617 | 36.487 | 58.246 | 90.063 | 93 | 0 | 0 |
| TR_rgb_rgb_s0_zero | 49.787 | 68.591 | 36.177 | 59.191 | 89.964 | 107 | 0 | 0 |
| TR_rgb_rgb_s1_full | 49.391 | 68.608 | 35.587 | 57.830 | 90.095 | 93 | 0 | 0 |
| TR_rgb_rgb_s2_full | 49.660 | 68.444 | 36.059 | 57.940 | 90.119 | 93 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.
