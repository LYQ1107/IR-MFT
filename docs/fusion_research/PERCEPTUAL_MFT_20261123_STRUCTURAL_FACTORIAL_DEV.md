# Complete32 missing S/C/R single/static component cases

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Four distinct seed0 fits,145600parameters/24passes/6912updates each; source17/prefix37/full37 checks precede all accuracy. Reuse49completed reference rows; this is not81newruns. Raw6/component effects and all equally selected curves retained; no Testselection.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FCT_ir_000_s0_e06 | 47.452 | 68.910 | 32.709 | 54.403 | 90.129 | 87 | 0 | 0 |
| FCT_ir_000_s0_e06_raw | 46.298 | 68.635 | 31.276 | 51.236 | 90.235 | 100 | 0 | 0 |
| FCT_ir_000_s0_e12 | 46.917 | 68.913 | 31.974 | 54.265 | 90.164 | 89 | 0 | 0 |
| FCT_ir_000_s0_e12_raw | 46.847 | 68.469 | 32.100 | 53.870 | 90.058 | 107 | 0 | 0 |
| FCT_ir_000_s0_e18 | 47.302 | 68.930 | 32.492 | 53.705 | 90.233 | 89 | 0 | 0 |
| FCT_ir_000_s0_e18_raw | 48.140 | 68.641 | 33.794 | 54.756 | 90.134 | 104 | 0 | 0 |
| FCT_ir_000_s0_e24 | 46.055 | 68.891 | 30.826 | 52.484 | 90.218 | 91 | 0 | 0 |
| FCT_ir_000_s0_e24_raw | 45.882 | 68.440 | 30.804 | 51.594 | 89.843 | 107 | 0 | 0 |
| FCT_ir_001_s0_e06 | 47.734 | 68.963 | 33.070 | 54.708 | 90.166 | 84 | 0 | 0 |
| FCT_ir_001_s0_e06_raw | 45.468 | 68.893 | 30.048 | 51.089 | 90.196 | 91 | 0 | 0 |
| FCT_ir_001_s0_e12 | 46.224 | 68.803 | 31.091 | 53.370 | 90.000 | 91 | 0 | 0 |
| FCT_ir_001_s0_e12_raw | 48.263 | 68.896 | 33.837 | 54.734 | 90.183 | 95 | 0 | 0 |
| FCT_ir_001_s0_e18 | 46.310 | 68.862 | 31.182 | 53.398 | 90.052 | 90 | 0 | 0 |
| FCT_ir_001_s0_e18_raw | 48.642 | 68.890 | 34.374 | 56.030 | 90.211 | 90 | 0 | 0 |
| FCT_ir_001_s0_e24 | 46.310 | 68.863 | 31.182 | 53.398 | 90.052 | 92 | 0 | 0 |
| FCT_ir_001_s0_e24_raw | 47.793 | 68.835 | 33.211 | 55.339 | 90.134 | 93 | 0 | 0 |
| FCT_ir_010_s0_e06 | 47.643 | 68.800 | 33.027 | 55.129 | 90.200 | 92 | 0 | 0 |
| FCT_ir_010_s0_e06_raw | 46.756 | 68.710 | 31.855 | 52.809 | 90.114 | 93 | 0 | 0 |
| FCT_ir_010_s0_e12 | 46.451 | 68.784 | 31.406 | 52.831 | 90.170 | 90 | 0 | 0 |
| FCT_ir_010_s0_e12_raw | 44.118 | 67.988 | 28.665 | 50.552 | 90.134 | 108 | 0 | 0 |
| FCT_ir_010_s0_e18 | 46.370 | 68.730 | 31.323 | 52.387 | 90.050 | 93 | 0 | 0 |
| FCT_ir_010_s0_e18_raw | 46.174 | 68.244 | 31.278 | 52.359 | 89.949 | 106 | 0 | 0 |
| FCT_ir_010_s0_e24 | 46.361 | 68.696 | 31.326 | 52.369 | 90.030 | 102 | 0 | 0 |
| FCT_ir_010_s0_e24_raw | 46.322 | 68.677 | 31.280 | 52.344 | 89.856 | 100 | 0 | 0 |
| FCT_ir_100_s0_e06 | 46.671 | 68.873 | 31.660 | 53.962 | 90.095 | 90 | 0 | 0 |
| FCT_ir_100_s0_e06_raw | 46.913 | 68.674 | 32.082 | 52.107 | 90.050 | 98 | 0 | 0 |
| FCT_ir_100_s0_e12 | 46.908 | 68.885 | 31.975 | 54.253 | 90.116 | 90 | 0 | 0 |
| FCT_ir_100_s0_e12_raw | 45.408 | 68.542 | 30.123 | 51.486 | 90.043 | 101 | 0 | 0 |
| FCT_ir_100_s0_e18 | 46.999 | 68.892 | 32.098 | 54.248 | 90.112 | 92 | 0 | 0 |
| FCT_ir_100_s0_e18_raw | 44.650 | 68.557 | 29.122 | 50.581 | 90.033 | 106 | 0 | 0 |
| FCT_ir_100_s0_e24 | 47.382 | 68.908 | 32.614 | 53.658 | 90.114 | 91 | 0 | 0 |
| FCT_ir_100_s0_e24_raw | 46.033 | 68.554 | 30.951 | 52.248 | 90.067 | 108 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.
