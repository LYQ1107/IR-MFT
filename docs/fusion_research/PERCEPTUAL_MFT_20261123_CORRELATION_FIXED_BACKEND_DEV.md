# 完整开发集：固定检测与点关联组合

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Three frozen detector adapters times RGB/IR physical-point heads and three association seeds. Adapter seed remains 0. All group means are below their corresponding plain ByteTrack parents; no positive composition contribution is established. No new Test selection.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CB_gray_ir_s0 | 43.386 | 68.067 | 27.713 | 46.813 | 88.853 | 188 | 0 | 0 |
| CB_gray_ir_s1 | 47.083 | 68.132 | 32.589 | 54.030 | 88.758 | 175 | 0 | 0 |
| CB_gray_ir_s2 | 49.254 | 68.028 | 35.715 | 56.207 | 88.743 | 180 | 0 | 0 |
| CB_gray_rgb_s0 | 43.071 | 68.081 | 27.305 | 46.673 | 88.763 | 175 | 0 | 0 |
| CB_gray_rgb_s1 | 44.988 | 68.087 | 29.776 | 49.470 | 88.784 | 170 | 0 | 0 |
| CB_gray_rgb_s2 | 46.038 | 67.879 | 31.291 | 50.590 | 88.692 | 177 | 0 | 0 |
| CB_ir_ir_s0 | 45.248 | 67.898 | 30.182 | 49.487 | 89.460 | 161 | 0 | 0 |
| CB_ir_ir_s1 | 46.928 | 68.079 | 32.390 | 53.708 | 89.378 | 143 | 0 | 0 |
| CB_ir_ir_s2 | 49.196 | 68.357 | 35.442 | 55.385 | 89.438 | 141 | 0 | 0 |
| CB_ir_rgb_s0 | 46.449 | 68.299 | 31.629 | 51.818 | 89.398 | 127 | 0 | 0 |
| CB_ir_rgb_s1 | 44.226 | 67.965 | 28.817 | 48.728 | 89.387 | 132 | 0 | 0 |
| CB_ir_rgb_s2 | 44.470 | 68.049 | 29.092 | 48.740 | 89.449 | 124 | 0 | 0 |
| CB_rgb_ir_s0 | 41.773 | 68.277 | 25.613 | 44.815 | 89.619 | 150 | 0 | 0 |
| CB_rgb_ir_s1 | 42.310 | 68.356 | 26.244 | 45.519 | 89.578 | 150 | 0 | 0 |
| CB_rgb_ir_s2 | 43.086 | 68.388 | 27.202 | 47.131 | 89.553 | 153 | 0 | 0 |
| CB_rgb_rgb_s0 | 44.180 | 68.214 | 28.682 | 48.223 | 89.574 | 142 | 0 | 0 |
| CB_rgb_rgb_s1 | 41.154 | 68.330 | 24.850 | 44.681 | 89.602 | 136 | 0 | 0 |
| CB_rgb_rgb_s2 | 43.796 | 68.223 | 28.183 | 48.560 | 89.578 | 149 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.
