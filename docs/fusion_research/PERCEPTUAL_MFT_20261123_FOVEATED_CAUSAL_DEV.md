# Complete causal background and cross-modal attention Dev controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All70 registered full-Dev controls: separate IR novelty, common RGB novelty and cross-modal consensus, identical detector/crop counts/scales/fusion and both global frontends. Ordinary MOG2/zoom/refinement alone is not novelty; compare strongest matched RGB and retain failures. No new Test.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| X_baseline_new | 46.236 | 68.766 | 31.137 | 53.119 | 90.224 | 108 | 0 | 0 |
| X_baseline_rgb | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 | 0 | 0 |
| X_new_ir_consensus_half_k2_nms | 31.162 | 60.568 | 16.111 | 30.578 | 75.640 | 819 | 0 | 0 |
| X_new_ir_consensus_half_k2_refine | 36.289 | 64.279 | 20.569 | 39.418 | 83.928 | 288 | 0 | 0 |
| X_new_ir_consensus_half_k4_nms | 26.278 | 57.948 | 11.988 | 24.658 | 70.201 | 1038 | 0 | 0 |
| X_new_ir_consensus_half_k4_refine | 33.025 | 62.006 | 17.676 | 34.893 | 81.535 | 361 | 0 | 0 |
| X_new_ir_consensus_quarter_k2_nms | 40.465 | 67.275 | 24.406 | 44.870 | 87.781 | 144 | 0 | 0 |
| X_new_ir_consensus_quarter_k2_refine | 40.582 | 66.496 | 24.831 | 45.403 | 86.178 | 145 | 0 | 0 |
| X_new_ir_consensus_quarter_k4_nms | 40.163 | 65.954 | 24.525 | 44.225 | 85.605 | 156 | 0 | 0 |
| X_new_ir_consensus_quarter_k4_refine | 39.965 | 64.925 | 24.666 | 44.251 | 83.244 | 210 | 0 | 0 |
| X_new_ir_physical_half_k2_nms | 28.563 | 60.675 | 13.524 | 27.593 | 75.392 | 812 | 0 | 0 |
| X_new_ir_physical_half_k2_refine | 39.669 | 64.117 | 24.619 | 41.043 | 84.092 | 293 | 0 | 0 |
| X_new_ir_physical_half_k4_nms | 27.630 | 57.918 | 13.257 | 25.304 | 70.494 | 961 | 0 | 0 |
| X_new_ir_physical_half_k4_refine | 34.036 | 62.140 | 18.745 | 33.593 | 81.384 | 370 | 0 | 0 |
| X_new_ir_physical_quarter_k2_nms | 42.647 | 67.185 | 27.135 | 47.766 | 87.430 | 147 | 0 | 0 |
| X_new_ir_physical_quarter_k2_refine | 40.779 | 66.392 | 25.115 | 46.321 | 85.614 | 173 | 0 | 0 |
| X_new_ir_physical_quarter_k4_nms | 38.757 | 65.880 | 22.877 | 40.958 | 85.095 | 165 | 0 | 0 |
| X_new_ir_physical_quarter_k4_refine | 38.181 | 64.740 | 22.591 | 41.305 | 82.077 | 238 | 0 | 0 |
| X_new_ir_self_half_k2_nms | 31.492 | 60.478 | 16.467 | 31.468 | 76.634 | 773 | 0 | 0 |
| X_new_ir_self_half_k2_refine | 37.171 | 64.443 | 21.525 | 39.998 | 84.546 | 311 | 0 | 0 |
| X_new_ir_self_half_k4_nms | 27.979 | 57.958 | 13.582 | 26.584 | 70.444 | 1031 | 0 | 0 |
| X_new_ir_self_half_k4_refine | 33.519 | 62.120 | 18.170 | 35.368 | 82.127 | 338 | 0 | 0 |
| X_new_ir_self_quarter_k2_nms | 44.307 | 67.472 | 29.144 | 49.204 | 88.358 | 119 | 0 | 0 |
| X_new_ir_self_quarter_k2_refine | 40.930 | 66.410 | 25.280 | 46.327 | 87.026 | 159 | 0 | 0 |
| X_new_ir_self_quarter_k4_nms | 41.930 | 66.698 | 26.421 | 45.862 | 86.630 | 139 | 0 | 0 |
| X_new_ir_self_quarter_k4_refine | 38.960 | 65.378 | 23.276 | 42.868 | 84.068 | 249 | 0 | 0 |
| X_new_rgb_half_k2_nms | 29.243 | 60.186 | 14.282 | 27.116 | 75.214 | 777 | 0 | 0 |
| X_new_rgb_half_k2_refine | 39.154 | 63.906 | 24.061 | 41.749 | 84.352 | 283 | 0 | 0 |
| X_new_rgb_half_k4_nms | 28.154 | 57.591 | 13.831 | 25.704 | 70.134 | 924 | 0 | 0 |
| X_new_rgb_half_k4_refine | 36.366 | 62.213 | 21.350 | 39.153 | 81.856 | 337 | 0 | 0 |
| X_new_rgb_quarter_k2_nms | 41.859 | 67.265 | 26.105 | 46.276 | 87.454 | 143 | 0 | 0 |
| X_new_rgb_quarter_k2_refine | 41.731 | 66.407 | 26.281 | 47.417 | 85.485 | 184 | 0 | 0 |
| X_new_rgb_quarter_k4_nms | 40.147 | 65.848 | 24.543 | 43.225 | 85.211 | 166 | 0 | 0 |
| X_new_rgb_quarter_k4_refine | 39.252 | 64.686 | 23.877 | 43.125 | 82.164 | 253 | 0 | 0 |
| X_new_uniform_k4_nms | 33.024 | 60.831 | 18.020 | 33.111 | 76.856 | 667 | 0 | 0 |
| X_new_uniform_k4_refine | 35.442 | 63.398 | 19.915 | 37.493 | 83.578 | 277 | 0 | 0 |
| X_rgb_ir_consensus_half_k2_nms | 33.148 | 61.080 | 18.047 | 33.779 | 77.056 | 604 | 0 | 0 |
| X_rgb_ir_consensus_half_k2_refine | 37.664 | 63.898 | 22.263 | 41.374 | 84.038 | 276 | 0 | 0 |
| X_rgb_ir_consensus_half_k4_nms | 31.176 | 58.705 | 16.620 | 28.919 | 71.981 | 793 | 0 | 0 |
| X_rgb_ir_consensus_half_k4_refine | 35.492 | 61.853 | 20.434 | 37.630 | 81.161 | 368 | 0 | 0 |
| X_rgb_ir_consensus_quarter_k2_nms | 41.034 | 67.350 | 25.054 | 44.457 | 87.801 | 129 | 0 | 0 |
| X_rgb_ir_consensus_quarter_k2_refine | 41.071 | 66.565 | 25.395 | 45.406 | 85.659 | 155 | 0 | 0 |
| X_rgb_ir_consensus_quarter_k4_nms | 39.014 | 66.140 | 23.072 | 41.223 | 85.678 | 150 | 0 | 0 |
| X_rgb_ir_consensus_quarter_k4_refine | 39.719 | 65.059 | 24.301 | 43.094 | 82.807 | 191 | 0 | 0 |
| X_rgb_ir_physical_half_k2_nms | 31.262 | 61.079 | 16.066 | 29.168 | 76.862 | 627 | 0 | 0 |
| X_rgb_ir_physical_half_k2_refine | 36.054 | 64.018 | 20.385 | 36.916 | 83.763 | 305 | 0 | 0 |
| X_rgb_ir_physical_half_k4_nms | 28.090 | 58.624 | 13.527 | 26.955 | 71.804 | 814 | 0 | 0 |
| X_rgb_ir_physical_half_k4_refine | 34.660 | 61.842 | 19.491 | 36.507 | 81.350 | 374 | 0 | 0 |
| X_rgb_ir_physical_quarter_k2_nms | 45.171 | 67.309 | 30.358 | 51.860 | 87.779 | 124 | 0 | 0 |
| X_rgb_ir_physical_quarter_k2_refine | 43.579 | 66.512 | 28.606 | 50.308 | 85.414 | 150 | 0 | 0 |
| X_rgb_ir_physical_quarter_k4_nms | 39.973 | 65.877 | 24.309 | 43.430 | 85.519 | 138 | 0 | 0 |
| X_rgb_ir_physical_quarter_k4_refine | 40.317 | 64.665 | 25.192 | 44.774 | 82.110 | 216 | 0 | 0 |
| X_rgb_ir_self_half_k2_nms | 30.069 | 61.557 | 14.755 | 29.268 | 77.818 | 638 | 0 | 0 |
| X_rgb_ir_self_half_k2_refine | 36.866 | 63.866 | 21.362 | 39.239 | 84.191 | 272 | 0 | 0 |
| X_rgb_ir_self_half_k4_nms | 27.411 | 58.924 | 12.831 | 25.461 | 72.586 | 801 | 0 | 0 |
| X_rgb_ir_self_half_k4_refine | 33.723 | 62.342 | 18.311 | 34.939 | 81.899 | 302 | 0 | 0 |
| X_rgb_ir_self_quarter_k2_nms | 43.299 | 67.687 | 27.741 | 48.089 | 88.270 | 119 | 0 | 0 |
| X_rgb_ir_self_quarter_k2_refine | 41.076 | 67.003 | 25.231 | 45.524 | 86.578 | 154 | 0 | 0 |
| X_rgb_ir_self_quarter_k4_nms | 40.982 | 66.656 | 25.249 | 44.564 | 86.507 | 139 | 0 | 0 |
| X_rgb_ir_self_quarter_k4_refine | 40.720 | 65.463 | 25.374 | 45.362 | 83.569 | 256 | 0 | 0 |
| X_rgb_rgb_half_k2_nms | 30.701 | 61.308 | 15.447 | 29.821 | 77.196 | 602 | 0 | 0 |
| X_rgb_rgb_half_k2_refine | 36.136 | 63.675 | 20.561 | 36.827 | 83.894 | 285 | 0 | 0 |
| X_rgb_rgb_half_k4_nms | 30.640 | 58.648 | 16.063 | 29.568 | 72.002 | 776 | 0 | 0 |
| X_rgb_rgb_half_k4_refine | 32.749 | 61.617 | 17.487 | 32.871 | 81.483 | 329 | 0 | 0 |
| X_rgb_rgb_quarter_k2_nms | 42.776 | 67.412 | 27.189 | 47.245 | 87.658 | 126 | 0 | 0 |
| X_rgb_rgb_quarter_k2_refine | 42.404 | 66.397 | 27.134 | 48.307 | 85.155 | 167 | 0 | 0 |
| X_rgb_rgb_quarter_k4_nms | 40.712 | 65.995 | 25.163 | 44.110 | 85.498 | 143 | 0 | 0 |
| X_rgb_rgb_quarter_k4_refine | 39.883 | 64.563 | 24.685 | 44.364 | 81.922 | 244 | 0 | 0 |
| X_rgb_uniform_k4_nms | 35.002 | 61.014 | 20.146 | 36.695 | 77.230 | 521 | 0 | 0 |
| X_rgb_uniform_k4_refine | 38.195 | 63.693 | 22.982 | 42.895 | 83.767 | 291 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.
