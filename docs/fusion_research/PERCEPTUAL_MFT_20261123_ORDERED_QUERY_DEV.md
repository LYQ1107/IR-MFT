# RGB-first ordered identity memory Dev controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All18 controls include six complete RGB-only memory references. Generic RGB memory itself hurts some seeds; exact native-zero parity is therefore necessary but insufficient to establish a strong baseline.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O_new_ir_s0 | 49.807 | 69.123 | 35.933 | 58.024 | 90.584 | 112 | 0 | 0 | 34 |
| O_new_ir_s1 | 46.492 | 68.261 | 31.707 | 54.039 | 90.528 | 113 | 0 | 0 | 29 |
| O_new_ir_s2 | 47.761 | 68.875 | 33.167 | 55.440 | 90.512 | 111 | 0 | 0 | 26 |
| O_new_reference_s0 | 49.445 | 68.560 | 35.707 | 58.061 | 90.558 | 113 | 0 | 0 | 26 |
| O_new_reference_s1 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| O_new_reference_s2 | 47.909 | 68.303 | 33.653 | 56.558 | 90.564 | 111 | 0 | 0 | 22 |
| O_new_rgb_s0 | 49.292 | 68.891 | 35.317 | 57.415 | 90.504 | 113 | 0 | 0 | 36 |
| O_new_rgb_s1 | 46.768 | 68.238 | 32.093 | 54.694 | 90.515 | 110 | 0 | 0 | 27 |
| O_new_rgb_s2 | 47.826 | 68.331 | 33.518 | 55.788 | 90.525 | 115 | 0 | 0 | 33 |
| O_rgb_ir_s0 | 43.939 | 68.693 | 28.154 | 49.164 | 89.794 | 133 | 0 | 0 | 46 |
| O_rgb_ir_s1 | 45.483 | 68.788 | 30.117 | 50.613 | 89.936 | 117 | 0 | 0 | 41 |
| O_rgb_ir_s2 | 44.688 | 68.535 | 29.188 | 50.852 | 89.985 | 125 | 0 | 0 | 33 |
| O_rgb_reference_s0 | 45.192 | 68.326 | 29.936 | 50.955 | 89.869 | 136 | 0 | 0 | 36 |
| O_rgb_reference_s1 | 46.463 | 68.441 | 31.585 | 53.605 | 89.944 | 122 | 0 | 0 | 30 |
| O_rgb_reference_s2 | 44.684 | 68.525 | 29.187 | 50.846 | 89.994 | 129 | 0 | 0 | 31 |
| O_rgb_rgb_s0 | 43.093 | 68.505 | 27.150 | 47.395 | 89.828 | 138 | 0 | 0 | 49 |
| O_rgb_rgb_s1 | 45.724 | 68.769 | 30.448 | 52.439 | 89.912 | 123 | 0 | 0 | 43 |
| O_rgb_rgb_s2 | 44.541 | 68.524 | 29.000 | 50.394 | 89.942 | 137 | 0 | 0 | 44 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.
