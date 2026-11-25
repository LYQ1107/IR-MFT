# Frozen full-Train family: complete7-sequence Test

All7historicalTest15985frames for every formal fullTrain configuration, including negative cases. Choices use all50complete Dev only. Three complete Train traversals/20736updates per fit, not72full-data epochs. PriorTest/pretrainingDev exposure remains; not newly blind. All metrics and failed versions retained. Conditional removal deltas cannot be added.

| Case | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FTT_primary_IR | 29.834 | 46.163 | 19.367 | 33.059 | 51.041 | 459 |
| FTT_matched_rgb | 29.467 | 46.241 | 18.864 | 32.337 | 50.878 | 456 |
| FTT_matched_gray | 29.422 | 45.967 | 18.918 | 32.165 | 51.121 | 454 |
| FTT_no_S_paired | 29.581 | 46.386 | 18.970 | 32.738 | 50.955 | 451 |
| FTT_no_S_Dev_best | 29.737 | 46.588 | 19.067 | 32.774 | 50.789 | 460 |
| FTT_no_C_paired | 29.357 | 46.105 | 18.773 | 32.303 | 50.761 | 454 |
| FTT_no_C_Dev_best | 29.419 | 46.129 | 18.843 | 31.897 | 50.830 | 476 |
| FTT_no_R_paired | 30.256 | 46.144 | 19.933 | 33.917 | 50.773 | 455 |
| FTT_no_R_Dev_best | 31.060 | 46.616 | 20.790 | 34.614 | 51.057 | 448 |
| FTT_IR_image_roll | 29.210 | 46.319 | 18.500 | 32.141 | 50.885 | 447 |
| FTT_IR_unavailable | 29.529 | 46.187 | 18.968 | 33.072 | 50.897 | 450 |
| FTT_source_RGB_fresh | 29.529 | 46.187 | 18.968 | 33.072 | 50.897 | 450 |
| FTT_source_RGB_canonical | 29.529 | 46.187 | 18.968 | 33.072 | 50.898 | 450 |

Frozen primary−canonicalRGB: +0.305; primary−freshRGB: +0.305.

| Removal | Same epoch/policy | Dev-best removal |
| --- | ---: | ---: |
| S | +0.253 | +0.097 |
| C | +0.477 | +0.415 |
| R | -0.422 | -1.226 |

A numerical threshold or positive removals alone does not establish2-3innovations. Manual complete-scope contribution and prior-art assessment is still required.

Runtime is separately measured; cached replay FPS excludes encoder/fusion/IR generation.
