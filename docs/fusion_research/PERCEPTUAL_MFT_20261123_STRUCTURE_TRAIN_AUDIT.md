# Train-only local self-similarity correspondence diagnostic

24 Train frames only, not heldout correspondence or tracking HOTA. Class1 box support is not a pixel landmark or segmentation label. Only synthetic grayscale translation has known displacement truth; actualIR current-vs-wrong structure discrimination does not prove correct fish geometry. All3 IR views use exactly the same intersection-textured source-grid pixels. No fitting, new full cache, Test evaluation or novelty claim is activated automatically.

| Video | View | Textured bbox cells | Best cost | Zero-offset cost | Cycle-consistent fraction | Known-shift exact fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BT-001 | gray_translated | 1496 | 0.00000 | 0.30592 | 1.0000 | 1.0000 |
| BT-001 | IR_current | 1139 | 0.13274 | 0.29108 | 0.4074 | — |
| BT-001 | IR_wrong_frame | 1139 | 0.17557 | 0.35435 | 0.3819 | — |
| BT-001 | IR_spatial_roll | 1139 | 0.17246 | 0.35409 | 0.3731 | — |
| BT-003 | gray_translated | 1906 | 0.00000 | 0.32500 | 1.0000 | 1.0000 |
| BT-003 | IR_current | 1308 | 0.12286 | 0.28012 | 0.4358 | — |
| BT-003 | IR_wrong_frame | 1308 | 0.16946 | 0.35525 | 0.3807 | — |
| BT-003 | IR_spatial_roll | 1308 | 0.17369 | 0.36262 | 0.3532 | — |
| MSK-002 | gray_translated | 1286 | 0.00000 | 0.35692 | 1.0000 | 1.0000 |
| MSK-002 | IR_current | 1169 | 0.08725 | 0.18792 | 0.5962 | — |
| MSK-002 | IR_wrong_frame | 1169 | 0.16566 | 0.35361 | 0.3952 | — |
| MSK-002 | IR_spatial_roll | 1169 | 0.19093 | 0.38766 | 0.3259 | — |
| PF-001 | gray_translated | 4207 | 0.00002 | 0.29922 | 0.9998 | 0.9995 |
| PF-001 | IR_current | 3267 | 0.15767 | 0.32494 | 0.3988 | — |
| PF-001 | IR_wrong_frame | 3267 | 0.17501 | 0.34296 | 0.3633 | — |
| PF-001 | IR_spatial_roll | 3267 | 0.19111 | 0.35914 | 0.3404 | — |
| SN-001 | gray_translated | 2161 | 0.00001 | 0.34037 | 1.0000 | 1.0000 |
| SN-001 | IR_current | 1949 | 0.13822 | 0.30676 | 0.4361 | — |
| SN-001 | IR_wrong_frame | 1949 | 0.16924 | 0.35459 | 0.3704 | — |
| SN-001 | IR_spatial_roll | 1949 | 0.18811 | 0.37206 | 0.3289 | — |
| SN-015 | gray_translated | 4496 | 0.00005 | 0.32758 | 1.0000 | 1.0000 |
| SN-015 | IR_current | 2400 | 0.13420 | 0.29701 | 0.4029 | — |
| SN-015 | IR_wrong_frame | 2400 | 0.15236 | 0.32328 | 0.3713 | — |
| SN-015 | IR_spatial_roll | 2400 | 0.20118 | 0.38871 | 0.2946 | — |

Prerequisite: True; known-translation passes 6/6, current-vs-wrong IR cost wins 6/6 and cycle wins 6/6.

Classical prior: [MIND2012](https://pubmed.ncbi.nlm.nih.gov/22722056/) and [self-similarity context2013](https://iacl.ece.jhu.edu/proceedings/miccai2013/papers/8149/81490187.pdf). This8-neighbor2D probe is an inspired implementation, not the original complete registration algorithm.
