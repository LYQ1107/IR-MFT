# All36 new no-cycle matched controls and removals

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Four distinct seed0 fits,24passes/6912updates/1152Traincacheframes and145600parameters each. Bothsource8/prefix32/full39proofs precede accuracy. Existing IR no_cycle/no_structure and canonical RGB17results are referenced in the analysis, not rerun. No newTest/noveltyclaim.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NC_IR_image_roll_s0_e06 | 47.693 | 68.877 | 33.061 | 55.854 | 90.205 | 93 | 0 | 0 |
| NC_IR_image_roll_s0_e06_raw | 47.376 | 68.827 | 32.649 | 55.264 | 90.237 | 96 | 0 | 0 |
| NC_IR_unavailable_s0_e06 | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 | 0 | 0 |
| NC_IR_unavailable_s0_e06_raw | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 | 0 | 0 |
| NC_gray_no_cycle_s0_e06 | 48.792 | 68.892 | 34.587 | 56.704 | 90.095 | 88 | 0 | 0 |
| NC_gray_no_cycle_s0_e06_raw | 49.132 | 68.506 | 35.280 | 56.570 | 89.927 | 88 | 0 | 0 |
| NC_gray_no_cycle_s0_e12 | 48.142 | 68.790 | 33.722 | 56.042 | 90.116 | 98 | 0 | 0 |
| NC_gray_no_cycle_s0_e12_raw | 48.642 | 68.522 | 34.567 | 56.791 | 89.895 | 98 | 0 | 0 |
| NC_gray_no_cycle_s0_e18 | 48.145 | 68.802 | 33.720 | 56.063 | 90.134 | 100 | 0 | 0 |
| NC_gray_no_cycle_s0_e18_raw | 48.609 | 68.413 | 34.584 | 56.732 | 89.860 | 103 | 0 | 0 |
| NC_gray_no_cycle_s0_e24 | 48.273 | 68.803 | 33.900 | 56.142 | 90.144 | 98 | 0 | 0 |
| NC_gray_no_cycle_s0_e24_raw | 48.739 | 68.472 | 34.739 | 56.708 | 89.824 | 106 | 0 | 0 |
| NC_ir_no_cycle_no_centering_s0_e06 | 47.129 | 68.746 | 32.342 | 54.005 | 90.147 | 88 | 0 | 0 |
| NC_ir_no_cycle_no_centering_s0_e06_raw | 46.849 | 68.788 | 31.946 | 52.697 | 90.198 | 92 | 0 | 0 |
| NC_ir_no_cycle_no_centering_s0_e12 | 47.598 | 68.925 | 32.903 | 54.399 | 90.106 | 87 | 0 | 0 |
| NC_ir_no_cycle_no_centering_s0_e12_raw | 47.971 | 68.768 | 33.496 | 53.683 | 90.138 | 88 | 0 | 0 |
| NC_ir_no_cycle_no_centering_s0_e18 | 47.602 | 68.947 | 32.899 | 54.416 | 90.142 | 86 | 0 | 0 |
| NC_ir_no_cycle_no_centering_s0_e18_raw | 46.203 | 68.631 | 31.144 | 51.299 | 90.052 | 96 | 0 | 0 |
| NC_ir_no_cycle_no_centering_s0_e24 | 47.602 | 68.943 | 32.900 | 54.416 | 90.136 | 87 | 0 | 0 |
| NC_ir_no_cycle_no_centering_s0_e24_raw | 47.606 | 68.679 | 33.035 | 53.312 | 90.138 | 90 | 0 | 0 |
| NC_ir_no_cycle_no_risk_s0_e06 | 47.777 | 68.837 | 33.193 | 56.104 | 90.187 | 94 | 0 | 0 |
| NC_ir_no_cycle_no_risk_s0_e06_raw | 47.846 | 68.496 | 33.456 | 55.022 | 89.852 | 95 | 0 | 0 |
| NC_ir_no_cycle_no_risk_s0_e12 | 46.612 | 68.873 | 31.581 | 53.223 | 90.110 | 93 | 0 | 0 |
| NC_ir_no_cycle_no_risk_s0_e12_raw | 45.501 | 68.178 | 30.414 | 51.613 | 89.832 | 100 | 0 | 0 |
| NC_ir_no_cycle_no_risk_s0_e18 | 48.629 | 68.924 | 34.338 | 55.979 | 90.215 | 93 | 0 | 0 |
| NC_ir_no_cycle_no_risk_s0_e18_raw | 47.784 | 68.359 | 33.437 | 55.115 | 90.043 | 100 | 0 | 0 |
| NC_ir_no_cycle_no_risk_s0_e24 | 48.622 | 68.916 | 34.332 | 55.973 | 90.200 | 93 | 0 | 0 |
| NC_ir_no_cycle_no_risk_s0_e24_raw | 46.785 | 68.407 | 32.033 | 53.321 | 90.005 | 97 | 0 | 0 |
| NC_rgb_no_cycle_s0_e06 | 47.918 | 68.918 | 33.351 | 56.256 | 90.125 | 89 | 0 | 0 |
| NC_rgb_no_cycle_s0_e06_raw | 47.887 | 68.490 | 33.517 | 55.351 | 90.125 | 97 | 0 | 0 |
| NC_rgb_no_cycle_s0_e12 | 46.363 | 68.934 | 31.210 | 52.077 | 90.140 | 84 | 0 | 0 |
| NC_rgb_no_cycle_s0_e12_raw | 45.512 | 68.469 | 30.289 | 50.832 | 90.007 | 99 | 0 | 0 |
| NC_rgb_no_cycle_s0_e18 | 46.445 | 68.959 | 31.309 | 52.065 | 90.177 | 83 | 0 | 0 |
| NC_rgb_no_cycle_s0_e18_raw | 45.312 | 68.341 | 30.077 | 50.395 | 89.953 | 103 | 0 | 0 |
| NC_rgb_no_cycle_s0_e24 | 46.360 | 68.930 | 31.209 | 52.078 | 90.138 | 84 | 0 | 0 |
| NC_rgb_no_cycle_s0_e24_raw | 45.968 | 68.274 | 30.984 | 52.257 | 89.959 | 103 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.
