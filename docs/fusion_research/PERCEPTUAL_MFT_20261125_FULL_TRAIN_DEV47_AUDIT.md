# Six full-Train fits: independent complete Dev audit

Independent original276column official47 extraction of all50completeDev cases. Original frozen primary/control/removal choices preserved; no Test accuracy input or new model selection. Dev gains cannot establish Test+5or2-3independent innovations; same-family structural removals are not distinct architectures.

Frozen primary: FT_ir_111_s0_e24_raw; Dev HOTA 48.764689.
Versus strongest matched RGB/gray: +0.835747 points.
Versus historical canonical RGB on Dev: +1.185526 points.

| Case | Dev HOTA | DetA | AssA | IDF1 |
| --- | ---: | ---: | ---: | ---: |
| FT_gray_111_s0_e24_priority | 47.567422 | 68.828821 | 32.908148 | 55.844864 |
| FT_gray_111_s0_e24_raw | 47.928942 | 68.350873 | 33.644869 | 55.262410 |
| FT_gray_111_s0_e48_priority | 45.199177 | 68.673479 | 29.784118 | 51.929097 |
| FT_gray_111_s0_e48_raw | 45.377294 | 68.772703 | 29.977806 | 51.573974 |
| FT_gray_111_s0_e72_priority | 45.195585 | 68.668748 | 29.781398 | 51.933063 |
| FT_gray_111_s0_e72_raw | 46.761820 | 68.363726 | 32.015071 | 52.863321 |
| FT_ir_011_s0_e24_priority | 47.571346 | 68.842323 | 32.907191 | 55.856544 |
| FT_ir_011_s0_e24_raw | 46.950312 | 68.765090 | 32.095337 | 54.105700 |
| FT_ir_011_s0_e48_priority | 47.029324 | 68.791886 | 32.188254 | 55.268299 |
| FT_ir_011_s0_e48_raw | 50.586362 | 68.744405 | 37.245806 | 58.610595 |
| FT_ir_011_s0_e72_priority | 47.100368 | 68.817956 | 32.272910 | 55.271830 |
| FT_ir_011_s0_e72_raw | 49.286334 | 68.901160 | 35.279895 | 56.422519 |
| FT_ir_101_s0_e24_priority | 47.568451 | 68.836934 | 32.905859 | 55.869677 |
| FT_ir_101_s0_e24_raw | 47.853305 | 68.367473 | 33.529119 | 55.789416 |
| FT_ir_101_s0_e48_priority | 47.925732 | 68.910865 | 33.364272 | 56.305235 |
| FT_ir_101_s0_e48_raw | 48.207604 | 68.891949 | 33.766512 | 56.314876 |
| FT_ir_101_s0_e72_priority | 47.180971 | 68.808316 | 32.391056 | 55.408355 |
| FT_ir_101_s0_e72_raw | 48.023321 | 68.920857 | 33.496547 | 55.457066 |
| FT_ir_110_s0_e24_priority | 47.766327 | 68.826468 | 33.183121 | 56.132734 |
| FT_ir_110_s0_e24_raw | 47.469356 | 68.230882 | 33.059638 | 55.644915 |
| FT_ir_110_s0_e48_priority | 47.800277 | 68.858622 | 33.214403 | 56.179358 |
| FT_ir_110_s0_e48_raw | 49.635292 | 68.276798 | 36.117393 | 59.390868 |
| FT_ir_110_s0_e72_priority | 48.600426 | 68.913567 | 34.305378 | 56.835302 |
| FT_ir_110_s0_e72_raw | 50.142461 | 68.380541 | 36.803225 | 59.804018 |
| FT_ir_111_s0_e24_IR_image_spatial_roll_priority | 48.193275 | 68.940785 | 33.723004 | 56.540536 |
| FT_ir_111_s0_e24_IR_image_spatial_roll_raw | 48.531046 | 68.340093 | 34.497789 | 56.731852 |
| FT_ir_111_s0_e24_priority | 47.920599 | 68.777217 | 33.420785 | 56.159258 |
| FT_ir_111_s0_e24_raw | 48.764689 | 68.853615 | 34.566912 | 56.876889 |
| FT_ir_111_s0_e24_unavailable_priority | 47.579162 | 68.863720 | 32.907745 | 55.865239 |
| FT_ir_111_s0_e24_unavailable_raw | 47.579162 | 68.863720 | 32.907745 | 55.865239 |
| FT_ir_111_s0_e48_IR_image_spatial_roll_priority | 47.815341 | 68.902381 | 33.214602 | 56.154174 |
| FT_ir_111_s0_e48_IR_image_spatial_roll_raw | 47.152044 | 68.375293 | 32.552948 | 55.383238 |
| FT_ir_111_s0_e48_priority | 48.208683 | 68.932308 | 33.746858 | 56.550445 |
| FT_ir_111_s0_e48_raw | 47.930959 | 68.321343 | 33.662512 | 56.065968 |
| FT_ir_111_s0_e48_unavailable_priority | 47.579162 | 68.863720 | 32.907745 | 55.865239 |
| FT_ir_111_s0_e48_unavailable_raw | 47.579162 | 68.863720 | 32.907745 | 55.865239 |
| FT_ir_111_s0_e72_IR_image_spatial_roll_priority | 47.803434 | 68.870484 | 33.213353 | 56.140351 |
| FT_ir_111_s0_e72_IR_image_spatial_roll_raw | 46.878603 | 68.373295 | 32.173068 | 54.822579 |
| FT_ir_111_s0_e72_priority | 48.243627 | 68.960252 | 33.781721 | 56.565392 |
| FT_ir_111_s0_e72_raw | 48.329519 | 68.382407 | 34.191518 | 56.588326 |
| FT_ir_111_s0_e72_unavailable_priority | 47.579162 | 68.863720 | 32.907745 | 55.865239 |
| FT_ir_111_s0_e72_unavailable_raw | 47.579162 | 68.863720 | 32.907745 | 55.865239 |
| FT_rgb_111_s0_e24_priority | 47.878421 | 68.652885 | 33.421589 | 56.553380 |
| FT_rgb_111_s0_e24_raw | 45.668579 | 68.201601 | 30.621843 | 53.912474 |
| FT_rgb_111_s0_e48_priority | 45.959884 | 68.804766 | 30.737853 | 53.362450 |
| FT_rgb_111_s0_e48_raw | 47.226172 | 68.517630 | 32.586238 | 54.967849 |
| FT_rgb_111_s0_e72_priority | 45.891988 | 68.777455 | 30.659823 | 53.359973 |
| FT_rgb_111_s0_e72_raw | 46.719233 | 68.392890 | 31.949768 | 54.436733 |
| FT_source_RGB_canonical | 47.579162 | 68.863720 | 32.907745 | 55.865239 |
| FT_source_RGB_fresh | 47.579162 | 68.863720 | 32.907745 | 55.865239 |

| Removed mechanism | Primary minus paired | Primary minus removal own best |
| --- | ---: | ---: |
| S | +1.814376 | -1.821674 |
| C | +0.911384 | +0.557085 |
| R | +1.295333 | -1.377773 |

Primary minus wrong-IR HOTA: +0.233642 points. Missing IR reproduces its policy RGB in every official field and entire CSV.

Dev only. New Test accuracy was not read. Three complete Train traversals per fit, not72full epochs. Goal remains active; supported independent contributions0.
