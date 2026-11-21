Status: **COMPLETE**

Dev-selected checkpoints; all adapted arms use a common maximum training budget.

| Method | Split | Selected joint epoch | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original RGB ByteTrack | dev | 0 | 43.936 | 70.257 | 27.514 | 50.402 | 91.472 | 83.000 |
| Original RGB ByteTrack | test | 0 | 29.191 | 43.866 | 19.570 | 30.932 | 47.207 | 390.000 |
| V3 + old legacy IR | dev | 3 | 45.764 | 67.849 | 30.899 | 52.179 | 90.345 | 89.000 |
| V3 + new DiffV2IR | dev | 3 | 46.236 | 68.766 | 31.137 | 53.119 | 90.224 | 108.000 |
| RGB matched fine-tune + ByteTrack | dev | 3 | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94.000 |
| V3 + old legacy IR | test | 3 | 29.738 | 46.595 | 19.068 | 33.736 | 51.667 | 455.000 |
| V3 + new DiffV2IR | test | 3 | 29.195 | 43.916 | 19.487 | 32.678 | 49.603 | 383.000 |
| RGB matched fine-tune + ByteTrack | test | 3 | 29.529 | 46.187 | 18.968 | 33.072 | 50.898 | 450.000 |

Supplementary official TrackEval components on the tracker outputs. HOTA detection ratios
average IoU thresholds; CLEAR recall/precision use IoU0.5. These are not raw detector AP.

| Method | Split | DetRe | DetPr | LocA | CLR_Re | CLR_Pr | CLR_FN | CLR_FP | Frag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original RGB ByteTrack | dev | 74.719 | 78.279 | 81.051 | 93.552 | 98.009 | 2996 | 883 | 299 |
| Original RGB ByteTrack | test | 49.786 | 63.380 | 75.358 | 63.014 | 80.219 | 53597 | 22517 | 1884 |
| V3 + old legacy IR | dev | 72.905 | 75.323 | 79.392 | 93.664 | 96.769 | 2944 | 1453 | 379 |
| V3 + new DiffV2IR | dev | 73.569 | 76.189 | 79.789 | 93.509 | 96.839 | 3016 | 1418 | 374 |
| RGB matched fine-tune + ByteTrack | dev | 73.813 | 75.977 | 79.729 | 93.778 | 96.526 | 2891 | 1568 | 375 |
| V3 + old legacy IR | test | 53.851 | 61.083 | 74.078 | 70.071 | 79.481 | 43371 | 26214 | 2221 |
| V3 + new DiffV2IR | test | 50.366 | 61.514 | 74.492 | 65.872 | 80.453 | 49455 | 23193 | 2069 |
| RGB matched fine-tune + ByteTrack | test | 53.525 | 61.103 | 74.336 | 69.404 | 79.230 | 44338 | 26366 | 2272 |

All Dev checkpoints (no Test scores used for selection):

| Method | Joint epoch | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V3 + old legacy IR | 3 | 45.764 | 67.849 | 30.899 | 52.179 | 90.345 | 89.000 | True |
| V3 + old legacy IR | 6 | 42.613 | 67.488 | 26.943 | 48.364 | 89.615 | 99.000 | False |
| V3 + old legacy IR | 9 | 42.345 | 66.775 | 26.882 | 46.905 | 88.414 | 104.000 | False |
| V3 + old legacy IR | 12 | 41.689 | 66.269 | 26.262 | 45.488 | 87.299 | 118.000 | False |
| V3 + new DiffV2IR | 3 | 46.236 | 68.766 | 31.137 | 53.119 | 90.224 | 108.000 | True |
| V3 + new DiffV2IR | 6 | 44.233 | 68.042 | 28.805 | 52.285 | 89.652 | 105.000 | False |
| V3 + new DiffV2IR | 9 | 41.617 | 66.885 | 25.934 | 46.603 | 87.934 | 106.000 | False |
| V3 + new DiffV2IR | 12 | 42.314 | 65.651 | 27.343 | 48.174 | 86.210 | 114.000 | False |
| RGB matched fine-tune + ByteTrack | 3 | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94.000 | True |
| RGB matched fine-tune + ByteTrack | 6 | 45.679 | 67.435 | 30.988 | 52.691 | 88.752 | 106.000 | False |
| RGB matched fine-tune + ByteTrack | 9 | 43.356 | 66.971 | 28.143 | 50.082 | 87.727 | 125.000 | False |
| RGB matched fine-tune + ByteTrack | 12 | 44.621 | 66.203 | 30.138 | 50.255 | 86.610 | 140.000 | False |

Individual Test sequences:

| Method | Sequence | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original RGB ByteTrack | BT-002 | 35.484 | 42.507 | 29.726 | 45.206 | 18.742 | 22.000 |
| Original RGB ByteTrack | BT-004 | 42.092 | 59.737 | 29.758 | 53.260 | 82.084 | 23.000 |
| Original RGB ByteTrack | MSK-003 | 14.532 | 8.862 | 24.090 | 14.423 | 3.975 | 13.000 |
| Original RGB ByteTrack | PF-002 | 61.160 | 65.238 | 57.351 | 85.628 | 86.050 | 1.000 |
| Original RGB ByteTrack | SN-009 | 27.966 | 58.920 | 13.342 | 27.619 | 73.286 | 118.000 |
| Original RGB ByteTrack | SN-011 | 17.561 | 39.464 | 7.973 | 17.526 | 24.043 | 140.000 |
| Original RGB ByteTrack | SN-014 | 40.424 | 60.233 | 27.222 | 47.165 | 76.565 | 73.000 |
| V3 + old legacy IR | BT-002 | 37.238 | 45.172 | 30.753 | 49.880 | 30.487 | 18.000 |
| V3 + old legacy IR | BT-004 | 48.058 | 59.670 | 38.748 | 63.873 | 81.363 | 14.000 |
| V3 + old legacy IR | MSK-003 | 24.062 | 24.592 | 23.749 | 31.681 | 16.096 | 52.000 |
| V3 + old legacy IR | PF-002 | 54.210 | 63.898 | 46.043 | 70.199 | 84.433 | 2.000 |
| V3 + old legacy IR | SN-009 | 28.044 | 56.871 | 13.899 | 30.119 | 72.145 | 138.000 |
| V3 + old legacy IR | SN-011 | 21.030 | 41.703 | 10.738 | 23.607 | 34.328 | 147.000 |
| V3 + old legacy IR | SN-014 | 33.611 | 58.056 | 19.540 | 35.122 | 73.979 | 84.000 |
| V3 + new DiffV2IR | BT-002 | 38.235 | 42.715 | 34.384 | 49.814 | 25.355 | 11.000 |
| V3 + new DiffV2IR | BT-004 | 52.293 | 57.458 | 47.615 | 73.124 | 77.523 | 16.000 |
| V3 + new DiffV2IR | MSK-003 | 15.558 | 10.709 | 22.840 | 18.336 | 9.328 | 14.000 |
| V3 + new DiffV2IR | PF-002 | 64.062 | 63.824 | 64.304 | 92.166 | 84.317 | 0.000 |
| V3 + new DiffV2IR | SN-009 | 27.061 | 56.699 | 12.974 | 29.088 | 70.927 | 125.000 |
| V3 + new DiffV2IR | SN-011 | 19.718 | 42.042 | 9.404 | 19.812 | 35.219 | 138.000 |
| V3 + new DiffV2IR | SN-014 | 32.591 | 57.263 | 18.628 | 38.882 | 73.301 | 79.000 |
| RGB matched fine-tune + ByteTrack | BT-002 | 40.641 | 43.789 | 37.853 | 53.296 | 29.513 | 14.000 |
| RGB matched fine-tune + ByteTrack | BT-004 | 46.118 | 61.090 | 34.887 | 58.791 | 82.163 | 13.000 |
| RGB matched fine-tune + ByteTrack | MSK-003 | 23.484 | 22.214 | 25.251 | 29.390 | 15.032 | 48.000 |
| RGB matched fine-tune + ByteTrack | PF-002 | 59.151 | 63.774 | 54.864 | 83.870 | 84.533 | 3.000 |
| RGB matched fine-tune + ByteTrack | SN-009 | 26.135 | 57.327 | 11.980 | 26.737 | 71.956 | 136.000 |
| RGB matched fine-tune + ByteTrack | SN-011 | 20.170 | 41.192 | 10.037 | 21.636 | 32.043 | 150.000 |
| RGB matched fine-tune + ByteTrack | SN-014 | 34.639 | 58.225 | 20.698 | 40.439 | 74.063 | 86.000 |
