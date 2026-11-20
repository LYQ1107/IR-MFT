Status: **INCOMPLETE**

Dev-selected checkpoints; all adapted arms use a common maximum training budget.

| Method | Split | Selected joint epoch | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original RGB ByteTrack | dev | 0 | 43.936 | 70.257 | 27.514 | 50.402 | 91.472 | 83.000 |
| Original RGB ByteTrack | test | 0 | 29.191 | 43.866 | 19.570 | 30.932 | 47.207 | 390.000 |
| V3 + old legacy IR | dev | 3 | 45.764 | 67.849 | 30.899 | 52.179 | 90.345 | 89.000 |
| V3 + new DiffV2IR | dev | 3 | 46.236 | 68.766 | 31.137 | 53.119 | 90.224 | 108.000 |
| RGB matched fine-tune + ByteTrack | dev | 3 | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94.000 |

Supplementary official TrackEval components on the tracker outputs. HOTA detection ratios
average IoU thresholds; CLEAR recall/precision use IoU0.5. These are not raw detector AP.

| Method | Split | DetRe | DetPr | LocA | CLR_Re | CLR_Pr | CLR_FN | CLR_FP | Frag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original RGB ByteTrack | dev | 74.719 | 78.279 | 81.051 | 93.552 | 98.009 | 2996 | 883 | 299 |
| Original RGB ByteTrack | test | 49.786 | 63.380 | 75.358 | 63.014 | 80.219 | 53597 | 22517 | 1884 |
| V3 + old legacy IR | dev | 72.905 | 75.323 | 79.392 | 93.664 | 96.769 | 2944 | 1453 | 379 |
| V3 + new DiffV2IR | dev | 73.569 | 76.189 | 79.789 | 93.509 | 96.839 | 3016 | 1418 | 374 |
| RGB matched fine-tune + ByteTrack | dev | 73.813 | 75.977 | 79.729 | 93.778 | 96.526 | 2891 | 1568 | 375 |

All Dev checkpoints (no Test scores used for selection):

| Method | Joint epoch | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V3 + old legacy IR | 3 | 45.764 | 67.849 | 30.899 | 52.179 | 90.345 | 89.000 | True |
| V3 + new DiffV2IR | 3 | 46.236 | 68.766 | 31.137 | 53.119 | 90.224 | 108.000 | True |
| V3 + new DiffV2IR | 6 | 44.233 | 68.042 | 28.805 | 52.285 | 89.652 | 105.000 | False |
| RGB matched fine-tune + ByteTrack | 3 | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94.000 | True |

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
