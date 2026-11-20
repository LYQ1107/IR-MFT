| Method | Split | Frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original RGB ByteTrack | dev | 4834 | 43.936 | 70.257 | 27.514 | 50.402 | 91.472 | 83.000 |
| Original RGB ByteTrack | test | 15985 | 29.191 | 43.866 | 19.570 | 30.932 | 47.207 | 390.000 |
| V3 + old legacy IR | dev | 4834 | 45.764 | 67.849 | 30.899 | 52.179 | 90.345 | 89.000 |
| V3 + old legacy IR | test | 15985 | 29.738 | 46.595 | 19.068 | 33.736 | 51.667 | 455.000 |
| V3 + new DiffV2IR | dev | 4834 | 46.236 | 68.766 | 31.137 | 53.119 | 90.224 | 108.000 |
| RGB matched fine-tune + ByteTrack | dev | 4834 | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94.000 |

Supplementary official TrackEval components on the tracker outputs. DetRe/DetPr/LocA average
the HOTA IoU thresholds; CLEAR recall/precision use IoU0.5. These are not raw detector AP.

| Method | Split | DetRe | DetPr | LocA | CLR_Re | CLR_Pr | CLR_FN | CLR_FP | Frag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original RGB ByteTrack | dev | 74.719 | 78.279 | 81.051 | 93.552 | 98.009 | 2996 | 883 | 299 |
| Original RGB ByteTrack | test | 49.786 | 63.380 | 75.358 | 63.014 | 80.219 | 53597 | 22517 | 1884 |
| V3 + old legacy IR | dev | 72.905 | 75.323 | 79.392 | 93.664 | 96.769 | 2944 | 1453 | 379 |
| V3 + old legacy IR | test | 53.851 | 61.083 | 74.078 | 70.071 | 79.481 | 43371 | 26214 | 2221 |
| V3 + new DiffV2IR | dev | 73.569 | 76.189 | 79.789 | 93.509 | 96.839 | 3016 | 1418 | 374 |
| RGB matched fine-tune + ByteTrack | dev | 73.813 | 75.977 | 79.729 | 93.778 | 96.526 | 2891 | 1568 | 375 |

Individual sequences (TrackEval percentages; IDSW is a count):

| Method | Split | Sequence | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original RGB ByteTrack | dev | BT-005 | 47.016 | 71.831 | 30.802 | 60.025 | 94.885 | 27.000 |
| Original RGB ByteTrack | dev | SN-013 | 41.806 | 69.255 | 25.285 | 44.057 | 89.246 | 56.000 |
| Original RGB ByteTrack | test | BT-002 | 35.484 | 42.507 | 29.726 | 45.206 | 18.742 | 22.000 |
| Original RGB ByteTrack | test | BT-004 | 42.092 | 59.737 | 29.758 | 53.260 | 82.084 | 23.000 |
| Original RGB ByteTrack | test | MSK-003 | 14.532 | 8.862 | 24.090 | 14.423 | 3.975 | 13.000 |
| Original RGB ByteTrack | test | PF-002 | 61.160 | 65.238 | 57.351 | 85.628 | 86.050 | 1.000 |
| Original RGB ByteTrack | test | SN-009 | 27.966 | 58.920 | 13.342 | 27.619 | 73.286 | 118.000 |
| Original RGB ByteTrack | test | SN-011 | 17.561 | 39.464 | 7.973 | 17.526 | 24.043 | 140.000 |
| Original RGB ByteTrack | test | SN-014 | 40.424 | 60.233 | 27.222 | 47.165 | 76.565 | 73.000 |
| V3 + old legacy IR | dev | BT-005 | 55.924 | 69.355 | 45.118 | 70.586 | 92.585 | 21.000 |
| V3 + old legacy IR | dev | SN-013 | 37.765 | 66.889 | 21.379 | 40.087 | 88.884 | 68.000 |
| V3 + old legacy IR | test | BT-002 | 37.238 | 45.172 | 30.753 | 49.880 | 30.487 | 18.000 |
| V3 + old legacy IR | test | BT-004 | 48.058 | 59.670 | 38.748 | 63.873 | 81.363 | 14.000 |
| V3 + old legacy IR | test | MSK-003 | 24.062 | 24.592 | 23.749 | 31.681 | 16.096 | 52.000 |
| V3 + old legacy IR | test | PF-002 | 54.210 | 63.898 | 46.043 | 70.199 | 84.433 | 2.000 |
| V3 + old legacy IR | test | SN-009 | 28.044 | 56.871 | 13.899 | 30.119 | 72.145 | 138.000 |
| V3 + old legacy IR | test | SN-011 | 21.030 | 41.703 | 10.738 | 23.607 | 34.328 | 147.000 |
| V3 + old legacy IR | test | SN-014 | 33.611 | 58.056 | 19.540 | 35.122 | 73.979 | 84.000 |
| V3 + new DiffV2IR | dev | BT-005 | 53.934 | 69.992 | 41.591 | 67.515 | 92.459 | 27.000 |
| V3 + new DiffV2IR | dev | SN-013 | 40.477 | 67.974 | 24.171 | 43.678 | 88.766 | 81.000 |
| RGB matched fine-tune + ByteTrack | dev | BT-005 | 55.174 | 69.658 | 43.722 | 71.457 | 92.268 | 23.000 |
| RGB matched fine-tune + ByteTrack | dev | SN-013 | 41.901 | 68.358 | 25.744 | 45.644 | 88.852 | 71.000 |
