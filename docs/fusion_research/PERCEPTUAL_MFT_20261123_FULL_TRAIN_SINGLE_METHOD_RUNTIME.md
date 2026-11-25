# Frozen full-Train methods: single-method runtime

Independent single-model B4 throughput on2Devvideos,8warmup+24timed frames/video. Every64frame proposal/policy output exactly matches frozen prefixes. Includes inputpreparation,transfers,CNN,fusion,NMS,selected raw/priority policy and ByteTrack. Memory-input condition excludes image reads/preload; both exclude IRgeneration/model loading/track-file serialization. No gate observer. Shared-host load and filecache uncontrolled; not B1latency or a real-time claim.

| Case | FPS | NN/NMS/policy/transfer FPS | Parameters | Peak own allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| FTT_primary_IR | 2.461 | 29.174 | 99140978 | 1.031 |
| FTT_matched_rgb | 8.266 | 38.938 | 99140978 | 0.929 |
| FTT_matched_gray | 7.308 | 31.358 | 99140978 | 1.030 |
| FTT_no_S_paired | 2.790 | 30.627 | 99140978 | 1.030 |
| FTT_no_S_Dev_best | 3.106 | 30.864 | 99140978 | 1.030 |
| FTT_no_C_paired | 3.147 | 30.851 | 99140978 | 1.030 |
| FTT_no_C_Dev_best | 3.083 | 30.286 | 99140978 | 1.030 |
| FTT_no_R_paired | 3.156 | 30.665 | 99140978 | 1.031 |
| FTT_no_R_Dev_best | 2.892 | 30.201 | 99140978 | 1.031 |
| FTT_source_RGB_fresh | 23.382 | 73.224 | 98995378 | 0.817 |
