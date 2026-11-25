# Frozen full-Train methods: single-method runtime

Independent single-model B4 throughput on2Devvideos,8warmup+24timed frames/video. Every64frame proposal/policy output exactly matches frozen prefixes. Includes inputpreparation,transfers,CNN,fusion,NMS,selected raw/priority policy and ByteTrack. Memory-input condition excludes image reads/preload; both exclude IRgeneration/model loading/track-file serialization. No gate observer. Shared-host load and filecache uncontrolled; not B1latency or a real-time claim.

| Case | FPS | NN/NMS/policy/transfer FPS | Parameters | Peak own allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| FTT_primary_IR | 4.776 | 31.088 | 99140978 | 1.031 |
| FTT_matched_rgb | 8.731 | 38.525 | 99140978 | 0.929 |
| FTT_matched_gray | 7.000 | 30.682 | 99140978 | 1.029 |
| FTT_no_S_paired | 5.073 | 31.602 | 99140978 | 1.030 |
| FTT_no_S_Dev_best | 4.823 | 30.847 | 99140978 | 1.030 |
| FTT_no_C_paired | 4.931 | 30.799 | 99140978 | 1.030 |
| FTT_no_C_Dev_best | 5.075 | 30.144 | 99140978 | 1.030 |
| FTT_no_R_paired | 5.112 | 31.251 | 99140978 | 1.031 |
| FTT_no_R_Dev_best | 5.120 | 30.908 | 99140978 | 1.031 |
| FTT_source_RGB_fresh | 28.951 | 78.422 | 98995378 | 0.817 |
