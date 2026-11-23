# Frozen seed0 methods: separate runtime and storage costs

One method at a time on physicalA100 GPU2; shared pretrained YOLOX-X weights. Each method uses8warmup and32timed frames per Dev video, batch4. All40 proposal arrays in each video exactly reproduce the frozen full-Dev packets (320 frame/method comparisons across4 methods). No GT, training or seed replication.

| Method | Extra parameters | Encoder passes | Live image-file pipeline FPS | RAM-input pipeline FPS | Detection/fusion/NMS/output-transfer FPS | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB_fresh_baseline | 0 | 1 | 22.72 | 25.71 | 74.45 | 0.817 |
| CGA_rgb_Dev_selected | 217619 | 1 | 18.82 | 19.03 | 52.09 | 1.175 |
| CGA_gray_Dev_selected | 217619 | 2 | 13.39 | 14.57 | 37.25 | 1.275 |
| CGA_ir_Dev_selected | 217619 | 2 | 3.57 | 14.54 | 37.21 | 1.275 |

The base model has98995378 parameters; each adapted model has99212997 total parameters. IR and grayscale use two shared-weight encoder passes, while RGB repeat reuses its RGB pass. Equal parameter capacity is not equal computation.

The live-file run includes image-file reads, decoding, preprocessing, GPU transfer, neural processing, NMS and ByteTrack. Filesystem/remote-cache state was uncontrolled. ActualIR PNGs are on NAS, so the3.57FPS IR result cannot be attributed entirely to neural computation. The original numeric report is retained with an explicit correction to its initial cache-scope wording.

The RAM-input run preloads the same original RGB/IR images for all four methods; preload time and temporary raw-image RAM usage are reported in its JSON. Its pipeline FPS excludes file IO and includes preprocessing, transfers, detection/fusion/NMS and tracking. The component FPS additionally excludes input preparation and ByteTrack. Synchronization brackets each section; exact output checks are outside timing.

All figures use pre-generated actualNovember DiffV2IR. Image generation is excluded. B4 throughput is not B1 online latency. A single short pass per method under concurrent host/GPU project load is an observed runtime, not a dedicated-hardware latency benchmark. Peak allocated/reserved tensor memory is reported separately; it does not include all CUDA context memory.

The first run failed while serializing a NumPy integer statistic, before any final report. Its exact source and logs are preserved outside Git; the type fix changed only logging. Both successful runs automatically resumed the waiting Dev controller, and GPU3 fitting continued.
