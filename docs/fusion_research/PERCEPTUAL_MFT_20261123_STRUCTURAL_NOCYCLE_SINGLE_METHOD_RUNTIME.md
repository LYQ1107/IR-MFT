# Structural IR: verified per-method runtime

Five separate methods, each2Devvideos x32frames;8warmup+24timed/video. B4throughput, notB1latency. Includes original image preprocessing, live descriptor extraction, transfers, requiredshared encoder passes, current/null/raw teacher heads, NMS and ByteTrack. Generation excluded. Memory-input mode excludes fileIO/preload. Per-method inference includes oneadapter only. Other project load and filesystem caches are uncontrolled. Every raw frame packet bit-exact to immutable prefixes or completed original IR packets; prefix/full39agreement is independently required by the Dev controller before accuracy. All adapters are deliberately profiled at6epochs for architecture cost, not selected using accuracy.

| Method at epoch6 | Adapter params | Live-file FPS | RAM-input pipeline FPS | NN/NMS/output-transfer FPS | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB_fresh_baseline | 0 | 23.379 | 26.769 | 77.684 | 0.817 |
| NC_rgb_e06_raw | 145600 | 7.860 | 7.897 | 45.953 | 0.928 |
| NC_gray_e06_raw | 145600 | 6.514 | 6.540 | 31.115 | 1.031 |
| NC_IR_e06_raw | 145600 | 2.707 | 4.611 | 31.274 | 1.030 |
| Full_cycle_IR_e06_raw | 145600 | 2.809 | 4.555 | 29.944 | 1.030 |

Both runs check320frame/method comparisons each on64uniqueDevframes. These are two input conditions, not additional model seeds or independent accuracy trials. Full JSON retains source packet/checkpoint/code hashes, preparation and tracking time. Dataset reading and IR generation are excluded only where explicitly stated; generation is excluded in both. B4throughput is not B1online latency. Shared hardware load is uncontrolled.
