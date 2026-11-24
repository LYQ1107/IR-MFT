# Structural IR: verified per-method runtime

Six separate methods, each2Devvideos x32frames;8warmup+24timed/video. B4throughput, notB1latency. Includes original image preprocessing, live descriptor extraction, transfers, requiredshared encoder passes, current/null/raw teacher heads, NMS and ByteTrack. Generation excluded. Memory-input mode excludes fileIO/preload. Per-method inference includes oneadapter only. Other project load and filesystem caches are uncontrolled. Every raw frame packet bit-exact to immutable prefixes or completed original IR packets; prefix/full37agreement is independently required by the Dev controller before accuracy. All adapters are deliberately profiled at6epochs for architecture cost, not selected using accuracy.

| Method at epoch6 | Adapter params | Live-file FPS | RAM-input pipeline FPS | NN/NMS/output-transfer FPS | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB_fresh_baseline | 0 | 24.457 | 29.083 | 78.339 | 0.817 |
| FCT_000_e06_raw | 145600 | 2.406 | 5.280 | 31.663 | 1.030 |
| FCT_001_e06_raw | 145600 | 2.753 | 5.304 | 31.942 | 1.030 |
| FCT_010_e06_raw | 145600 | 2.706 | 5.279 | 31.918 | 1.030 |
| FCT_100_e06_raw | 145600 | 2.721 | 5.156 | 31.039 | 1.030 |
| FCT_111_e06_raw | 145600 | 2.540 | 5.204 | 31.404 | 1.030 |

Both runs check384frame/method comparisons each on64uniqueDevframes. These are two input conditions, not additional model seeds or independent accuracy trials. Full JSON retains source packet/checkpoint/code hashes, preparation and tracking time. Dataset reading and IR generation are excluded only where explicitly stated; generation is excluded in both. B4throughput is not B1online latency. Shared hardware load is uncontrolled.
