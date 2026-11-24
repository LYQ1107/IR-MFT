# Native36 robust and consensus full Test backfill

Two already trained native36 query/memory backends, both frozen detector frontends and both matched nativeRGB/NovemberIR cues. Each original complete6-snapshot Dev curve selects its own checkpoint before this Test. All normal/off/current-frame-shuffled cases run regardless of Dev sign. No Test-based promotion; historicalTest exposed. Other formal families remain outside this subset.

| Case | HOTA | IDF1 | DetA | AssA | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NC36_consensus_new_ir_normal | 31.442 | 33.756 | 50.293 | 19.767 | 53.399 | 582 |
| NC36_consensus_new_ir_unavailable | 31.442 | 33.756 | 50.293 | 19.767 | 53.399 | 582 |
| NC36_consensus_new_ir_mismatched | 31.442 | 33.756 | 50.293 | 19.767 | 53.399 | 582 |
| NC36_consensus_new_rgb_normal | 31.443 | 33.755 | 50.292 | 19.768 | 53.400 | 582 |
| NC36_consensus_new_rgb_unavailable | 31.442 | 33.756 | 50.293 | 19.767 | 53.399 | 582 |
| NC36_consensus_new_rgb_mismatched | 31.442 | 33.756 | 50.293 | 19.767 | 53.399 | 582 |
| NC36_consensus_rgb_ir_normal | 30.174 | 31.552 | 50.115 | 18.263 | 52.192 | 641 |
| NC36_consensus_rgb_ir_unavailable | 30.174 | 31.552 | 50.115 | 18.263 | 52.192 | 641 |
| NC36_consensus_rgb_ir_mismatched | 30.174 | 31.552 | 50.115 | 18.263 | 52.192 | 641 |
| NC36_consensus_rgb_rgb_normal | 30.171 | 31.551 | 50.111 | 18.262 | 52.184 | 640 |
| NC36_consensus_rgb_rgb_unavailable | 30.174 | 31.552 | 50.115 | 18.263 | 52.192 | 641 |
| NC36_consensus_rgb_rgb_mismatched | 30.174 | 31.552 | 50.115 | 18.263 | 52.192 | 641 |
| NC36_robust_new_ir_normal | 31.110 | 33.186 | 50.283 | 19.352 | 53.472 | 584 |
| NC36_robust_new_ir_unavailable | 31.442 | 33.756 | 50.293 | 19.767 | 53.399 | 582 |
| NC36_robust_new_ir_mismatched | 31.401 | 33.642 | 50.280 | 19.722 | 53.443 | 583 |
| NC36_robust_new_rgb_normal | 31.019 | 33.075 | 50.286 | 19.239 | 53.354 | 584 |
| NC36_robust_new_rgb_unavailable | 31.442 | 33.756 | 50.293 | 19.767 | 53.399 | 582 |
| NC36_robust_new_rgb_mismatched | 31.400 | 33.638 | 50.276 | 19.722 | 53.436 | 585 |
| NC36_robust_rgb_ir_normal | 30.172 | 31.535 | 50.120 | 18.260 | 52.195 | 638 |
| NC36_robust_rgb_ir_unavailable | 30.174 | 31.552 | 50.115 | 18.263 | 52.192 | 641 |
| NC36_robust_rgb_ir_mismatched | 30.193 | 31.557 | 50.119 | 18.285 | 52.200 | 640 |
| NC36_robust_rgb_rgb_normal | 30.458 | 32.001 | 50.093 | 18.616 | 52.166 | 640 |
| NC36_robust_rgb_rgb_unavailable | 30.174 | 31.552 | 50.115 | 18.263 | 52.192 | 641 |
| NC36_robust_rgb_rgb_mismatched | 30.236 | 31.709 | 50.123 | 18.336 | 52.189 | 639 |
| NC36_rgb_ByteTrack | 29.529 | 33.072 | 46.187 | 18.968 | 50.898 | 450 |
| NC36_new_ByteTrack | 29.195 | 32.678 | 43.916 | 19.487 | 49.603 | 383 |

Disables only the native residual. Preserves the complete learned RGB real/null prior and backend lifecycle; it is not assumed equal to plain ByteTrack.

Existing complete current-frame encoded features and point packets, no new encoding, no extra seeds and no GPU processes. Cached replay FPS excludes detector/encoders/points.

All26 cases cover7Test sequences/15985frames. No partial-case winner selection or automatic goal claim.
