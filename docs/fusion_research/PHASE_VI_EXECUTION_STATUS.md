# Phase VI execution status

This branch is the reproducible implementation for the causal multimodal
fusion study. The task-book path `/data1/LWR/vranlee/CELL/IR-MFT` was not
present on this host; the repository is `/data3/liuyeqiang/IR-MFT`, the frozen
MFT26-IR data is `/data3/liuyeqiang/datasets/mft26-ir`, and weights/results are
kept under `/data3/liuyeqiang/IR-MFT_assets`.

All GPU commands are restricted to physical GPUs 2 and 3 through
`CUDA_VISIBLE_DEVICES=2,3` (or one of those two devices for a shard). No other
GPU is used by this study.

Completed and checked:

- Protocol V2: Train 6 sequences / 27,251 frames, Dev 2 / 4,834, historical
  Test 7 / 15,985. The validator and the per-split dataset filters are in
  `tools/validate_mft26ir_protocol.py` and `docs/fusion_research/PROTOCOL_V2_VALIDATION.json`.
- Strict causal RGB-derived auxiliary generation: the 15-sequence prefix
  perturbation audit passes, including sequence-state reset.
- Unified `DynamicQualityFusion`, `SpatialComplementarityFusion`, and
  `TemporalComplementarityFusion` interfaces, plus quality-spatial and
  redundancy ablations. QRA shape/gradient tests pass and the zero-initialized
  dynamic candidate is numerically equivalent to the V3 fusion output on the
  fixed smoke input.
- The 20 pinned public source files were audited with exact raw-file SHA256,
  symbol/line evidence, interface/cost notes, and license boundaries in
  `OPEN_SOURCE_20_CODE_AUDIT.{json,md}`. No third-party source was copied.
- V3-compatible initialization checkpoints were produced outside Git for the
  Dynamic Quality, Quality+Spatial, Quality+Spatial+Redundancy, and Temporal
  candidates. Their load reports record copied keys and the only new keys.

The Train causal cache is stored as a single-channel response JPEG and
re-colourized at load time; it covers all 27,251 Train frames and is checked
against selected online-generated frames by `tools/validate_causal_cache.py`.
Large checkpoints, causal cache frames, prediction files, and TrackEval
artifacts are intentionally outside Git. Candidate training/evaluation remains
an external running job (Stage 1 uses 320x576 training crops for the short
screening budget; formal tracking evaluation stays at the fixed 800x1440 test
size) and its status is recorded in the asset directory.

## Stage 2 screening evidence

The first two trained candidates were run with the same detector/tracker
protocol. Their checkpoints and per-frame outputs are external to Git under
`/data3/liuyeqiang/IR-MFT_assets`.

| candidate | split | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F1 dynamic quality | Dev | 4,834 | 13.049 | 28.794 | 6.0217 | 9.2367 | -45.718 | 1,112 | gate collapse / negative |
| F1 dynamic quality | Test | 15,985 | 16.529 | 27.106 | 10.328 | 12.562 | -48.971 | 2,422 | gate collapse / negative |
| F2a quality + spatial | Dev | 4,834 | 14.742 | 33.616 | 6.5436 | 11.860 | -12.206 | 867 | negative, but better than F1 Dev |
| F2a quality + spatial | Test | 15,985 | 16.501 | 28.268 | 9.7849 | 12.983 | -40.636 | 2,224 | negative, no HOTA gain over F1 |
| F2b quality + spatial | Dev | 4,834 | 14.332 | 31.188 | 6.6399 | 10.656 | -25.589 | 1,035 | negative / gate collapse |
| F2b quality + spatial | Test | 15,985 | 15.679 | 27.131 | 9.2641 | 12.002 | -47.678 | 2,430 | negative / gate collapse |

The fixed V3 static-fusion reference is HOTA 46.086 on Dev (4,834 frames) and
the historical Test reference is HOTA 31.091 on 15,985 frames. Therefore F1
and F2a are retained as auditable negative experiments; they are not reported
as improvements. F1's learned gate is nearly constant (roughly 0.054--0.061
per pyramid level). F2a has the same collapse pattern, although its Dev
detection scores and MOTA are less poor. These observations motivate testing
the quality-spatial and redundancy-aware candidates with the same protocol
before drawing a conclusion.

F2b (`quality_spatial`) and F2c (`quality_spatial_redundancy`) are trained next
with two-process DDP on physical GPUs 2 and 3 only. The short screening
training resolution is 320x576, while all formal Dev/Test tracking metrics use
the fixed protocol resolution and all seven Test sequences. No other GPU is
used, and no checkpoint, cache frame, prediction file, or TrackEval artifact is
tracked in Git.

F2b used the final three-epoch checkpoint. On all 4,834 Dev frames its
TrackEval summary is HOTA 14.332, DetA 31.188, AssA 6.6399, IDF1 10.656,
MOTA -25.589, and IDSW 1,035 at 6.21 FPS. On all 15,985 Test frames it is
HOTA 15.679, DetA 27.131, AssA 9.2641, IDF1 12.002, MOTA -47.678, and
IDSW 2,430 at 6.68 FPS. Both splits remain negative against the fixed V3
reference and F2a; the Test result also declines from F1/F2a HOTA.
