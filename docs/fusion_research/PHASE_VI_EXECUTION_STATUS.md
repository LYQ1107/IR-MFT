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
