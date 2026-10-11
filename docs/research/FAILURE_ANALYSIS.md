# Failure and repair ledger

Status: historical reproduction drift RESOLVED; causal-input and representation diagnosis IN_PROGRESS, 2026-10-12. All paths below are local experiment artifacts.

## P0-01: full inference differs from historical V3

Run `phase0_v3_cu118_full_20261012`: unchanged detector/tracker code, strict E3 checkpoint, all 15,985 historical frames, batch 8 FP16, fixed historical thresholds. Finished successfully in 1,070.55 seconds (includes I/O, tracking and metric computation; not a per-frame speed benchmark).

| Metric | Historical | Current CUDA 11.8 |
|---|---:|---:|
| HOTA | 31.091 | 30.907 |
| AssA | 19.619 | 19.371 |
| DetA | 49.575 | 49.580 |
| IDF1 | 35.044 | 34.684 |
| MOTA | 52.325 | 52.289 |
| IDSW | 560 | 568 |
| FP | 31,038 | 31,034 |
| FN | 37,489 | 37,537 |
| Frag | 2,560 | 2,558 |

Evidence: `/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/v3_reproduction_20261012/metrics.json`, inference log, `cache_drift.json`. Matched cache boxes are mostly identical or differ at subpixel FP16 quantization, with occasional changed NMS survivors. Compare by frame and IoU; elementwise comparison alone conflates ordering and unmatched detections. Actual image/preprocessed input parity with an old runtime remains to verify.

Controlled checks:

1. Migrated E3 SHA256 matches original transfer; CPU strict state load succeeds, all 186,199,381 parameters loaded.
2. Same historical tracks evaluated with official TrackEval reproduce every common rounded summary within its precision. HOTA/CLEAR/Identity kernels are byte-identical to the recovered fork. Metric implementation is not the source of detector drift.
3. **Old detection caches replay through unchanged BYTETracker produce all seven historical MOT files byte-for-byte**, including scores/IDs. Saved at `historical_detection_replay_fp32`, verified by `historical_replay_parity.json`. Geometry, lifecycle, scores, formatting and frame order are consistent given identical detections. Drift is upstream of association.
4. Historical GPU model is also A100. Old Torch 2.0.0 CUDA 11.7 / TorchVision 0.15.0 / cuDNN 8.5 vs inherited current Torch 2.0.0 CUDA 11.8 / TorchVision 0.15.1 / cuDNN 8.7. Controlled 128-frame comparison now confirms runtime-dependent drift: both preprocessed input batch hashes are identical, and restoring cu117/cuDNN 8.5 exactly matches all historical boxes, scores, offsets and frame IDs. Evidence: `runtime_probe_parity.json`. The cu118 probe differs on these same inputs. Full cu117 inference is running to verify this across every video. This identifies the changed runtime stack, not one individually isolated cuDNN or NMS component.

Diagnosis detail: although the detector runs FP16, `postprocess` concatenates `class_pred.float()`, promoting its outputs to float32 before BYTETracker. A diagnostic attempt to round cached tracker inputs back to float16 failed parity and was rejected; the canonical float32 replay reproduces exactly. Do not introduce that rounding into baselines or future experiments.

Repair status: full seven-video cu117 inference completed. All cache frame IDs, offsets, boxes and scores equal historical arrays exactly; all seven MOT files are byte-identical. All historical TrackEval results reproduce. `full_cu117_parity.json` records checks; runtime restoration took a full 1,092.98-second run. No tracker threshold changes were needed. No GO/NO_GO decision applies to H1/H2 yet.

## P0-02: malformed identity annotation

SN-013 has two duplicate frame/identity keys (967/4 and 1001/6), four ambiguous boxes in both original JSON and GT. Versioned protocol v2 excludes all four from identity samples; derived evaluation GT uses unique dummy IDs and official distractor class 8. Raw files remain intact. Actual official TrackEval validation ignores the four matching predictions as intended. Every new method must use the same derived GT. See `phase0/annotation_audit.json` and `gt_mask_validation`.

## P0-03: future information in constructed inputs

Recovered `trans-sn0612.py` takes median background from up to 80 uniformly sampled frames over the whole sequence. `trans-auto.py` also selects local/temporal strategy from that future-derived background, even for its local branch. Four source hashes match NAS; `trans-tir0611.py` outputs unified_v2, a fixed single-frame version. Original mixed inputs cannot establish a strictly online full pipeline. Separate causal EMA generation computes responses before background updates and uses no sequence-dependent future strategy. Future perturbation, sequence reset, original stateless adapter parity and temporal pixel-equivalence tests pass. Full causal caches and equivalent RGB-history controls remain necessary.

## I-01: weak and inconsistent frozen identity evidence

Frozen ImageNet ResNet50 legacy-input diagnosis is complete. Equal fusion gains 2.857pp retrieval on BT-005 but loses 3.571pp on SN-013. RGB pair AUC is only 0.521/0.586; domain/identity training is required before a utility-module claim. Do not choose a scene-specific IR version using these outcomes. No fish ReID repair round has occurred yet.

## Infrastructure repair

Official dataset loader still uses removed NumPy float/int aliases; process-local builtin alias compatibility fixed the first evaluation error, with upstream source unchanged. Initial pytest import missed pluggy because dependencies were deliberately installed without global resolution; installed lightweight test dependencies in the project venv only. An installed `tools` package shadows the repository namespace; the diagnostic test loads the CLI by its exact file path. Twenty-four tests pass. PyWavelets 1.6.0 was added without dependencies only to the project venv for recovered local-generation controls. No other environment modified.

## Future H1/H2 diagnoses

Not run. Each core hypothesis gets at most three targeted evidence-driven repair rounds. Separate feature separability, crop shift, candidate recall, utility calibration, temporal collapse and memory contamination. Store per-video outcomes and events; use training/dev only for research diagnosis and historical test only for reproduction/frozen reporting.
