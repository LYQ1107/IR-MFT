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
4. Historical GPU model is also A100. Old Torch 2.0.0 CUDA 11.7 / TorchVision 0.15.0 / cuDNN 8.5 vs inherited current Torch 2.0.0 CUDA 11.8 / TorchVision 0.15.1 / cuDNN 8.7. Controlled 128-frame comparison now confirms runtime-dependent drift: both preprocessed input batch hashes are identical, and restoring cu117/cuDNN 8.5 exactly matches all historical boxes, scores, offsets and frame IDs. Evidence: `runtime_probe_parity.json`. The cu118 probe differs on these same inputs. Full cu117 inference now confirms parity across every video. This identifies the changed runtime stack, not one individually isolated cuDNN or NMS component.

Diagnosis detail: although the detector runs FP16, `postprocess` concatenates `class_pred.float()`, promoting its outputs to float32 before BYTETracker. A diagnostic attempt to round cached tracker inputs back to float16 failed parity and was rejected; the canonical float32 replay reproduces exactly. Do not introduce that rounding into baselines or future experiments.

Repair status: full seven-video cu117 inference completed. All cache frame IDs, offsets, boxes and scores equal historical arrays exactly; all seven MOT files are byte-identical. All historical TrackEval results reproduce. `full_cu117_parity.json` records checks; runtime restoration took a full 1,092.98-second run. No tracker threshold changes were needed. No GO/NO_GO decision applies to H1/H2 yet.

## P0-02: malformed identity annotation

SN-013 has two duplicate frame/identity keys (967/4 and 1001/6), four ambiguous boxes in both original JSON and GT. Versioned protocol v2 excludes all four from identity samples; derived evaluation GT uses unique dummy IDs and official distractor class 8. Raw files remain intact. Actual official TrackEval validation ignores the four matching predictions as intended. Every new method must use the same derived GT. See `phase0/annotation_audit.json` and `gt_mask_validation`.

## P0-03: future information in constructed inputs

Recovered `trans-sn0612.py` takes median background from up to 80 uniformly sampled frames over the whole sequence. `trans-auto.py` also selects local/temporal strategy from that future-derived background, even for its local branch. Four source hashes match NAS; `trans-tir0611.py` outputs unified_v2, a fixed single-frame version. Original mixed inputs cannot establish a strictly online full pipeline. Separate causal EMA generation computes responses before background updates and uses no sequence-dependent future strategy. Future perturbation, sequence reset, original stateless adapter parity and temporal pixel-equivalence tests pass. Full causal caches and equivalent RGB-history input datasets are complete; matched control training/comparison remains necessary.

## I-01: weak and inconsistent frozen identity evidence

Frozen ImageNet ResNet50 legacy-input diagnosis is complete. Equal fusion gains 2.857pp retrieval on BT-005 but loses 3.571pp on SN-013. RGB pair AUC is only 0.521/0.586; domain/identity training is required before a utility-module claim. Do not choose a scene-specific IR version using these outcomes. Three-seed fish training and paired causal diagnosis are now complete; mean IR and mean fixed-fusion retrieval remain below RGB. Per-query corrections/harms are retained.

## I-02: narrow metric cone and identity-insensitive association

Pre-neck fish features are normalized but concentrated: BT-005 proposal feature variance sums to about 0.01, and first-ten proposal pair cosines exceed 0.98. Final RGB seed 0 classifier accuracy is 90.90%, while batch-hard triplet remains 0.3125 (margin 0.3). Default .3 appearance weight gives HOTA 47.0802 / AssA 31.3048 / IDF1 55.9773 / IDSW 98, below ByteTrack 49.2967 / 34.0115 / 58.8771 / 86. Different RGB/IR/fusion inputs and seeds produce byte-identical tracks. The constant identical-vector control at .3 also reproduces these tracks exactly. The .1 configurations give 50.7595 HOTA but identical modality trajectories, so no IR benefit is established.

Hypothesis: small absolute cosine distances reduce geometry costs and change acceptance without meaningful identity ranking. First repair: use the existing trained BN neck as the normalized inference metric, with unchanged frozen weights, candidate rules and all-seed comparison. Declared in `reid_repair_v1.json` before post-neck extraction. Further repair must depend on candidate/crop/memory evidence, not arbitrary depth or test tuning. Identity-blind controls cover all declared weights. This is baseline diagnosis, not a rejection or validation of CMUE/PLIM.

## External association compatibility

First HybridSORT full replay fails when its prediction returns singleton score arrays placed beside scalars in a NumPy row; NumPy 1.24 rejects that sequence. Failure log/output remain at `phase1/causal_dev_hybrid_sort_seed0_c2`. A process-local wrapper converts only finite singleton scores to Python float, preserving values; multi-value/nonfinite scores are rejected. Both full dev videos then complete, HOTA 46.3243 / IDSW 141 at declared upstream high/low appearance 4/4.4. No vendor file changed. BoT-SORT completes via an explicit shared-feature provider with exact native confidence/filter order verification, HOTA 49.7996 / IDSW 112 at default .25. All-seed/search comparisons are now complete; see the full development selections in ABLATION_RESULTS.md.

## Infrastructure repair

Official dataset loader still uses removed NumPy float/int aliases; process-local builtin alias compatibility fixed the first evaluation error, with upstream source unchanged. Initial pytest import missed pluggy because dependencies were deliberately installed without global resolution; installed lightweight test dependencies in the project venv only. An installed `tools` package shadows the repository namespace; tests load CLIs by exact file paths. Tests now cover external filter/order checking and representation-stage invariance in addition to causality/fallback. PyWavelets 1.6.0 was added without dependencies only to the project venv for recovered local-generation controls. No other environment modified. A new paired-feature composition CLI's unmatched parenthesis failed before outputs were created; syntax was corrected, compileall passed and all three paired diagnoses completed, with the failed log retained.

## H1/H2 diagnosis contract

H1 has three completed seed0 targeted rounds; H2 has completed firstseed0 tracking, independent write refusal andactual-history round2 training. Each core hypothesis gets at most three targeted evidence-driven repair rounds. Separate feature separability, crop shift, candidate recall, utility calibration, temporal collapse and memory contamination. Store per-video outcomes and events; use training/dev only for research diagnosis and historical test only for reproduction/frozen reporting.

## I-03: completed cone repair and residual association errors

Post-neck features remove the narrow cone without new SGD. All 27 ordinary and 18 external post-neck runs complete. Best BoT-SORT mean HOTA is 52.4818±0.1341 and AssA 37.4980±0.1653, versus ByteTrack 49.2967/34.0115. However IDF1 decreases to 57.4768±0.5930 and IDSW increases to 113.6667±4.5092. The repair is useful to HOTA but does not fix persistent identity. Fixed fusion's selected mean HOTA is 49.0889±0.5450, below the constant geometry control. No constructed-IR contribution is validated. Candidate/lifecycle/memory contamination remain distinct possible causes; do not retrain the detector for these association errors.

Exact upstream CLEAR observer (unchanged official matching) records all switches and matched rows. Constant .1 gating removes two ByteTrack switches, both BT-005 frame 1049, GT 3 and 7. GT 3 changes ByteTrack ID 13→2, GT 7 changes 2→13; constant control keeps 13/2. Both are officially matched at t and t+25 with IoU >.70, not dropped predictions. All other current-frame/GT switch keys are shared. This specific identity recovery is explained by geometry alone. CLEAR matching is continuity-biased retrospective evaluation and must not become physical proposal-label supervision for CMUE.

A second baseline repair is frozen before outcomes: current-ROI long-axis alignment addresses measured letterbox resolution waste; no model-depth or epoch-budget increase. Declare and quantify orientation-transition risk and re-run matching controls if retained. Original crops, models and metrics are preserved.

Real CUDA ResNet/AdamW/AMP resume proof passes: a dedicated synthetic fixture interrupted after epoch 1 resumes to bit-identical final model, optimizer and RNG states versus uninterrupted training. This is infrastructure evidence, not a fish-method result. No unrelated process was interrupted.

The first long-axis IR cache call rejected the canonical `temporal_ema` generator as an unknown auxiliary mode. No output was created. Its mode is now explicitly classified as causal constructed IR; RGB controls retain distinct labels. Both complete aligned caches are finite and paired. Initial post-neck capacity campaign stdout filenames also collided when `Path.with_suffix` stripped decimal weights; per-run manifests/predictions/metric files were intact. Filename handling now appends `.log`. All nine original prediction files were re-evaluated with official TrackEval, every metric verified equal, and separate verification logs/registry provenance preserve that limitation.


## II-01: utility learns rejection penalty without established identity benefit

Round1 local-candidate supervision yields near-chance calibration sign AUC0.5066. Its best seed0 HOTA50.9769 is only0.0241pp above an identity-ranking-free constant IR distance control, with worse IDSW86 vs85. Dev loss decomposition attributes most positive target utility to NULL penalties while genuine candidate-ranking mean utility is negative. Do not claim these gains are counterfactual identity selection.

Independent candidate audit finds148 dev queries (88BT/60SN) where a correct geometry-matchable proposal existed outside the appearance-modifiable local subset. Round2 preserves the complete native geometry candidate context in both teachers and model evidence. Sign AUC remains0.5082 and best seed0 HOTA49.3071, below ordinary/quality50.3556. Smaller target MSE is partly a change of target scale and not proof of sign learning. Preserve both rounds; any third repair must follow this evidence. Native NULL rejection/lifecycle remains unlearned.

Accepted utility rows originally risked FP32-rounding RGB costs on individual unavailable IR edges. A tested edge-specific update now preserves those unchanged costs exactly, including when another edge has available IR. Actual completed raw source snapshots remain attached; no old metric is overwritten.

## III-01: actual accepted memory contamination

PLIM query construction retains sticky earliest physically known past identity anchors while tracker decisions remain GT-free. Early original-train BT-001/MSK-002 histories have mean purity about0.49/0.47 and fully contaminated fractions about0.43/0.48. Clean teacher-forced histories and actual accepted histories are stored separately. This motivates separating predictor accuracy from write contamination, not attributing all failures to temporal architecture. These are preliminary complete-video query observations, not learned predictor outcomes.

A broad test run passed63 tests and encountered one original pairing test's old default server path. Its existing MFT26IR_ROOT environment setting was switched to the migrated original dataset and that test passed without altering original code. Current suite invocation declares this root.

First four seed0 temporal models completed eight epochs. All remain below FIFO on SN-015 query holdout; GRU predictive variance sum0.882/0.885 rules out trivial constant collapse. On canonical-dev BT-005 both predictive and ordinary GRU beat last in frozen query rank1, and ordinary GRU is slightly stronger; SN-013 favors simple memories. Thus a scene-specific temporal improvement is not yet independent forecasting evidence. The shared geometry-candidate graph is deliberately measured against all-history EMA and constant-distance controls, and full tracking is underway. Torch2.0 Transformer evaluation reports an internal float-mask-to-bool performance warning; masks contain only0/-inf and strict-past inputs remain guarded. It is not a measured speed or model-gain claim.


## I-04: crop coverage did not predict tracking gain

Six models/54 runs complete the frozen rotation repair. RGB48.9386 and fusion48.6361 mean HOTA are below original49.2917/49.0889. Rectangle coverage is not foreground or identity value. Discrete aspect-triggered rotation is a possible degradation factor, not an isolated proved cause. The variant is not retained; all originals/results are preserved.

## II-02: ranking/sign supervision remains unsupported

The third fixed-capacity trial removes NULL penalties and balances physical decision types equally for all gates. Sign supervision only affects CMUE loss; utility sign never controls sampling. Calibration AUC .4047/correlation .0400 and dev49.3071 below ordinary50.3556 do not establish benefit. Smaller MSE under a new target scale is not validation. These bounded experiments do not imply a universal claim about RGB preprocessing; encoder/domain limitations and two-video development remain disclosed.

## III-02: write refusal can indirectly damage prediction

Refusal leaves current geometry updates intact but reduces Transformer HOTA51.7256→49.7085 and increases IDSW84→90. Exact counts change BT19→24 and SN65→66. Verified new crossing1049 switches for GT3/7 persist with matched predictions; ordinary writes avoid them. Feature gaps reach127 versus38 frames, outside trained30, but only about .5% of attempts exceed30. No sole-cause claim is made. Physical purity effects differ by video; all logs/parity/cases are kept.

Actual-history training gives predictive GRU40.991%/Transformer43.991% on the SN-015 candidate query holdout versus last57.820%/FIFO58.370%. Fully wrong short histories can lack the original physical identity cue, producing contradictory supervised targets. A third repair should preserve observable context or bound damaging predictions, with matched controls. No larger-network or test-tuning remedy is justified by these observations.


Third anchor repair does not validate forecasting independently: predictive Transformer53.055 seed0 becomes50.606±2.214 across seeds, and ordinary anchor GRU51.885 exceeds predictive51.477. Training-only physical-anchor age audit identifies77–99% saturation of the clipped timestamp mapping and unknown initial proposal identities; unknown is not evidence of a wrong fish. One frozen log-time repair tests age encoding without changing capacity/queries/epochs/candidates. Joint CMUE accepts no rows and leaves PLIM unchanged; fixed/ordinary IR mixtures and the independent write-refusal policy worsen the measured seed0 anchor system. See ONLINE_INTEGRATION.md and machine records.
