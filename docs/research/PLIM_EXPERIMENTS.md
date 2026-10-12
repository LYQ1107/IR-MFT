# PLIM experiments

Status: FIRST_THREE_SEED_MODELS_TRAINED / FIRST_VALIDATION_COMPLETE / THIRD_REPAIR_PENDING, 2026-10-12. Twelve RGB predictors across seeds0/1/2 and four RGB actual-history round2 models are trained; IR and paired RGB/IR prediction remain outstanding. No final method gain is claimed.

The first frozen trial uses original-crop main post-neck512 targets, eight past observations, gaps1/5/15/30, eight fixed epochs, GRU231,040 parameters and Transformer230,144 parameters. History tensors and current targets are stopped; timestamps are strictly increasing and earlier than query time. Residual output starts as exactly last-observation prediction. Empty histories fall back. Cosine forecasting plus same-frame candidate CE is compared against identical architectures with candidate CE alone, and last/bounded EMA/FIFO/all-history EMA/constant controls. Every family receives the same geometry candidate graph and three appearance settings. No predicted-distance eligibility gate is introduced without these controls.

Query construction replays RGB tracking without GT-dependent decisions. Independent current-frame physical GT labels annotate targets, hard negatives and retrospective purity only; they never change tracker outputs. There are30,811 train queries (19,545 fit;11,266 SN-015 holdout) and8,293 dev queries. Main encoders trained on SN-015, so this is a temporal query holdout only. Frozen crossfold latent coordinate spaces are not mixed. The training hard-negative pool includes all labelled current fish; deployment has a geometry gate, a disclosed difference.

## First fixed-model diagnoses

On SN-015, clean-trained predictive GRU gives91.186% candidate Rank@1 with clean histories and57.394% with actual histories, below FIFO92.872%/58.370%. Variance sums0.882/0.885 exclude a constant prototype. On dev BT-005, GRU improves actual-history rank53.591%→57.122% over last observation, but ordinary GRU is slightly stronger; SN-013 favors simple memories. These are supervised query diagnostics, not tracking IDF1. Four-model results, hashes and a static figure are registered.

## Firstseed0 full-video results

All30 comparisons are complete. At appearanceweight .3:

| Model/control | HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| Predictive GRU |51.4971|37.0722|61.7256|84|
| Association-only GRU |51.4100|36.9519|61.7237|85|
| Predictive Transformer |51.7256|37.4644|61.9316|84|
| Association-only Transformer |49.7624|34.6541|59.0992|88|
| Bounded EMA |50.1899|35.2579|60.4104|89|
| All-history EMA |50.3775|35.5383|59.2464|86|
| Constant distance .5 |50.4945|35.7435|61.0240|87|

The constant-zero control selects .1 at HOTA50.7595/IDSW84, reproducing the previous geometry effect. Predictive Transformer merits three-seed validation; GRU forecasting's independent difference is much smaller. Native BoT mean52.4818 still has higher HOTA, with lower IDF1/more switches. Seeds1/2 full grids are complete; do not compare a winning seed as a final mean. Complete per-video/official metrics: `phase3_seed0_tracking_results_20261012.json`.

## Independent write refusal

`ReliableWritesMixin` computes confidence and row/column margins before any current write, then allows or refuses identity writes after unchanged geometry updates. Threshold .68 matches native birth confidence; .02 margins are fixed without dev search. Low/unscored existing writes are refused; confident newborns initialize normally. No GT is loaded. Disabled unit traces and all three full-video counterparts are byte-identical to their originals.

At fixed .3 and identical weights, refusal helps EMA HOTA50.1899→51.1402, but harms predictive GRU51.4971→48.9655 (IDSW84→95) and Transformer51.7256→49.7085 (84→90). The policy is not retained as a proven improvement. Transformer write precision changes BT0.8099→0.7354 and SN0.7171→0.7631, with unknown/initial-anchor samples excluded. These are within-run audits; trajectories differ, so they are not paired causal correction counts.

Memory ages reach68BT/127SN versus36/38 ordinary, outside trained gap30, but only about .5% of refused-policy attempts exceed30. This is a candidate cause, not a proven sole explanation. Exact CLEAR traces show10new/5removed BT switch keys and3new/2removed SN keys. At BT frame1049, GT3/7 switch under refusal while ordinary prediction retains identities; both remain matched. [Verified case figure](/data3/liuyeqiang/IR-MFT_assets/cfp/phase3/safe_write_switch_figures_v1/BT-005_frame1049_gt3.png). Full results: `phase3_write_policy_results_20261012.json`.

## Actual-history training round2

Four models use identical original queries, architectures, stopped teachers and budgets; only historical tensors change from clean to actual accepted observations. Fully contaminated rows remain and no purity mask enters inference. On SN-015 actual-history rank, predictive GRU falls57.394%→40.991%; predictive Transformer gives43.991%, association-only GRU37.236% and Transformer40.698%. Its twelve video comparisons are complete.

A bounded history containing only the wrong fish may have no observable cue for the original identity target. The third targeted repair must address discarded identity context or harmful residuals with matched memory/models, rather than add capacity or epochs. Long-term observable anchors, RGB/IR variants, combined CMUE, true online cost and final evaluation remain outstanding. The goal remains active.


Three-seed first-round tracking is now complete (90 runs). At each family's selected mean-HOTA setting: predictive GRU51.3199±0.1774, ordinary GRU51.4567±0.2952, predictive Transformer51.1208±0.7566, ordinary Transformer50.3063±1.0017. Thus forecasting is not consistently superior to matched ordinary temporal models; neither beats native BoT52.4818±0.1341. Selection uses the mean across seeds, not the favorable seed0 result. Full setting/per-video metrics are indexed by `configs/cfp/plim_first_three_seed_selection_v1.json`.

Actual-history round2's twelve seed0 video runs are also complete: predictive GRU at.3 gives HOTA51.7572/AssA37.5118/IDF161.9677/IDSW85; ordinary GRU51.3927/36.9161/61.3822/82. Predictive Transformer49.5317/34.3263/59.5048/88 versus ordinary49.0639/33.6910/58.9340/89. Poor offline ranking and full-video scores have different behavior; neither substitutes for the other. These are single-seed diagnoses, not final selected means.


## Completed third repair: observable anchor plus recent history

All twelve fixed-epoch8 models across seeds0/1/2 and all99 full-video comparisons are complete. The first actually accepted available visual observation is retained without GT selection; the input remains at most8 tokens (anchor plus latest7). Native removed/lost object reactivation required retaining the anchor while the object can still match; regression tests cover that lifecycle boundary. The failed initial build is retained separately; valid runs use the repaired query manifests.

At each family's selected mean-HOTA setting (all weight.3): predictive GRU51.4769±0.8695, association-only GRU51.8853±0.9933, predictive Transformer50.6058±2.2142 and association-only Transformer50.6112±1.1680. Seed0's Transformer53.0551 is not stable: seeds1/2 give50.0163/48.7460 at the same weight. Ordinary anchor GRU's IDF162.2983 improves over ByteTrack58.8771, but its IDSW89 exceeds86 and its HOTA remains below BoT52.4818. Forecasting is not independently established by this repair.

Exact settings/full scalar metrics/three-seed sample deviations are in `configs/cfp/plim_anchor_three_seed_selection_v3.json`; all99 runs are registered with source snapshots and original commands. Full current-frame GRU trials and frozen utility/write composition are now running. These remain development diagnoses; no final-test freeze or completed-goal claim exists.


## Complete planned modality prediction controls

Eighteen frozen GRU models and54 full-video runs across IR-only512, RGBIR1024 and twoRGB1024 views are complete. Queries, anchor8, clipped timestamps, eight epochs and three appearance settings are unchanged. PairedRGBIR and pairedTwoRGB have identical362,624 trainable predictor parameters; IR-only231,040. TwoRGB reuses the next existing mainRGB seed consistently on train/dev; cyclic frozen encoder overlap between capacity-control pipelines is disclosed, so seedSD is descriptive rather than independent-pipeline confidence. No new feature/imagebanks are saved.

Three-seed selected means (all weight.3): IRpredict50.4247±0.9583 versus IRordinary49.7862±0.9607; RGBIRpredict51.8846±1.3844 versus pairedordinary49.4645±0.7176; twoRGBpredict51.6723±1.1147 versus twoRGBordinary50.6212±1.7474. Prediction helps the paired representation, but RGBIR's0.2123pp HOTA difference over twoRGB is not stable IR-specific evidence, and ordinary singleRGBanchor51.8853 essentially matches it at lower capacity. None beats BoT52.4818. All per-video/current-proposal outcomes are registered in `phase3_modality_prediction_measured_20261012.json`.

The single bounded log-age repair is complete:12 models/36 video runs. Predictive GRU51.3589±0.6283, ordinary GRU51.6848±1.8983, predictiveTransformer50.2843±1.0714, ordinaryTransformer49.5120±0.7330. Fixing timestamp saturation alone does not improve tracking, so this variant is not retained.
