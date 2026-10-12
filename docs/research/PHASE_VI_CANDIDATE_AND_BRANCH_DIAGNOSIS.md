# Phase VI candidate-prediction and branch evidence update

Status: IN_PROGRESS. The first controlled fits and branching trials are complete; diagnoses below motivate new discriminating experiments, not a project-level stop.

## Candidate-conditioned prediction: first controlled fits

All five models use four fit videos, BT-001 calibration, SN-015 outer validation, fixed final epoch8/64 steps/batch64. Attention ordinary/noEMA/EMA have exactly 109,186 parameters, byte-identical initial tensor content hashes and all eight query orders identical. EMA has 512 successful updates and no teacher gradients. Geometry/confidence/current crowding enter the network; cached full-history similarity/cost features are zeroed so masked evidence cannot leak. Ordinary GRU/MLP candidate variants allocate 117,506/92,546 parameters and are architecture baselines, not exact-capacity EMA interventions.

| Model | Known-positive Rank@1 % | Local N01 | Local N10 | Net | NULL correct / 1087 |
|---|---:|---:|---:|---:|---:|
| ordinary_candidate_attention | 46.4426 | 130 | 1645 | -1515 | 6 |
| candidate_jepa_ema | 39.6172 | 198 | 2404 | -2206 | 4 |
| candidate_jepa_noema | 42.1475 | 210 | 2167 | -1957 | 3 |
| ordinary_candidate_gru | 47.9178 | 58 | 1510 | -1452 | 4 |
| ordinary_candidate_mlp | 54.4631 | 52 | 789 | -737 | 19 |

The 10,710 known-positive examples and 1,087 known NULL examples do not include all unknown candidate-missing cases. Unknown newly predicted candidates are reported separately, not called wrong. Rank is evaluated on original fixed states, not a full new trajectory. Ordinary MLP is strongest among these five; EMA does not improve its identical ordinary attention/noEMA controls. Prediction variance is nonzero, so this is not evidence of exact constant collapse. Lower latent loss does not demonstrate better association.

Observed failure / alternatives: the learned 512->64 identity coordinates may discard the frozen raw-space baseline; candidate-conditioned latent prediction can regularize toward shared identities rather than difficult discrimination; the fit data provide very little reliable NULL supervision; warm-start actual-state shifts may remain. The four fit videos have only128 known-NULL examples (BT0030,MSK0020,PF00159,SN00169), compared with1087 on outer SN015. Unknowns cannot be converted to negatives to fix this imbalance.

Discriminating experiments proposed: raw-space mean/ordinary-GRU ranking controls on exactly these queries; bounded residual scores that preserve raw identity similarity; separately controlled known-negative-only synthetic candidate-missing training and NULL-relative score calibration; video-excluded ordinary teacher generation before genuine OOF claims. No architecture is promoted solely by a best seed or these local metrics.

## Independent K1/K2 and simple protected writing

Full six-train-video replay and official TrackEval (all18 metrics and per-video values in PHASE_VI_BRANCH_FIRST_MEASURED.json). This train aggregate is NOT directly comparable to exposed two-dev HOTA53.591: it includes difficult dense scenes and different videos. K1 disabled outputs are byte-identical to preserved train current-measurement files.

| Model | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| baseline_K1 | 32.890868 | 18.836977 | 58.052536 | 34.932436 | 69.662663 | 679.000000 |
| ambiguity_K2 | 32.733793 | 18.671661 | 58.018170 | 34.533280 | 69.615241 | 785.000000 |
| protected_write_K1 | 32.659095 | 18.569909 | 58.022038 | 34.996441 | 69.623068 | 693.000000 |

The branch implementation is real: near-tied pair assignments produce independent mutable histories/motion/lifecycle, up to K2; the frozen predictor is shared read-only. Outputs are streamed for current frames and never rewritten. The cost accumulator is a heuristic, not a posterior. The simple protected-write control defers ambiguous/low-confidence memory retention; it is not an offline smoothing method or fixed two-frame confirmation queue.

Observed failure: K2 increases IDSW679->785 and slightly decreases identity metrics; even a single expansion in BT003 produces15 branch selections over time. Root alternatives: marginal branch-score flicker, self-reinforcing wrong branch memories, or inadequate candidate scores. Discriminating experiments registered: current/past score-margin hysteresis (.02) and two-consecutive-frame challenger confirmation, each preserving v1 scorer/data/discount and keeping two distinct states. These run in separate immutable outputs; the v1 negative result remains. Ordinary delaying/protecting memory remains required as a control, not assumed inferior.

## Joint recoverability oracle and episode scope

Offline training-only joint assignment preserves unknown-reference allocations, uses dedicated NULL columns, maximizes known correct assignments and preserves already correct ties. It uses existing proposals, never GT-added detections; stage/full graph versions ignore original cost thresholds deliberately. This is a local diagnostic intervention on frozen states, NOT a whole-tracker/HOTA or information-theoretic ceiling.

| Video | Joint stage N01 | Joint expanded N01 | Consecutive known-wrong episodes | Wrong frame/track rows |
|---|---:|---:|---:|---:|
| BT-001 | 3413 | 3704 | 697 | 9094 |
| BT-003 | 2 | 7 | 14 | 160 |
| MSK-002 | 7189 | 9416 | 1516 | 22869 |
| PF-001 | 4658 | 4667 | 118 | 11480 |
| SN-001 | 2093 | 2197 | 148 | 3379 |
| SN-015 | 5712 | 6010 | 484 | 24621 |

All oracle N10 counts are0 under its preserve-correct tie policy; rejected wrong assignments are separate wrong-to-abstention counts, not successful corrections. One correct proposal cannot repair multiple duplicate-reference tracks simultaneously. For SN015,24621 known-wrong rows correspond to484 operational episodes, not24621 official ID switches. Episodes use consecutive known-wrong rows; gaps/unknowns break the convention. Newborn/termination causal contributions and inference-available cue absence remain unproven.

All current negative results remain registered. Next work addresses which scoring/NULL/data/branch-choice explanations are supported; no fixed iteration count, threshold54 or A/B failure cancels C. Genuine heldout association training, remaining ablations, promising-scheme seed expansion and complete end-to-end online measurement are still outstanding.
