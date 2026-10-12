# Phase VI actual-error recoverability diagnosis

Status: VI1_LOCAL_DIAGNOSIS_COMPLETE; overall Phase VI IN_PROGRESS. Complete six-train-video and exposed two-dev-video seed0 actual traces were reclassified offline. The fixed reference is >=3 distinct earliest known past accepted observations with >=.6 consensus, bounded first five. A rolling modal-reference sensitivity is retained separately. Neither is an online label or unconditional biological identity truth.

Counts below are final stage-deduplicated frame/track decisions, NOT IDSW or distinct recovery episodes. Persistent swapped tracks contribute many rows. Wrong requires both a known reference and known assigned physical identity. Unknown labels are never wrong.

| Video | Wrong rows | Local reranking feasible at old threshold | Correct stage candidate but rescoring/acceptance needed | Current stage missing/full pool present | Detector absent or label unverified | Reference not visible | Known correct past cue | Cue unverified | All-known past lacks cue | Wrong reactivation | Unmatched visible/stage-covered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BT-001 (train) | 9094 | 547 | 5213 | 609 | 2725 | 0 | 5134 | 3960 | 0 | 14 | 533 |
| BT-003 (train) | 160 | 8 | 122 | 5 | 25 | 0 | 157 | 3 | 0 | 0 | 8 |
| MSK-002 (train) | 22869 | 48 | 11299 | 3706 | 7816 | 0 | 19373 | 3459 | 37 | 41 | 1923 |
| PF-001 (train) | 11480 | 5816 | 5287 | 27 | 350 | 0 | 9077 | 2403 | 0 | 2 | 163 |
| SN-001 (train) | 3379 | 162 | 2521 | 176 | 508 | 12 | 2928 | 451 | 0 | 8 | 175 |
| SN-015 (train) | 24621 | 2787 | 14692 | 729 | 1828 | 4585 | 13852 | 10769 | 0 | 24 | 1944 |
| BT-005 (historical dev) | 3328 | 371 | 2572 | 21 | 364 | 0 | 3321 | 7 | 0 | 5 | 183 |
| SN-013 (historical dev) | 7323 | 632 | 5297 | 285 | 597 | 512 | 3276 | 4047 | 0 | 12 | 546 |

The first five opportunity columns after wrong rows partition known wrong decisions under this local reference. Past-cue, reactivation, contamination and competition counts overlap these categories. A correct column often belongs to another output track; independent reranking cannot necessarily improve the global one-to-one assignment. These are scoped intervention opportunities, not a measured HOTA ceiling.

The original GRU geometry eligibility controls appearance blending, not a hard association mask. Even raw distance>.9 can remain Hungarian-feasible by geometry cost<=.97. The analyzer reports blending eligibility, score threshold feasibility, full proposal/stage coverage and history-mean appearance top1/2/3/5 separately. Geometry/top3 union is a diagnostic candidate proposal, not a tested new online rule. Top-K recall is relative to the observable history mean, not a GT-clean or teacher future embedding.

A known correct past token is retrospective annotation evidence. A no-correct-token context containing unknown labels remains unverified, not proven unrecoverable. Risk estimates at inference have no physical labels. The two-axis correct-stage-candidate x retrospectively known-correct-past-cue distributions and margin/separability strata are in the measured JSON.

Observed failure: many errors persist despite a current-stage positive; most need score/acceptance repair rather than merely another ranking among already-feasible positives. Root-cause alternatives: motion state drift after a swap, weak current-vs-history discrimination, competition for occupied correct columns, or contaminated memory. Discriminating experiments: same-input MLP vs set-context risk; original-score risk gates vs candidate-conditioned rescoring; delayed memory vs isolated K=2 hypotheses; expanded appearance blending only if video-heldout evidence supports it. No HOTA/AssA gain is claimed at this stage.

Lifecycle scope: wrong reactivations and visible-target unmatched proxies are measured; newborn/duplicate/termination-specific causal contributions and sustained recovery episodes still need separate full-sequence analysis. Do not treat this diagnostic as completed proof of every lifecycle cause.

Historical BT-005/SN-013 are descriptive references, not architecture search criteria. Next: registered six outer whole-video folds with separate inner calibration video, comparing context risk to ordinary same-input MLP, on actual teacher states. This is new-module heldout evidence, not genuine whole-tracker OOF; regenerate legal video-excluded ordinary states before OOF claims. Frozen V3/FishReID exposures remain.
