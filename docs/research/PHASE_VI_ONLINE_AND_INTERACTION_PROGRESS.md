# Phase VI: complete online risk verification and interaction controls

Status: IN_PROGRESS, 2026-10-13. The final scientific synthesis awaits complete residual repetitions and same-fold combination evaluations. Historical dev references remain exposed and are not a blind test.

The canonical seed0 risk MLP finished all 4,834 raw frames with causal IR generation, V3/NMS, RGB crops/encoding, ordinary GRU, matched 10% risk write quota, association and current measurement output. Official combined HOTA is 53.1035, AssA 38.3609, DetA 73.5676, IDF1 62.1078, MOTA 93.6700 and IDSW 87. Both dev videos preserve the ordinary cached frame/ID outputs: this deployment run does not demonstrate an identity improvement.

| Exposed reference | HOTA | IDF1 | IDSW | Online/cache comparison |
|---|---:|---:|---:|---|
| BT-005 | 59.9983 | 74.5622 | 21 | All 18,204 rows byte-identical |
| SN-013 | 48.1678 | 53.9370 | 66 | Same frame/IDs and risk/quota counters; seven numeric rows differ only at frames2999/3000 |

The final detector batch has two canonical frames but device padding repeats each current frame eight times. Those seven rows differ by at most0.338pixels/score0.000952. Do not claim full byte equality. Independently hashed full predictions, official metrics and numeric difference vectors are retained in PHASE_VI_RISK_ONLINE_EVIDENCE.json.

Full online time is3907.5533seconds,1.2371FPS,p50739.5ms/p951116.8ms. Parent/worker peak allocated GPU memory is652,593,664/1,947,126,784bytes; these are separate process peaks. Parameters: ordinary231,040,risk1,793,RGB encoder24,607,808,detector186,199,381. Countable affine risk MLP cost is1,728MACs or3,456multiply/add FLOPs per scored edge, excluding risk feature extraction and all other pipeline components. Timing is shared-load measurement and does not establish a controlled speedup.

54 ordinary/noEMA/EMA residual initialization/utility fits are now complete. All final8-epoch models retain fixed scale0.05 and backbone/producer0. Full outer-video replay and official evaluation are still running; ranking/loss results are not a tracking gain claim. The actual residual epoch3 stop/resume proof matches every model/EMA/optimizer/RNG/order/loss/guard field exactly, excluding only Git provenance (PHASE_VI_RESIDUAL_RESUME_PROOF.json).

The same-fold B0/A/B/C/AB/AC/BC/ABC matrix reuses12 immutable A/B controls and executes36 new whole-video replays, including six all-disabled baseline byte checks. A=MLP write quota10%, B=bounded EMA representative (EMA-specific usefulness unproven), C=K2 with fixed0.02 hysteresis. In AB/ABC the risk estimator sees B-corrected current costs; this deliberate interaction is disclosed. Nested quota/alternate assignment, history/evidence/counter isolation and zero-residual behavior passed integration tests. The full project suite passed186tests with the migrated dataset root, including the new combination and explanation-label guards.

A complete BT001 B+C explanation trace contains9,569 known-wrong winning frame/track decisions; only30 have a known-correct retained challenger.14,243 unknown decisions remain excluded;697 consecutive known-wrong episodes are not official IDSW. This is a retrospective birth-reference diagnostic of internal explanations before area output filtering, not proof of future recoveries. The birth audit detected no emitted reuse across distinct birth observations. Other scenes and final MOT outcomes remain pending.

Reproduce online evidence with tools/cfp_collect_phase_vi_online_evidence.py. Execute the frozen interaction plan with tools/cfp_run_oof_combination_campaign.py and aggregate/evaluate with tools/cfp_evaluate_oof_families.py. Physical labels are only loaded by the separate retrospective diagnostic; the causal tracker never receives GT/future/teacher inputs.
