# Phase VI: action, memory, state shift and relation targets

Status: IN_PROGRESS. All tracking numbers below use the six training videos and seed0, with official TrackEval and identical frozen proposals. They do not replace the exposed two-video three-seed baseline HOTA53.591. Frozen detector/encoder and original GRU exposure remains explicit.

| Action | HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| Preserved ordinary GRU K1 | 32.8909 | 18.8370 | 34.9324 | 679 |
| Context risk, memory writes only | 33.2667 | 19.2688 | 35.4979 | 677 |
| Ordinary MLP risk, memory writes only | 32.9266 | 18.8862 | 35.0227 | 680 |
| Constant reject existing identity writes | 32.1581 | 18.0214 | 34.3531 | 710 |
| K2 with causal margin hysteresis | 33.1236 | 19.1163 | 35.2235 | 683 |
| K2 with two-frame confirmation | 32.7319 | 18.6698 | 34.5305 | 748 |

Memory-only risk preserves the existing current-state assignment costs, Hungarian result, geometry and box output; subsequent changed histories can change future assignments. Newborns and unscored later-stage matches use original writes. The implementation currently supports K1; its closures must not be copied into K2. Five of six video HOTA values improve for context memory, but the gain is small and requires new producer states and multiple seeds. Constant rejection loses, so a universal write freeze does not explain this preliminary signal. Eighteen fixed-budget models now provide no-identity-history, learned-cost-threshold and calibrated-untrained controls; their full-video action evaluations are pending.

Direct context risk association scores produce HOTA22.6779/IDSW7284; rejecting high-risk associations produces HOTA12.6530/IDSW15669. Ordinary MLP alternatives also lose severely. Fixed margin yields30.8622, entropy6.5009 and confidence5.9661. Rejection creates fragmentation and many missed outputs. Discrimination and calibration alone do not justify replacing geometric assignment cost. Detailed official per-video tables and provenance are in PHASE_VI_RISK_ACTION_MEASURED.json.

Hysteresis reduces branch flicker but its aggregate gain mainly comes from SN015 (+.9676 HOTA); MSK002/SN001 are unchanged and three other videos lose slightly. This is a causal current/past intervention, with isolated states and no rewritten historical outputs, but does not yet demonstrate consistent multi-video benefit. Confirmation is not an effective substitute under the present heuristic score.

Raw-coordinate diagnosis separates pure identity retrieval from strong tracking. On SN015 known-positive actual-state queries, raw mean ranks62.3623%, while the frozen original GRU prototype alone ranks35.8450%. Original geometric/blended association can therefore be strong despite weak isolated prototype ranking. On the SAME8631 queries with verified retained GT-correct past tokens, raw mean ranks76.9899%; offline selection of only the actual retained correct tokens ranks83.4897%. This GT-only diagnostic adds neither synthetic clean histories nor future evidence, and is not an online bound or a tracking improvement. Other queries with no verified correct cue can contain unknown tokens; they are not proved irrecoverable.

All six new ordinary GRU producers now exclude an entire outer video and distinct inner calibration video from fitting. Their clean bootstrap supervision is separate from their subsequent causal six-video accepted-history replay. These actual replays supply new candidate graphs. Backbone pretraining still includes these videos, so this is conditional association OOF, not blind whole-system validation. First3+recent5 history views use actual unfiltered observations; their first3 tokens are neither GT-selected nor permanently trusted.

The next candidate-relative JEPA predicts relations between a clean teacher-only identity target and current candidates while retaining source RGB cosine geometry. Its student uses actual past observations/current candidates only. Ordinary/noEMA/EMA controls share architecture and fixed budgets. Known-negative missing augmentation removes the positive and unknown candidates from an augmented training pool, then recomputes crowding; unknowns are never relabelled wrong. Ordinary/noEMA fits completed; EMA v1 exited before epoch1 because an update keyword disagreed with the teacher API. The failed source, config and log remain; a new v2 driver/output repairs only that call. Results from these models still require controlled comparison and actual tracking replay.

An optional runtime optimization transfers one float32 current IR tensor and pads the same batch of eight on CUDA, retaining identical FP16 values and contiguous layout. RGB-only cropping skips an unused identity IR crop. Default frontend behavior remains unchanged. Matched current-load 32-frame-per-video probes give identical predictions to each other and the historical prefix, with1.0920 versus1.2516 FPS; preprocessing/transfer averages .2075 versus .0589 seconds. These are short probes, not full4834-frame speed or new method accuracy claims. IR generation remains the dominant unchanged cost.

Validation:166 tests passed. Weights, feature banks, detector caches and full traces remain local, outside Git. Full official metrics, model hashes and state provenance are saved in the accompanying JSON reports. Research continues; neither the positive signals nor the negative first controls establish a final submission claim.
