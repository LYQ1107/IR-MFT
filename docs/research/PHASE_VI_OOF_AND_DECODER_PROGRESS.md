# Phase VI: state transfer and required decoder controls

Status: IN_PROGRESS. These are development research findings, not independent test claims. Historical BT005/SN013 remain exposed references. Frozen encoder/detector pretraining remains exposed in every association OOF fold.

| Six-video aggregate, seed0 | HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| Original warm-state ordinary GRU | 32.8909 | 18.8370 | 34.9324 | 679 |
| Warm context memory | 33.2667 | 19.2688 | 35.4979 | 677 |
| Warm learned-cost threshold,2 parameters | 33.3015 | 19.2675 | 35.7155 | 683 |
| Warm no identity history | 32.4364 | 18.2986 | 34.4371 | 679 |
| Warm untrained context + calibration | 32.4116 | 18.2726 | 34.4326 | 685 |
| New excluded-video producer baseline | 33.4884 | 19.6046 | 35.7681 | 676 |
| New excluded-video context memory | 32.4190 | 18.3336 | 34.7858 | 686 |
| New excluded-video MLP memory | 32.4971 | 18.4189 | 34.9014 | 688 |
| New excluded-video learned threshold memory | 32.3156 | 18.2213 | 34.7522 | 692 |

The JSON artifact contains all official per-video metrics. Comparisons must use the corresponding ordinary producer, not subtract old32.8909 from new32.4190. New outer-video predictions use six distinct models: this aggregate does not describe a single deployed predictor.

The warm context gain does not establish candidate reasoning: a two-parameter ordinary threshold explains at least as much aggregate HOTA benefit. The new producer changes accepted-history error distributions; none of the three absolute calibrated probability gates transfers positively. Inner-video calibration does not guarantee outer-video risk prevalence or calibration. Do not use a video-specific outer threshold to repair these failures.

Discriminating experiment now registered: per high/unconfirmed assignment stage, reject only floor(10% of actually matched scored edges), preserving all current Hungarian matches, geometry and outputs. Temperature/bias are monotone and thus irrelevant to within-stage ranking; compare learned context/MLP/cost-threshold ordering against confidence, margin, entropy and seeded random ordering at the identical quota. Low/unscored/newborn writes retain ordinary behavior. Candidate pools smaller than10 cannot reject any edge; the implementation skips redundant risk scoring there. This is a causal action budget, not GT recovery or a posterior confidence claim. All42 outer-video replays are running; no benefit is claimed yet.

The writer was rebuilt without instance-capturing closures. Track histories, evidence, accounting and branch RNG belong to the copied branch. FullBT0013000-frame outputs and memory counts equal the first writer byte-for-byte; branch-isolation tests pass. The risk+K2+hysteresis combination is separately running with the previously fixed .2 gate/.02 hysteresis and current/past decisions only. It does not invalidate the earlier independent C control.

Required B controls now include a genuine two-layer ordinary Transformer105153 parameters and an explicit temporary-ID dictionary decoder108416 parameters, compared to109186 for the first candidate attention/JEPA family. The ID control classifies each current detection among active trajectory ID codes plus newborn, with actual past features, timestamps and randomly reassigned codes during training. It is MOTIP-inspired, not the official MOTIP detector/training system. Each step consumes64 track histories, but whole-frame grouping, loss orientation and other-track raw context differ from per-row JEPA; these differences prevent claiming an exactly isolated architecture comparison. The small decoder performs poorly and needs a credible source-feature prior or longer explicitly matched training before use as a strong literature benchmark. Raw mean retrieval already remains a stronger nonlearned identity control.

Four candidate-JEPA controls completed for masking off, corruption off, NULL off and recomputed recoverability evidence. Recomputed history cues use only the visible perturbed past, ignoring cached full-history/prototype channels; unknown physical labels remain excluded from supervision. No-NULL keeps the same query sampling and ignores absence-label CE, rather than pretending those rows contain a positive. Complete ranking/unknown/NULL/N01/N10 tables are in PHASE_VI_CANDIDATE_EXTENDED_MEASURED.json. These are fixed-state local diagnostics, not IDSW or HOTA.

Five candidate-relative controls use new actual first3+recent5 histories and4372 parameters. First3 observations are unfiltered past inputs and not trusted GT anchors. Across all36 fold/video banks, current query keys, candidate indices, labels, original decisions and current geometry/confidence/crowding are exactly equal between anchor+recent and first3+recent5 views. EMA/noEMA/current relation prediction does not yet deliver a convincing correction margin; known-negative missing augmentation substantially increases NULL false rejection. No unknown candidate was relabelled wrong.

The configured relation epoch3-stop/resume proof matches model, target encoder/update count, optimizer, all query/mask/drop/Python/Torch/CUDA RNG, sample orders and losses exactly at epoch8. Git provenance differs because a stage was committed between starts; it is explicitly excluded from the numerical state equality. This is not an arbitrary minibatch crash proof.
