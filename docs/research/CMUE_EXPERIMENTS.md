# CMUE experiments

Status: TWO_TARGETED_SEED0_ROUNDS_COMPLETE / BENEFIT_UNVALIDATED, 2026-10-12. All twelve two-fold, three-seed RGB/causal-IR teachers were trained from original ImageNet for fixed twelve epochs. Canonical dev was never used for teacher or utility SGD/calibration. Only seed0 teacher proposal banks and learned utility pipelines have been evaluated so far; these are diagnoses, not final three-seed selections.

Utility supervision is CE(RGB) minus CE(RGB+IR) on identical candidates/history and independent current-frame physical proposal labels (IoU >=0.5, top-two margin >=0.15). Heldout identities never use training-class classifier CE. GT/ambiguity annotate training examples without changing online outputs. Sticky earliest known past identities retain actual contaminated histories. The five fit videos exclude SN-015 query fitting; SN-015 calibrates a nonnegative affine utility transform. It is not completely isolated from all teacher learning: the opposite teacher fold trained there. Canonical dev stays held out.

All gates have 5,697 parameters, eight fixed epochs, identical video-uniform sampling, and three declared association weights .1/.3/.5. Same-evidence ordinary and quality-only gates optimize candidate CE. Execution threshold .02 and IR cap .5 were fixed before tracking. No parameter was fitted on the two development videos. Each round retains commands, SHA256 inputs/checkpoints, source snapshots, per-video official metrics and logs in the registry. `phase2_measured_rounds_20261012.json` indexes compact measured results.

## Round 1: local modifiable candidates

Frozen `cmue_trial_v1.json` / `utility_execution_v1.json`. 35,668 OOF queries: 23,357 fit and 12,311 calibration. Complete seed0 best HOTA/identity outcomes at w.3:

| Gate/control | HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| CMUE | 50.9769 | 36.3273 | 61.2131 | 86 |
| Same-evidence ordinary | 50.3778 | 35.5223 | 60.7252 | 91 |
| Quality-only / matched fixed fusion | 50.3556 | 35.4912 | 60.7232 | 90 |
| Constant IR distance .5, removes identity ranking | 50.9528 | 36.2939 | 61.2089 | 85 |
| Row-mean IR distance, removes identity ranking | 50.3721 | 35.4837 | 59.6349 | 87 |

The best CMUE outcome is only 0.0241pp HOTA above the constant-distance control and has one more switch. No independent utility or IR-identity contribution is established. Local query decomposition shows positive utility concentrated in NULL targets; true-candidate means are negative. Candidate-only ranking utility is also negative on both dev videos. Calibration sign AUC is 0.5066, correlation 0.3320; .02 accepts 91.98% on SN-015. A varying gate alone is not evidence of learned benefit.

Audit identifies 88 BT-005 and 60 SN-013 queries whose correct geometry-matchable candidate was excluded by the local RGB appearance mask and thus labelled local NULL. These are supervision-context omissions; original measurements remain intact. The first slower NPZ decompression attempt and duplicate-ambiguity correction are retained separately, never relabelled completed runs.

## Round 2: complete native row context

Frozen `cmue_trial_v2.json` / `utility_execution_v2.json`, declared before new queries. Candidate pool contains every geometry-matchable edge OR original RGB-modifiable edge. Teachers preserve identical geometry fallback, use fixed appearance weight .3 and native row NULL threshold .97/.7, and differ only on shared modifiable edges. Online evidence follows this complete context; online IR changes remain restricted to the original RGB mask. Same model/training budget. 35,668 OOF queries, same fit/calibration counts.

| Gate/control / best declared seed0 weight | HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| CMUE / .1 | 49.3071 | 34.0069 | 58.8742 | 85 |
| Same-evidence ordinary / .3 | 50.3556 | 35.4912 | 60.7232 | 90 |
| Quality-only / .3 | 50.3556 | 35.4912 | 60.7232 | 90 |

CMUE .3 gives HOTA48.9278/IDSW87; .5 gives47.1039/99. Calibration MSE improves to .008794 and correlation .4990, but sign AUC remains 0.5082. Threshold .02 accepts73.54% on SN-015. Numerical MSE scales differ because the teacher cost context changed; do not interpret the decrease alone as successful usefulness learning. Native Hungarian/lifecycle still handles NULL; no learned NULL/newborn decoder exists. An exact missing-IR edge precision guard was subsequently added and tested; first-round raw source snapshots preserve actual measured versions.

Neither round beats the strongest completed three-seed BoT-SORT HOTA52.4818 or establishes benefit versus ranking-free penalties. Do not declare global NO_GO after these two targeted rounds. A third repair, if justified, must address measured sign/ranking/calibration failure with frozen controls; it cannot expand the detector/model indiscriminately or tune historical test. Missing controls include matched untrained/no-calibration/no-fallback runs, actual input corruption/interruption, all final seeds and end-to-end cost.
