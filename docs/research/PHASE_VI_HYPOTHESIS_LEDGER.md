# Phase VI hypothesis ledger

Status: IN_PROGRESS. Observations and causal claims are separate. No experiment has yet demonstrated a Phase VI benefit.

| ID | Question / discriminating experiment | Status |
|---|---|---|
| H0 | Can disabled modules reproduce the strong GRU and preserved official metrics? Hash audit, full-output replay and official reevaluation. | REGISTERED |
| H1 | Among wrong/unmatched online rows, is the true candidate absent, gated, present with weak history, or wrongly ranked despite cues? Retrospective two-axis classification; do not use GT online. | REGISTERED |
| H2 | Does conditional risk help beyond fixed confidence, margin/entropy and same-input MLP? Video-excluded supervised estimation and action/assignment checks. | PROPOSED |
| H3 | Does candidate-conditioned EMA prediction add anything beyond same-budget GRU/Transformer/MOTIP-style ID decoding? Matched training then complete video evaluation. | PROPOSED |
| H4 | Do isolated K=2 explanations prevent contamination better than K=1 and delayed write? Strict online bounded branch tests and complete videos. | PROPOSED |
| H5 | Do coverage or temporal interaction adaptations address the remaining dominant errors? Select experiments after H1 evidence. | OPEN |

Every completed experiment will append observed failure, root-cause hypotheses, discriminating experiments, repair/pivot and evidence update. Negative results are retained; promising signals expand to three seeds before model claims.

## H0 / H1 evidence update

H0 baseline protection validated: 88 hashes, three byte-identical strong GRU replay outputs, six exact official metric comparisons. Initial path error retained and repaired in a new audit run.

H1 actual-state diagnostic complete: known wrong accepted rows are often score-infeasible with a correct stage candidate, while geometry eligibility is not a hard mask. Early-reference and rolling-reference counts differ substantially; persistent drift must not be counted as distinct ID switches. Unknown past labels prevent absence-of-evidence claims.

Repair/pivot: train whole-video risk/context vs ordinary MLP to assess calibration and discrimination; then check actual Hungarian changes. Retain C independent of A/B outcome. Coverage-only intervention is a separate later hypothesis, not assumed to explain all failures. Remaining lifecycle/oracle episode analyses explicitly unperformed.

## Risk warm-start loader repair

Observed engineering failure: compressed NPZ members were accessed inside each row iteration, repeatedly decompressing entire arrays. Own parent/two training processes were stopped before any epoch or checkpoint existed. Original empty run directories, logs, campaign and source are retained. Repair materializes every member once and precomputes NumPy sampling groups. New immutable loadrepair_v2 outputs reuse the same config/data/budget; this is not another scientific hyperparameter search.

## H2 first video-heldout evidence

12 eight-epoch seed0 risk/MLP controls completed. Context does not consistently beat ordinary MLP; SN-015 accepted-edge AUC .5573 vs .5979 and local net correction −2102 vs −1984, while PF-001/SN-001 show positive local counts. This is not online IDSW evidence. Calibration transfer is weak in multiple outer videos.

Evidence update: candidate-set pooling of scalar cues is insufficient for a robust claim. Repair/pivot: examine actual assignment/acceptance changes and training-state shift; test raw candidate-history interactions and independent K2 branches. The warm-start producer limitation remains explicit; no branch-wide or project-wide NO_GO.
