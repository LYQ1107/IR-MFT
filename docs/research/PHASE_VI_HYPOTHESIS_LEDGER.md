# Phase VI hypothesis ledger

Status: IN_PROGRESS. Observations and causal claims are separate. Preliminary seed0 training-video signals require OOF/multiseed validation.

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

## H3 / H4 controlled failures and discriminating extensions

Five real candidate-conditioned fits: matched attention ordinary/noEMA/EMA share initialization, all eight sample orders,109186 parameters. EMA512 updates are genuine, but outer SN015 ranking39.62% loses ordinary attention46.44%, noEMA42.15% and ordinary candidateMLP54.46%. Cached masked-history scalar channels are blocked; no teacher/GT/future in student forward/export. Fit NULL labels128 versus outer known-NULL1087 expose a supervised-absence imbalance, not permission to relabel unknowns as wrong. Test raw-coordinate priors, task-aligned/residual scoring and known-only missing-candidate training separately.

Independent K2 complete six-video official HOTA32.7338 versusK132.8909; IDSW785 versus679. One BT003 expansion leads15 later branch selections. Current/past hysteresis and causal two-frame challenger confirmation are registered and running, keeping the v1 scoring/branching as a control. No past output revision.

Joint GT oracle on fixed training states reduces naive local opportunity estimates through one-to-one competition. SN01524621 wrong frame/track rows correspond to484 consecutive known-wrong episodes; joint stage/full N01=5712/6010, N10=0 under preserve-correct tie handling. Wrong->NULL is abstention, not correction. Not a whole-tracker ceiling.

State-shift hypothesis: warm-start teacher exposure and raw-score coordinate changes are distinct from irrecoverable identity evidence. New clean bootstrap32356queries derive solely from training physical annotations/frozen proposals, independent of any historical trained tracer. Fit models exclude complete outer and inner calibration videos. Their subsequent causal replay—not cleanGT training contexts—will supply new actual accepted histories. Backbone pretraining exposure remains disclosed. No project-level termination rule is based on the number of these failed configurations.

## H2 state transfer and H3 required controls

Warm memory benefit is matched by a2-parameter learned-cost threshold. No-history/untrained calibrated controls lose. All18 outer-video memory replays complete: new baseline H33.4884, context32.4190, MLP32.4971, learned threshold32.3156. Absolute calibrated gates fail on actual excluded-video states. Pivot: fixed10% actual-matched write budget with seven ordering controls, all thresholds global and frozen before runs. No outer-video gate adjustment.

Ordinary two-layer Transformer and explicit temporary-ID decoder trained with measured capacities105153/108416 versus candidate family109186. Four explicit masking/corruption/NULL/evidence controls complete; raw metric baseline remains stronger. Group loss/other-track information differences of the ID decoder are declared and its poor result is not a faithful official MOTIP defeat claim.

## H3 bounded correction and inherited-goal continuation (2026-10-13)

Cost replacement causes massive IDSW (13,777 EMA vs676 ordinary producer) even with better fixed-state relations. Native eligibility controls restore stability, supporting preservation of ordinary geometry as a separate mechanism. Four bounded0.05 correction families completed all24 outer-video replays: ordinary32.9522, noEMA33.6817, EMA33.6817, futureEMA33.5538 versus corresponding ordinary GRU33.4884. IDSW679/677 for EMA/noEMA versus676; HOTA gains are modest and not identity-switch reductions. EMA does not have an independent demonstrated advantage.

Alternative explanations: auxiliary prediction regularizes ordinary supervised scoring; cue-utility supervision or a fortunate initialization determines the signal. Registered54 fits/replays discriminate these explanations: ordinary/noEMA/EMA initialization-query seeds1/2 with frozen encoder/producer0 (36 fits); same three families without utility at initialization0 (18 fits). Identical epoch8/scale0.05/protocol and full-video outer evaluation, no outer-dependent selection. No fixed trial count is a project stopping rule.

## H2/H3/H4 same-fold interactions (2026-10-13)

Independent quota/bounded-prediction signals are small and neither establishes EMA necessity. Warm-state A+C cannot substitute for same-fold ABC controls. Register all8 B0/A/B/C/AB/AC/BC/ABC combinations on the six excluded-video producers, with frozen10% writes, residual0.05, ambiguity0.05 and hysteresis0.02. A and B predictions are reused and rehashed; B0/newC/AB/AC/BC/ABC require36 full replays. No new model selection is performed. Risk features see the corrected current cost in AB/ABC; that interaction is explicit. Hypothesis traces record causal current explanations and birth keys without labels; retrospective diagnostics may later label correct/incorrect/unknown explanations. Complete disabled outputs must equal every preserved outer baseline.

## Deployment transfer and exact residual continuation

Full4834-frame canonical riskMLP online method completed with official HOTA53.1035/IDF162.1078/IDSW87, at1.2371FPS shared load (p50739.5ms/p951116.8ms). Cached replay preserves canonical ordinary current frame/ID outputs on both historical dev references: no deployed identity improvement. BT005 full output is byte equal; SN013 differs in7 numerical rows only in final2999/3000 detector batch boundary (maximum0.338px/score0.000952), with equal ID/frame and risk/write counters. Do not claim whole-stream byte equality or controlled speedup.

Bounded-residual epoch3 stop/resume was verified against the preserved continuous run on the same physicalGPU0: all model/EMA/optimizer/RNG/order/loss/guard fields exactly equal, excluding only report Git commit.54 new fits are complete; full-video outcomes remain pending.

The first complete BT001 B+C causal trace contains9569 known-wrong winning frame/track explanations, only30 with a known-correct retained challenger and0 correct winners with a known-wrong challenger. This uses observed-birth physical labels for retrospective diagnosis and excludes14243 unknown decisions;697 consecutive known-wrong episodes are not official IDSW. Other scenes/final MOT effects are pending. Retaining two explanations most of the video does not itself establish useful ambiguity coverage.

## H2/H3 complementarity after complete same-fold matrix

Seed0 full six-video official results: B033.4884,A33.9488,B33.6817,C33.5032,AB34.3611,AC33.9040,BC33.6597,ABC33.9299. AB has a new positive signal; adding C reduces it. Common-birth retrospective AB N01420/N10438 excludes changed births and77,445 unknown decisions, so it cannot alone explain the global HOTA gain or prove physical recovery. Register24 whole-video replays for coupled A/B initialization/query seeds1/2 and two mechanism controls: noEMA+B with learnedA, and fixed margin write ordering with the same B. Frozen backbone/producer0 and all thresholds; no test/dev selection. If the signal survives, whole-pipeline encoder/producer seeds remain required.

## Repeated predictive signal and ordinary-mechanism simplification

Three initialization/query repeats at fixed backbone/producer0: ordinaryCE residual HOTA33.0149±0.0743, noEMA predictive33.6428±0.0318, EMA33.6296±0.0368 versus corresponding GRU33.4884. AB three-init mean34.1118±0.1784; fixed margin+EMA B34.4198 and noEMA B+learnedA34.3855. NoEMA+fixed margin at identity0 also34.4198, exactly the EMA-margin metric result. Learned context, EMA andK2 are not supported as necessary explanations of the positive signal. Calibration removal produces exact whole trajectories and actions in all sixABvideos, so probability calibration is unnecessary for this frozen quota policy (not a claim about absolute gates).

Retain prediction+bounded correction+fixed-margin write budget as an open research candidate. Independently frozen RGB1/2 coordinates and12 new excluded-video ordinary producers are now fitted with identical physical bootstrap indices; next generate actual causal histories,24 matched ordinaryCE/noEMA modules and48 within-seed full replays. This changes encoder/producer/module seed together while detector remains shared and pretraining exposure remains disclosed. No new outer-specific thresholds or historical-dev selection. Causal event traces independently reproduce official CLEAR totals; frame-level paired-birth corrections exclude changed births/unknown support and are not IDSW.

## Identity-coordinate replication and scoped conclusion (2026-10-13)

The registered continuation completed12 actual six-video states,12 first3/recent5 query banks,24 matched ordinaryCE/noEMA fits,48 whole-video replays and8 official family evaluations. Two disabled baselines were evaluated from the corresponding seed1/2 causal state replays; seed0's older anchor was not used as their baseline. The frozen detector remains shared, so this is conditional association evidence with disclosed backbone exposure.

For predictive residual plus fixed-margin writing, HOTA relative to each seed's own disabled baseline was +0.025819 (seed1) and +0.061330 (seed2), while IDSW rose by2 in both. The earlier seed0 anchor gained +0.931433 HOTA but rose by16 IDSW. Ordinary residual-only was −0.504564/−0.258273 HOTA for seed1/2; predictive residual without margin was −0.407682/+0.161139; margin-only was +0.144201/−0.132530. Predictive and margin mechanisms therefore do not establish a stable deployment-level identity recovery gain. The final scoped conclusion retains ordinary GRU/current measurement as the recommendation; the candidate raw-online deployment is explicitly unperformed rather than claimed. External datasets, faithful full MOTIP/GOT reproduction, larger hypothesis groups, learned branch selection and detector/encoder retraining remain open.
