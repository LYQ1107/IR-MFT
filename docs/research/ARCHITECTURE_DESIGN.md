# CFP-FishMOT prospective architecture

Status: controlled design, benefits not yet measured. Original V1/V2/V3 files remain unchanged. New implementation belongs to `cfp/` and `tools/cfp_*`. User-confirmed whole-sequence median IR is noncausal; strict online experiments require separately rebuilt current/past-only IR and detector caches. Legacy-input diagnostics are exploratory references, never proof of a causal complete system.

## Baseline contract

V3-E3 is a 186,199,381-parameter YOLOX-X with two independent YOLOPAFPN branches, RGB-dominant scalar gates per P3/P4/P5 and standard ByteTrack. Strict checkpoint loading passed. Gate values: 0.051871, 0.054622, 0.054250. Input is six channels, visible then constructed IR, 800×1440 for historical inference, FP16. Detection cache stores original 1920×1080 xyxy boxes and scores after confidence 0.1/NMS 0.7. Freeze this detector first. Disabling all CFP modules must call the original BYTETracker and reproduce its outputs exactly.

## Phase I: target identity information

Use pretrained RGB crop encoders before detector changes. Available local ImageNet ResNet50 and person OSNet weights must be hashed and their domain disclosed. DINOv3 ViT-S/16 is a candidate frozen-feature baseline if licensed weights/runtime are accessible; do not pretend other checkpoints are DINOv3. RGB/IR heads use equal dimensional embeddings and L2 normalization. Compare frozen features, small projection fitting, and controlled fish ReID fine-tuning with CE + batch-hard triplet; CE-only and triplet-only controls when diagnosis warrants. Batch construction samples identities uniformly and temporally separated observations with same-frame nearby negatives; no labels are shared across videos.

Use paired GT boxes for supervised feature diagnosis, then evaluate on identical detector boxes to quantify crop/domain shift. Gallery must precede query frames for causal Recall@1; pairwise AUC is a separate offline diagnostic. Candidate-level softmax calibration must include unmatched labels, not calibrate arbitrary pair probabilities as if complete assignments. Compare RGB, IR, concatenated normalized branches, fixed scalar mixing, learned gate, and matched-parameter MLP/attention. Correct IR, zero IR, RGB repeated as IR, randomized IR features and wrong paired IR are interventions at identical candidate conditions. Never choose IR corruption or best fusion on test.

## CMUE: conditional algorithmic intervention

For one track's current candidate set, freeze RGB and RGB+IR teachers on training-disjoint folds. Let supervised candidate loss include the true candidate or NULL when none is correct. Target u = CE(RGB) − CE(RGB+IR). This is a fixed-candidate algorithmic intervention, not physical causal identification. Initial estimator: small MLP of RGB/IR scores and disagreement, geometric distance, detector confidence, score margins, online quality, age and accepted-memory reliability. No GT or future inputs. Start with Huber utility regression and utility-sign calibration; add an association loss only in a separately recorded trial.

Estimator predicts candidate-set conditional benefit; mixing uses an explicitly calibrated positive-utility acceptance threshold fixed before tracking; the first trials use .02. Negative/rejected utility returns the exact RGB score tensor, avoiding unnecessary arithmetic differences. Accepted utility caps IR contribution to protect RGB. Compare fixed/learned/quality/entropy gates, same-evidence non-counterfactual MLP, parameter-matched alternatives, no supervision, no fallback and no calibration. Count accepted/rejected decisions and actual corrections/harm, plot utility calibration/variation. Near-constant gates do not establish conditional benefit. Teacher-heldout targets, true utility on dev and online tracking outcomes are separate evidence levels.

## PLIM: identity prediction and writes

Input: at most L∈{4,8,16} accepted historical embeddings per track, timestamps/gaps and reliability. Predict normalized current identity representation with a small GRU or causal temporal Transformer; incorporate gap explicitly. Frozen target encoders generate future training targets with stop-gradient. Start cosine predictive loss + same-frame hard-negative ranking to avoid average-fish collapse. Evaluate predicted-feature diversity, variance, same-ID recovery and errors against last/EMA/FIFO, a nonpredictive GRU, matched-size Transformer and motion-only baseline.

Keep geometry update and identity writes separate. Reliability policy is independently toggled and compared on identical predictor weights: normal EMA writes, lowered weights, reject/delay ambiguous writes. Reliability uses observable assignment margins, confidence, occlusion proxies and modality conflict only. Never use online oracle identity. Compare teacher-forced clean histories to histories from actual tracker outputs to diagnose contamination. Predict before any write for current frame; reset memory/IDs between videos and bound memory. NULL leaves track unmatched; newborn allocates a unique persistent output ID. Retain Hungarian initially; direct ID decoder becomes a separate experiment only if useful on dev. MOTIP's reusable internal vocabulary labels are not persistent output identities.

## Optional interaction

H3 only after H1/H2 controlled evidence. Local relative position/velocity and candidate ambiguity can supply capped auxiliary scores. Compare simple neighbor averages, fixed rules, local attention and fallback. Fish need not share velocity. Do not let group scores overwrite strong individual evidence. Remove the module if it fails fair comparison.

## Required evidence and tests

Feature/online correspondence, train/dev leakage, paired modality alignment, finite module outputs, unique frame/track IDs, strict causal generation/predict/score/write order, exact RGB fallback, exact original baseline fallback, full-video replay and TrackEval parity. Causal IR at t must stay identical if frames after t are modified; this future-perturbation test also applies to detector proposals. Record every trained model/config/teacher/crop/detection hash, seeds, command, source commit, selection metric, outputs and failure hypotheses. Main unknowns: exact generator scripts/configs and causal replacement, actual modality identity usefulness, limited video diversity and exposed historical test. None may be resolved by invented performance claims.


## Implemented trial state (2026-10-12)

CMUE is implemented and two seed0 trial rounds are complete, without validated independent benefit. Round1 local mask and round2 complete native candidate context are distinct frozen configs. Round2 includes geometry-matchable RGB fallback edges in teacher CE/evidence and restricts actual IR edits to the same original RGB mask. Model/evidence capacity and budget stay fixed. NULL cost is the stage's native threshold; native Hungarian/lifecycle has no learned identity decoder. Exact rejected/missing-IR fallback is tested at both row and edge precision. Physical supervision is independent current-frame IoU, not official continuity-biased evaluation matches.

PLIM's tested bounded GRU/Transformer implementation uses stopped targets, strict past ordering and cosine-plus-same-frame contrast, with zero residual initialization. The first declaration fixes history8; lengths4/16 remain prospective extensions, not measured runs. Query construction is complete; neither predictor nor reliable-write policy is trained/validated yet. Both deployment modules preserve score-before-write ordering. Run source snapshots retain exact uncommitted bytes in addition to the checkout commit.
