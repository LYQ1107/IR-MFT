# Phase VII full-set joint identity association experiments

Status: IN_PROGRESS. The new head and physical-label guards are implemented and tested; no new MOT gain is claimed.

`cfp/joint_identity_decoder.py::JointFishIDDecoder` consumes N×8×512 strictly past accepted features, M current features, native geometry, frozen GRU appearance, observable age/gap/active-lost state and detection confidence. It pools per-track temporal memory, applies track self-attention and detection self-attention, and exchanges both complete sets through cross-attention with geometry/appearance bias. An explicit unmatched token keeps fully masked and empty pools finite. Its M×(N+1) output addresses supplied track columns plus no-old-match; physical fish IDs never enter forward. Distinct local dictionary codes are randomized every training frame. This is independently implemented; it is not the official MOTIP detector/training system and attention alone is not a novelty claim.

Two routes have separate objectives. The bounded route adds at most0.03 to the unchanged ordinary GRU cost on observable legal edges; the residual starts exactly zero and retains the original assignment threshold0.97. The full learned route uses ID logits with global Hungarian assignment and a private unmatched choice per detection. Each old track can match only once, while multiple detections can remain unmatched. No-old-match does not mean verified newborn: downstream confidence, duplicate suppression, confirmation and reactivation still need explicit integration into the native tracker. Full online integration remains pending.

`cfp/joint_identity_supervision.py` lives outside online inference. Known physical IDs are scoped to an explicit video. Unknown tracks are omitted from known-negative loss denominators; if their identity coverage is unknown, an unmatched known detection cannot receive a fabricated NULL label. Duplicate physical track references and duplicate same-frame proposals lack a unique assignment target and remain model inputs but are excluded from this first ID loss. Losses include known-label CE, same-frame hard-negative ranking and excess track-column probability consistency. Ground-truth occlusion and first-ever birth labels are not claimed.

The warm-source builder reads all tracks, including unknown references, from each actual high-stage teacher snapshot. Previous Phase VI row training dropped unknown track references; those rows cannot be treated as a complete set. The new source preserves all rows and all high-stage proposals, with independently saved labels and observable inputs. Low stages and initial frames absent from the historical trace are an explicit coverage limit. EarlyReference is a strictly-past consensus convention (>=3 known observations, consensus>=0.6), not unconditional physical identity truth. This first teacher remains exposed through inherited backbone/GRU fitting and cannot establish encoder-level leave-video-out generalization.

The preregistered probe `configs/cfp/phase_vii_joint_probe_v1.json` fits BT-001/BT-003/MSK-002/PF-001, uses SN-001 for post-training calibration diagnosis and evaluates SN-015 only after fixed epoch8. A same-input independent pair scorer shares memory pooling, confidence, status, local dictionary and geometry/appearance inputs but has no set attention. Both use8epochs×400 uniformly sampled video/frame steps at seed0. This is an initial conditional head probe; three seeds, all outer folds, ordinary full-set Transformer, MOTIP-head adaptation, attention/motion/NULL/memory ablations, teacher-to-endogenous transition and all-frame online metrics remain pending.

Resume checkpoints include model, optimizer, scheduler, Python/NumPy-generator/globalNumPy/CPU/CUDA RNG and exact source/config/data/runtime/device guards. Checkpoint choice is fixed last epoch; historical dev/test videos never fit this probe. Cache/head training time is not complete online latency.

## First registered training result

Both seed0 heads completed8×400steps on the same sampled frames and local dictionary permutations. Joint training took137.80s and pair92.25s. Both allocate150,533parameters; the optimizer states show142,084participating joint parameters versus83,845pair parameters, so this is a matched-input/training-budget control, not an equal-effective-parameter comparison. Peak PyTorch allocated memory was25.14/25.55MB; CUDA context/reservation uses more and these are not complete-system memory figures.

| Fixed actual teacher snapshots | Baseline GRU | Pair full ID | Joint full ID |
|---|---:|---:|---:|
| SN-001 all known assignment accuracy (%) |26.3699|25.2283|26.5982|
| SN-001 existing-ID accuracy (%) |31.2230|28.7770|29.9281|
| SN-001 no-old-match recall (%) |7.7348|11.6022|13.8122|
| SN-015 all known assignment accuracy (%) |54.8239|51.7272|52.7222|
| SN-015 existing-ID accuracy (%) |83.6619|74.2500|64.7523|
| SN-015 no-old-match recall (%) |0.7712|9.5116|30.1735|

SN-001 has876supervised detections(695existing/181NULL); SN-015 has8,945(5,833existing/3,112NULL). Unknown/nonunique assignments remain excluded from measurement, not from model inputs. The0.03bounded residual in either head leaves assignment accuracy unchanged on these frozen states, although some existing-target unmatched counts change. The full joint head trades existing-ID recall for better NULL detection and is not superior to the baseline overall. These are not HOTA/IDF1 results. Detailed reproducible evidence is in `PHASE_VII_JOINT_PROBE_MEASURED.json`; checkpoints stay outside Git.

`cfp/online_joint_identity.py` now integrates both routes into the original native lifecycle with matching AnchorRecent memory. It processes the whole confirmed/lost high-stage set before writes; low/unconfirmed stages remain native. The full route's private-NULL global assignment is encoded into the native high-stage matrix, so unmatched high detections subsequently follow the existing confirmation/birth path. NULL does not force a fabricated birth or mark a physical identity as known. All-frame disabled byte parity, joint/pair bounded/full replays and pinned official TrackEval metrics have completed for SN-001/SN-015. See ABLATIONS and JOINT_REPLAY_MEASURED for the negative full-ID result and uniform-positive cost control. Raw online detector/encoder/generator/decode profiling is still pending.

## Residual identifiability result

On fitting/inner-calibration data, learned correct and wrong edge residuals both approach the+0.03bound. SN-001's mean correct/wrong edits are0.02772/0.02802; only8of175known no-old-match detections have all old costs capable of crossing threshold0.97even at maximum allowed edit. An untrained uniform+0.03control exactly reproduces pair-bounded outputs on both complete videos and joint-bounded on SN-015. Thus the first outer bounded gain has a sufficient cost/threshold explanation. This control uses the predeclared theoretical bound from the probe and the fitting/calibration residual diagnosis; no outer amplitude sweep is performed. It is not retained as a new learned identity contribution. Robust NULL prediction, raw/local identity preservation and on-policy set learning remain active research directions.

## Completed MOTIP/FDTA source controls

MOTIP exactffc0e905 uses five same-frame candidate self-attention layers and six cross layers with strictly past trajectory masks, random local50-word IDs plus newborn and explicit collision assignment; its trajectory model is residualFFNs. FDTA exactb3b3b778 adds six causal per-object history layers and a current-image predicted-depth branch. Offline sequence-wide teacher depth remains training supervision; it is not an online input. Its quality threshold/contrastive weights are actually GIoU despite IoU variable names. Both replace detector/query features in their original systems; the required baseline must adapt them to identical native fullsets/features/lifecycle. Dynamic absence/unknown supervision and vocabulary-cap controls remain required. No new implementation/training is claimed from source auditing.

## Additional eight source audit constraints

SAM2 current predictor permits newly prompted objects during tracking but assigns caller-supplied IDs independently; its past-pointer-only option does not forbid future spatial conditioning memory. SAMURAI bundled legacy predictor rejects newly prompted objects after propagation and stores shared SOT motion state. Detector-derived prompts, per-track state, explicit conditioning-frame guards, lifecycle adaptation and measured full-set identity competition are needed before either becomes a MOT control. TFMFT released geometry tracker contains duplicate unmatched births; missing detector/query core is a source-coverage exclusion.

## Additional three source audit constraints

MeMOTR exacteb7a177 has cross-track memory attention and score-gated detached feature EMA, plus native persistent detection/track queries. It is a prior joint-memory mechanism, not a full N+1 IDclassifier. TCEI exact145d1b8 reuses MOTIP joint IDdecoder and updates entropy-ranked current/cross-video caches before row calibration. Per-video reset, video/row order, self-inclusion and fixed native detector/lifecycle require controlled adaptations. TCEI project license/train entry absent at pin; no source/weight transplant or executed official result claimed.

## Final three source audit constraints

MOTRv2 exact1aac7c3 actual config10learned256queries plus current proposal anchors/history queries differs from parser300; GT denoising stays training-only, and default QIMv2 does not invoke declared drop/FP methods. TrackTrack exactee7f1c5 uses nonlearned mutual-argmin competition/TAI and two-NMS inventories, with offline _post linking/smoothing excluded. COVTrack exact9b0ced5 supplies learned confidence-conditioned cue fusion plus input-row greedy memo assignment, not N+1fishIDwords. All three require same inputs/lifecycle controls; source completion is not trained migration.

## Registered centered-cost / separate-NULL repair, three-seed complete fits

Allnine controlled fits (pair, parameter-matched crossTransformer, fullset joint ×seeds0/1/2) completed fixed8×400 CPUupdates each:28800actual optimizer updates. Per-epoch sampled-frame/local-code digests match across allthree architectures for eachseed. CrossTransformer keeps the full cross-set conditioning but replaces each set-self-attention with independently gradient-active per-token FFN of exactly4h²+4hparameters. Pair has lower activecapacity; allocated parametercount alone is not an equalityclaim. Same physical known/unknown/nonunique masks and original512conditionalwarmdata retained, with inheritedencoder/GRU exposure explicitly disclosed.

The bounded route now removes per-candidate common legal-edge shifts before tanh, then zero-centers the bounded edits; every edit≤0.03 and illegaledges unchanged. Uniformpositive correction becomes exactzero, so it cannot reproduce a threshold-only improvement. Conditionalranking trains only knownexisting-IDcompetition; NULL/unknown are excluded from this objective. FullN+1 privateNULL classification remains a separate existing/NULL-group-balanced objective on only verified labels. No fabricated newborn labels or unknown hardnegatives. Untraineduniformcentered control reproduces all684SN001/6570SN015 original baseline predictionbytes exactly, zerochangededges.

The actual jointseed1 epoch2exit/freshresume equals uninterrupted8epochtraining exactly:197tensor/1array/931scalar leaves acrossmodel/optimizer/scheduler/allRNG, digests, losses, gradients and all fixed-state metrics. Walltime/containerbytes excluded; no crosshardware guarantee. Full rawproof is hash-linked in PHASE_VII_CENTERED_JOINT_THREE_SEED_MEASURED.json.

| Architecture | InnerSN001 existing-ID accuracy | InnerSN001 NULL recall | OuterSN015 existing-ID accuracy | OuterSN015 NULL recall |
|---|---:|---:|---:|---:|
| pair | 24.029 ±2.173 | 20.994 ±4.385 | 54.860 ±4.562 | 46.572 ±11.419 |
| cross_transformer | 25.995 ±1.717 | 23.573 ±3.552 | 42.928 ±2.754 | 67.909 ±1.132 |
| joint | 23.741 ±1.295 | 29.650 ±3.237 | 38.739 ±2.752 | 69.751 ±2.523 |

First jointseed0 actualnative bounded HOTA46.219/31.538 onSN001/SN015 versus original46.238/31.983; fullroute28.120/13.225, withSN015IDSW2206. The threshold confound is removed without establishing a positive learned contribution; balancing NULL still costs existing-ID continuity. All18sameinput model/route replays (36fullvideos) continue and onlycompletedofficial rowscount as evidence. No outerthreshold/recipe search.

Inner-only graph audit records2of801selectedexisting edges outside the original .97feasiblecost graph, and451of695knownphysicalexisting targets outside that graph. This separates candidate-feasibility/support limits from model identity/NULL errors. The latter targets are not relabelledNULL; forcing them toNULL would fabricate labels. The2edge observation alone doesnot explain native degradation or justify a broadgraphfix. On-policy history/labelcoverage, rawcue preservation, dictionary sensitivity, currentset competition and matched expanded-graph controls remain meaningful next diagnoses. Outer graph diagnosis is rejected before model/query loading.
