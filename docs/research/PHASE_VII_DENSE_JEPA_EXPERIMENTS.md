# Phase VII object-centric Dense JEPA experiments

Status: IN_PROGRESS. Actual independent local-token JEPA and eight controlled alternatives have now been trained; current identity superiority is not established.

A0 shows why prediction-loss optimization alone is insufficient: frozen GRU pure retrieval loses to retained global features on clean SN-015 histories, while fixed native geometry scores hide much of that difference. Joint probes also show that apparent bounded gains can be reproduced by a constant cost edit. A future DenseJEPA result must add identity evidence under matched current candidate support, not merely lower latent error or move the acceptance threshold.

The required implementation uses current object crop local/spatial tokens from verified prior/current observations, visible past-only online context and an EMA target encoder used only during training. Future training target tokens may be stop-gradient labels but cannot enter history, online memory, masks, reliability or decisions. Masked target tokens cannot leak through teacher pooling or context construction. Target masks must correspond to annotated/observable validity; missing physical identity is not a negative.

Matched local TemporalTransformer, contrastive temporal learning, noEMA/noPrediction/noID and simple local aggregation controls remain required. Input coordinate, producer, crop budget, candidate pool, training video split and parameter/compute budget must be recorded. Paired crop features/retrieval and I-JEPA/V-JEPA2/GOT source-field audits are complete. All30reference audits and licensed DINOv2 observations are complete; the implemented controlled localBfirst-seed study is reported below. No dense predictive novelty or performance claim is retained at this stage.

## I-JEPA source audit completed

Exactcommit52c1ae95d05f743e000e8f10a1f3a79b10cff048 has16verifiedsource/config/license files andall11fields inREPOSITORY_REGISTRY. Code is [CCBYNC4](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/LICENSE), notMIT/Apache; no officialcodecopied/executed orhugeweightsdownloaded. Staticimagecontext/maskedregionSmoothL1/stoppedEMA doesnot providefishidentitysupervision. Actualmasker relaxesdisjointness afterretries andrectangularpositioninterpolation isunfit for128x256; independentlyimplemented DenseJEPA musttesttargetexclusion/current-futurecausality andrectangularlocalshapes explicitly. This audit preparesB; no actuallocalDenseJEPAtraining resultis claimed.

## Source audit completion: thirty available-source field audits

All30 references now have all11 available source-level semantic fields and exact commit/blob/SHA reading proofs. The final completed set includes MOTRv2, TrackTrack and COVTrack;599listed source/config/license paths have exact provenance and targeted reading records. TFMFT missing model/criterion/sampler and TCEI missing train-entry modules remain explicit coverage exclusions, rather than verified complete training; this is not an exhaustive every-line reading claim or official pretrained reproduction. Weight delivery, unverified derivative/data terms and execution exclusions remain explicit in REPOSITORY_REGISTRY.

GOT-JEPA's registered head5f382645 also contains newerGOT-Edit; the README's paper release84e9324317 was independently materialized from existing local Git objects into a new isolated snapshot. Both actor versions define the same callable twice and the active last body ends with an unassigned tuple expression instead of the earlier loss/statistics return. Target/context feature-extractor parameters are shared; noEMA teacher update is implemented. Its latent target is a256-dimensional tracking filter, not Kfish identity tokens. Static AST and canonical source proofs are external/hash-linked; no foreign code was executed.

V-JEPA2/2.1 ordinary encoders attend bidirectional video clips. AC predictor's frame-causal mask does not establish causal backbone features.2.1Base's published train config selects a model name omitted from the train entry's embed-dimension branch; pretrained hub inference is a different path. CoTracker3's online wrapper initializes query/state separately and advances16frame windows by8; earlier window outputs can use later frames in the same window. Current-frame MOT needs explicit prefix boundaries, delay/emission policy and runtime accounting. Original full-video offline caches are excluded.

TransReID and CLIP-ReID static evaluators remove same-ID/same-camera positives, which would remove within-video fish positives if camera=video. Fish adapters must use physical(video,fish) identities and the established past-only unknown-preserving retrieval masks. Original JPM four branches are hardcoded, and CLIP prompt labels are training-class anchors rather than necessary onlineGT. CLIP stage1 actual contrastive temperature is1.0, not assumed.07; the final cache minibatch requires an empty-batch guard when exactly divisible.

The complete task book explicitly requires first completing environment/all30 source audits before further method development. Earlier executed visual/joint controls remain evidence, with their negative/confounded results preserved. All30available-source audits are now complete with missing-module/license/weight exclusions; this prerequisite gate is satisfied. The next work is controlled strong/local visual training, followed by genuine localJEPA and same-input C/D. Full goal remains IN_PROGRESS; actual localDenseJEPA training, stronger lawful pretrained baselines, repairedsetcompetition, independentmotion and all-fold/three-seed/fullrawonline remain required. See PHASE_VII_SOURCE_AUDIT_MEASURED.json for exact counts and artifact hashes.

## Additional eight source audit constraints

DINOv3 uses stopped EMA DINO/iBOT teachers and optional patch-relation Gram anchoring; default source config disables Gram and is not the full published7B recipe. DINO-Tracker full-test-video optimization has no causal JEPA teacher path. These source findings strengthen the requirement for matched local-token temporal prediction controls and forbid presenting pretrained patch distillation or offline video fitting as new online JEPA evidence.

## Additional three source audit constraints

MeMOTR feature memory EMA0.01 with detached reads and confidence gating is not an EMA teacher network or local temporal prediction loss. TCEI calibration is no_grad/cache based; DiffMOT supervision predicts geometry via diffusion rather than local visual latent evidence. These strengthen required same-input temporal/memory/prediction controls and prevent attributing ordinary memory/gating or nonlinear box prediction to new DenseJEPA novelty.

## Final three source audit constraints

All30source audits are complete. COVTrack already combines current/past roundtrip consistency confidence with learned residual appearance/position/semantic fusion; its CycleLoss helper is used even when cycle training loss is disabled. This is not a stopped EMAteacher/local prediction mechanism. DINOv2 genuine spatial162tokens now extracted from official pretrained source; B can use these as frozen observations, but must train a local context/temporal predictor and stopped EMA target under strict outer/inner exclusions, with matched noEMA/noPrediction/noID/contrastive temporal controls.

## Executed object-local Dense JEPA controls

A genuine new local model is now implemented and fitted, distinct from inherited global512PLIM. Immutable licensed DINO observations provide real K4spatial384tokens and currentCLS. An identity-initialized local encoder learns observed-region attention/reliability; a temporal context processes only strictly past object observations. A regional masked-target decoder predictsKlocal tokens. Independent stopped deepcopy target encoder follows EMAafter successfuloptimizer updates, with clean currentGT targets only in training. No currentcandidate/GTtarget/teacher feature enters the prediction path; currentcandidates enter a separate cross-region relation scorer. CurrentGT classification is a separate ordinary training-supervision branch, never a predictor or online input. No copied I-JEPA/V-JEPA/FishMAE code is executed.

All nine first-seed fits complete at fixed8epochs/1200CPUupdates, excluding outerSN015/innerBT001 from gradients/classifier state: currentCLS global, ordinary384GRU, localattention, localTemporalTransformer, localtemporalcontrastive, noEMA, no masked-prediction objective, no explicitIDloss, fullJEPA. The raw backbone is immutable; this is the task-book-allowed learned adapter route. EMA/instant-target methods have1200real stopped-target updates, other models0; actual visual-context/temporal/predictor gradients and changedparameters are recorded. Independent teacher storage and no target gradients are verified. No-ID still uses annotated temporal grouping, so it is not annotation-free learning. NoPrediction keeps the regional query architecture trained through IDrelations but removes masked latent supervision. True active parameters/budgets are reported, including inactive frozen modules.

Training uses exact same per-epoch sampled current/past/target rows and target masks across all nine modes, verified by input digests. Same-frame known different-fish proposals are negatives; unknowns remain in the current inference pool and are excluded from labeled negative softmax denominators. Training past views mix uniqueGT and verified unique-detector observations50/50, gaps1/25/100. Evaluation separates GT/verifieddetector histories, GT/actualdetector current pools and all gaps. It retains currently absent pastknown inventory and empty sampled scenes. These are clean physical offline diagnostic groups, not endogenous native tracker states. `target_absent` denotes verified-positive absence, not proof that an unknown proposal cannot match or a valid NULL training label. Ground-truth identifiers/teacher/target never enter student forward. Natural occlusion/anatomical labels are absent; observed fractions are crop/padding support only.

| Seed0 method | InnerBT001 GT-history→detector R1 | OuterSN015 GT-history→detector R1 |
|---|---:|---:|
| Frozen CLS FIFO | 31.544 |64.894|
| Learned global currentCLS |37.584|72.695|
| Global+ordinary GRU384 |32.886|51.241|
| Local patch attention |36.465|66.312|
| Local temporal Transformer |36.913|64.716|
| Local temporal contrastive |29.306|47.518|
| Dense without EMA |27.293|44.858|
| Without masked-prediction loss |27.740|41.844|
| Without explicitIDloss |15.213|26.241|
| Full Dense JEPA |26.846|44.504|

All entries use the same current pools/unknowns/support: inner447 and outer564 queries with both verifiedpositive and observedpast at gap25. Raw query totals640/793, with verified-target-absent184/217 and history-absent10/17 separately recorded. FullJEPA reduces latent loss yet remains below global/ordinarylocalTemporal/frozen controls. It beats its query-decoder IDonly ablation onouterSN15, which does not establish superiority over the strong controls. No EMA/localJEPA identity or native improvement/novelty claim is retained. Three-seed confirmation remains running. Numeric fullallprotocol R1/R5/mAP/AUC/EER/margins/variance/unknown/support/source/checkpoint evidence is PHASE_VII_LOCAL_DENSE_FIRST_SEED_MEASURED.json. Physical perquery annotation traces and model arrays remain outsideGit.

A read-only fixed-full-checkpoint diagnostic reads innerBT001 only and rejects outer query requests. Keeping the same learned currentencoder/relation scorer and supports, encoder-only FIFO37.584%, temporalcontextpool37.584%, regionalprediction26.846%. Thus a measured inference decoder bottleneck is present; a blanket localencoder-information-loss explanation is unsupported on this inner pool. This motivates a separately registered representation-learning control: keep masked prediction as a training-only auxiliary objective and use encoder/context evidence for identity decisions, as ordinary JEPA downstream evaluation does. No new recipe is selected from outerSN015; the original model/results remain preserved. K8/allfold/onpolicy/native/rawonline, strongsource and finalnovelty work remain outstanding.

The genuine localJEPA fullseed1 real epoch2exit/fresh-process resume now exactly matches continuous8epochCPUtraining:250tensor/1array/1626scalar leaves acrossstudent, independentEMAteacher+1200updatecounter, AdamW, scheduler, all RNG, per-epoch targetmask/inputdigest/losses and allGT/verified-detector scene metrics/supports. Teachergradientsremainabsent; noCUDAallocation. Walltime/filepaths/containerbytes excluded; no crosshardware guarantee. Exact proof is PHASE_VII_LOCAL_DENSE_RESTART_MEASURED.json.

## Completed first-fold three-seed confirmation

All27 original V1 fits are COMPLETE at the fixed final epoch8, 1200 successful optimizer updates each. Same sampled scene/mask digests and frozen exact query supports are verified; no outer checkpoint or threshold selection. These are offline clean physical groups, not native track histories. SN-015 GT-history/current-detector gap25 supported564 of793 queries, target-present576, verified-positive-absent217 and history-absent17. Mean ± sampleSD over seeds0/1/2:

| Model | Rank1 (%) |
|---|---:|
| global | 72.813 ±0.369 |
| global_gru | 51.537 ±0.369 |
| local_attention | 65.307 ±1.589 |
| local_temporal | 63.948 ±0.800 |
| local_contrastive | 49.527 ±2.888 |
| no_ema | 44.740 ±1.245 |
| no_prediction | 43.203 ±1.647 |
| no_id | 25.591 ±1.609 |
| full | 45.095 ±1.181 |

Frozen CLS FIFO is64.894%. The fullJEPA45.095±1.181% result is below global72.813±0.369% and ordinary local-temporal63.948±0.800%. All12 history/current/gap protocols, R1/R5/mAP/AUC/EER and immutable source/checkpoint/support proofs are preserved in PHASE_VII_LOCAL_DENSE_THREE_SEED_MEASURED.json. NoID removes explicit ID losses but retains supervised GT temporal grouping; verified-positive absence does not certify physical absence or NULL.

## Registered context-identity V2 repair

The innerBT001-only frozen-model stage diagnosis motivated a separately owned V2 class/config/trainer: past local-temporal context supplies identity scores; regional masked prediction remains an auxiliary training objective. Deployment uses `.online()` and never executes the prediction decoder or teacher. Decoder perturbation cannot change identity scores; local/temporal/predictor gradient paths and stoppedteacher behavior are tested. V1 files and negative evidence remain unchanged.

V2fullseed0 has1200 successful optimizer/EMA updates. Offline supported SN015 detector R1 is63.652%, innerBT00136.242%, while the matched V2no_prediction seed0 is64.716%/36.913%. It restores the decoder bottleneck without establishing an extra identity benefit. The V2 no_prediction and local_temporal are exact control aliases; V2no_predictionseed0 student tensors equal the completed original local_temporalseed0, verified outsideGit in local_dense_context_no_prediction_alias_v2.json. Do not count these labels as independent evidence. V2three-seed completion and allfold/K8/native positive effects remain pending; the actual V2restart proof below is now verified.

## Actual native history integration

New features are encoded from each current RGB frame and original causal V3 proposals, including empty states. Native scoring uses only each track’s accepted proposal references with exact strictly-past timestamps, last4 slots/max age1500, plus current candidate observations. No physicalID, GT history, future frame or teacher enters replay. Frozen384DINO B coordinates are checkpoint-guarded; supervised384A features are a different producer and cannot silently feed these B models. The common learned logit scale is explicitly inverted into bounded cosine-cost units, preserving the small relation residual. Geometry/lifecycle/low/unconfirmed/writes remain the registered ordinary tracker.

The actual first512SN015frames parity audit replaces the learned cue with exact FIFO and reproduces accepted boxes/localIDs/lifecycle/history exactly, maximum association-cost difference3.6322e-8 across1024calls. It certifies this native prefix, not full-video learned parity. The first V2full native6570frame replay is complete; official evaluation and matched three-seed V1global/GRU/localAttention/localTemporal/full/ablations plus V2full are running. Offline crop R1 is not reported as nativeMOT improvement.

The first full6570SN015frame contextV2seed0 official result is HOTA30.694/AssA13.405/IDF125.298/MOTA91.032/IDSW146. Same-current frozenFIFO is32.225/14.786/30.445/91.019/152. Original Bglobal model gives31.017/13.664/26.362/91.003/145. Fewer switches accompanies weaker identity continuity; clean physical-history retrieval is not endogenous native performance. No recipe is selected from outerSN015; matched native allseed controls continue.

The separate V2contextfullseed1 real epoch2exit/fresh-process resume exactly matches its completed uninterrupted8epoch execution:250tensor/1array/1626scalar leaves across student, independent stoppedEMAteacher and1200updatecount, optimizer/scheduler, all RNG, observed-input/mask digests, losses and all heldout protocol identity/discrimination/support metrics. Both use the same guarded V2class/config/trainer bytes. Walltime, containerbytes and per-run outputpaths are excluded; no crosshardware guarantee. PHASE_VII_LOCAL_DENSE_CONTEXT_RESTART_MEASURED.json preserves the actual proof. This is execution correctness, not identity or tracking superiority.
