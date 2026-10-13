# Phase VII scientific research report

Status: IN_PROGRESS. This is a running evidence report; the EvidenceFishMOT final goal is active and unachieved. No new positive research contribution is asserted from the first probes.

The full goal/task book and existing PhaseVI assets are preserved on `research/evidence-fishmot-v1`. Completed work includes exact inherited six-video rank/cue reproduction, actual-cost and history-support diagnosis, original producer/history re-evaluation, independent full-set ID/NULL implementation, registered joint/pair training, native complete-video replay, an untrained residual identifiability control and both-crop metadata foundation,47,751crop encoding/paired retrieval and four adapter fits plus six three-seed actual backbone fits and a real restart-equivalence verification. At the last project-wide check216tests pass with18warnings.

The key first scientific finding is a measured confound: the bounded SN-015 HOTA increase31.9832→33.1384is exactly reproduced by an untrained uniform+0.03cost change, with identical prediction bytes. Both complete learned ID routes reduce HOTA and increase switches. This is actionable evidence for preserving raw/global/local identity cues, diagnosing NULL/existing tradeoffs and testing genuine set competition on future matched controls. It does not terminate the research direction or justify a fixed-count NO_GO.

| Final question | Current evidence / remaining work |
|---|---|
|1. 当前FishReID的主要缺陷是什么？ | Actual-state global features retain strong clean identity evidence; temporal compression/history and producer dependence are measured. Matched-pool GT/detector and raw/preBN/BN/local diagnosis now run: crop damage is severe on MSK002, smaller on PF001/SN015. No blanket backbone-defect claim. |
|2. 新ReID改善了多少身份检索？ | First-fold backbone global/local3seeds complete: outer detectorR1means42.80±2.22/43.76±3.31% versus frozenImageNet53.65%; no gain retained. More folds/seeds/strong pretrained controls pending. |
|3. 新ReID是否改善真实在线HOTA/AssA？ | Pending. First new C heads use the inherited encoder; raw new-ReID online comparison has not run. |
|4. 局部Patch是否优于单一全局特征？ | Actual K4/8untrained strips and four fitted adapters plus global/local CNN controls run; no outer gain above raw frozen ImageNet. K4train/transformer tokens/all-fold confirmation pending; no anatomy claims. |
|5. Dense JEPA是否真正增加身份区分信息？ | Pending actual local-token prediction, identity retrieval and noPrediction/noID controls. |
|6. Dense JEPA是否超越普通时序Transformer？ | Pending matched local TemporalTransformer/contrastive/noEMA experiments. |
|7. 新Joint Decoder能否处理NULL和多轨迹竞争？ | Real full-set attention and private-NULL one-to-one assignment implemented/tested. NULL/existing tradeoff and complete-video degradation measured; successful robust handling is not yet established. |
|8. 是否超过同输入普通Hungarian？ | First full-ID heads do not. Bounded SN015 gain is exactly reproduced by uniform+0.03cost control; no learned joint-identity superiority retained. |
|9. 运动/遮挡模块是否有独立贡献？ | Pending isolated Kalman/UKF/FishIoU/AGCIoU/observable occlusion experiments. |
|10. HOTA、AssA、IDF1、MOTA、IDSW是否稳定改善？ | Not established. Seed0bounded changes differ across videos and have threshold confound; full-ID routes sharply degrade. All-fold/3seed positive evidence pending. |
|11. 三种子变化如何？ | Visual global/local first-fold3seeds complete; seed0localadvantage notstable. C probes seed0only; otherfold/module/nativeMOTthree-seed confirmation pending. |
|12. 跨视频表现是否一致？ | First conditional SN001/SN015 effects are inconsistent; six-video encoder/association folds pending. |
|13. 对未知场景的泛化证据是什么？ | No new unknown-scene evidence yet. Old RGB backbone exposed to all6; first visual fits exclude both outer/inner and have no outer baseline gain. More folds, external data/license/overlap and independent heldout protocols pending. |
|14. 运行速度和实际成本是多少？ | CPU cache/head time and parameters measured in ONLINE_RUNTIME; complete raw roughly1.2FPSbaseline bottleneck/profile and new encoders remain pending. |
|15. 哪些模块是真正必要的？ | Current learned bounded head is unnecessary to reproduce SN015 output; full no-attention/noNULL/noMotion/noMemory/local/JEPA independent necessity controls pending. |
|16. 哪些负结果有科学价值？ | Native-score versus pure-rank decomposition, scene/history-dependent GRU, full-ID NULL/existing tradeoff, exact untrained threshold reproduction, matched-pool crop damage and negative first visual fitting are discriminating evidence with targeted follow-ups. |
|17. 30篇参考中哪些方法实际迁移成功？ | Independent full-set/IDdictionary ideas implemented, but no completed successful official-method transplant claim.16/30source-field audits complete (including newSU-T,TIDE,MOTIP,FDTA,Fish_Re_Identification,FishMAE),30exactheads recorded. Independent ImageNet local controls run; official weight reproduction and14audits remain pending. |
|18. 哪些已有工作可能削弱我们的创新主张？ | MOTIP full-set/local-ID competition, FDTA spatial/temporal/identity refinement, DINO patch/EMAdistillation and JEPA prediction are relevant overlap. 16/30semantic audits complete; remaining14novelty audits and stronger baselines pending. |
|19. 最终是否形成独立方法论文的可靠证据？ | Not yet. Full A/B/C/Dcontrolled positive evidence, all-fold/three-seed/raw online performance and final novelty assessment remain required. |
|20. 最终推荐使用哪套架构，为什么？ | Final recommendation pending. Keep the frozen native baseline as current reference while developing visual/local evidence, genuine DenseJEPA, repaired joint competition and independent motion with matched controls. |

The left column preserves the user's20final questions. Continue source auditing, legal visual foundation, local identity learning, actual DenseJEPA, repaired joint training and independent motion, then selective combinations, all-fold/three-seed complete-video and raw online tests. Failed hypotheses stay recorded with discriminating follow-ups; neither partial audit nor seed0head results close the task.

The executed crop/retrieval/training evidence is hash-linked in `PHASE_VII_VISUAL_PILOT_MEASURED.json`; first-fold data and trained weights stay outside Git. No positive paper contribution or completed final-goal claim follows from these pilots.

Fitting/inner stage diagnosis separates raw2048 from projected/preLN/LN256; LayerNorm removal barely changes ranks, while inner raw local backbone16.35% versus head13.94% motivates raw evidence/domain/crop controls. No outer parameter is selected from this diagnostic. Real state-equivalent restart is verified; all scientific controls still need their full required protocols.

## Source audit continuation: sixteen completed field audits

Sixteen of30 references now have all11 source-level semantic fields and exact commit/blob/SHA reading proofs: ByteTrack, BoT-SORT, HybridSORT, DINOv2, I-JEPA, V-JEPA2/2.1, GOT-JEPA, CoTracker3, TransReID, CLIP-ReID, SU-T, TIDE, MOTIP, FDTA, Fish_Re_Identification and FishMAE.326listed source/config/license paths have exact provenance and targeted reading records; this is not an exhaustive every-line reading claim or official pretrained reproduction. Weight delivery, unverified derivative/data terms and execution exclusions remain explicit in REPOSITORY_REGISTRY.

GOT-JEPA's registered head5f382645 also contains newerGOT-Edit; the README's paper release84e9324317 was independently materialized from existing local Git objects into a new isolated snapshot. Both actor versions define the same callable twice and the active last body ends with an unassigned tuple expression instead of the earlier loss/statistics return. Target/context feature-extractor parameters are shared; noEMA teacher update is implemented. Its latent target is a256-dimensional tracking filter, not Kfish identity tokens. Static AST and canonical source proofs are external/hash-linked; no foreign code was executed.

V-JEPA2/2.1 ordinary encoders attend bidirectional video clips. AC predictor's frame-causal mask does not establish causal backbone features.2.1Base's published train config selects a model name omitted from the train entry's embed-dimension branch; pretrained hub inference is a different path. CoTracker3's online wrapper initializes query/state separately and advances16frame windows by8; earlier window outputs can use later frames in the same window. Current-frame MOT needs explicit prefix boundaries, delay/emission policy and runtime accounting. Original full-video offline caches are excluded.

TransReID and CLIP-ReID static evaluators remove same-ID/same-camera positives, which would remove within-video fish positives if camera=video. Fish adapters must use physical(video,fish) identities and the established past-only unknown-preserving retrieval masks. Original JPM four branches are hardcoded, and CLIP prompt labels are training-class anchors rather than necessary onlineGT. CLIP stage1 actual contrastive temperature is1.0, not assumed.07; the final cache minibatch requires an empty-batch guard when exactly divisible.

The complete task book explicitly requires first completing environment/all30 source audits before further method development. Earlier executed visual/joint controls remain evidence, with their negative/confounded results preserved. The next priority is completing the remaining14 source audits before new B/C/D architecture work. Full goal remains IN_PROGRESS; actual localDenseJEPA training, stronger lawful pretrained baselines, repairedsetcompetition, independentmotion and all-fold/three-seed/fullrawonline remain required. See PHASE_VII_SOURCE_AUDIT_MEASURED.json for exact counts and artifact hashes.

COVTrack source delivery now succeeds at exact9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b, bringing exactsourcepins to30/30. Its semantic audit remains one of14pending; this doesnot claim allreferenceauditsoranyalgorithm complete.
