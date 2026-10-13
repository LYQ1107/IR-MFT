# Phase VII scientific research report

Status: IN_PROGRESS. This is a running evidence report; the EvidenceFishMOT final goal is active and unachieved. No new positive research contribution is asserted from the first probes.

The full goal/task book and existing PhaseVI assets are preserved on `research/evidence-fishmot-v1`. Completed work includes exact inherited six-video rank/cue reproduction, actual-cost and history-support diagnosis, original producer/history re-evaluation, independent full-set ID/NULL implementation, registered joint/pair training, native complete-video replay, an untrained residual identifiability control and both-crop metadata foundation,47,751crop encoding/paired retrieval and six real visual fits. At the last project-wide check216tests pass with18warnings.

The key first scientific finding is a measured confound: the bounded SN-015 HOTA increase31.9832→33.1384is exactly reproduced by an untrained uniform+0.03cost change, with identical prediction bytes. Both complete learned ID routes reduce HOTA and increase switches. This is actionable evidence for preserving raw/global/local identity cues, diagnosing NULL/existing tradeoffs and testing genuine set competition on future matched controls. It does not terminate the research direction or justify a fixed-count NO_GO.

| Final question | Current evidence / remaining work |
|---|---|
|1. 当前FishReID的主要缺陷是什么？ | Actual-state global features retain strong clean identity evidence; temporal compression/history and producer dependence are measured. Matched-pool GT/detector and raw/preBN/BN/local diagnosis now run: crop damage is severe on MSK002, smaller on PF001/SN015. No blanket backbone-defect claim. |
|2. 新ReID改善了多少身份检索？ | Six first-fold seed0 fits complete. Outer detector R1=42.70–46.67% versus frozen ImageNet53.65%; no gain above that baseline retained. More folds/seeds/strong pretrained controls pending. |
|3. 新ReID是否改善真实在线HOTA/AssA？ | Pending. First new C heads use the inherited encoder; raw new-ReID online comparison has not run. |
|4. 局部Patch是否优于单一全局特征？ | Actual K4/8untrained strips and four fitted adapters plus global/local CNN controls run; no outer gain above raw frozen ImageNet. K4train/transformer tokens/all-fold confirmation pending; no anatomy claims. |
|5. Dense JEPA是否真正增加身份区分信息？ | Pending actual local-token prediction, identity retrieval and noPrediction/noID controls. |
|6. Dense JEPA是否超越普通时序Transformer？ | Pending matched local TemporalTransformer/contrastive/noEMA experiments. |
|7. 新Joint Decoder能否处理NULL和多轨迹竞争？ | Real full-set attention and private-NULL one-to-one assignment implemented/tested. NULL/existing tradeoff and complete-video degradation measured; successful robust handling is not yet established. |
|8. 是否超过同输入普通Hungarian？ | First full-ID heads do not. Bounded SN015 gain is exactly reproduced by uniform+0.03cost control; no learned joint-identity superiority retained. |
|9. 运动/遮挡模块是否有独立贡献？ | Pending isolated Kalman/UKF/FishIoU/AGCIoU/observable occlusion experiments. |
|10. HOTA、AssA、IDF1、MOTA、IDSW是否稳定改善？ | Not established. Seed0bounded changes differ across videos and have threshold confound; full-ID routes sharply degrade. All-fold/3seed positive evidence pending. |
|11. 三种子变化如何？ | First new probes are seed0only; three-seed confirmation pending. |
|12. 跨视频表现是否一致？ | First conditional SN001/SN015 effects are inconsistent; six-video encoder/association folds pending. |
|13. 对未知场景的泛化证据是什么？ | No new unknown-scene evidence yet. Old RGB backbone exposed to all6; first visual fits exclude both outer/inner and have no outer baseline gain. More folds, external data/license/overlap and independent heldout protocols pending. |
|14. 运行速度和实际成本是多少？ | CPU cache/head time and parameters measured in ONLINE_RUNTIME; complete raw roughly1.2FPSbaseline bottleneck/profile and new encoders remain pending. |
|15. 哪些模块是真正必要的？ | Current learned bounded head is unnecessary to reproduce SN015 output; full no-attention/noNULL/noMotion/noMemory/local/JEPA independent necessity controls pending. |
|16. 哪些负结果有科学价值？ | Native-score versus pure-rank decomposition, scene/history-dependent GRU, full-ID NULL/existing tradeoff, exact untrained threshold reproduction, matched-pool crop damage and negative first visual fitting are discriminating evidence with targeted follow-ups. |
|17. 30篇参考中哪些方法实际迁移成功？ | Independent full-set/IDdictionary ideas implemented, but no completed successful official-method transplant claim.3/30source-field audits complete(ByteTrack/BoT-SORT/DINOv2),29exactheads recorded. Independent ImageNet local controls run; official weight reproduction and27audits remain pending. |
|18. 哪些已有工作可能削弱我们的创新主张？ | MOTIP full-set/local-ID competition, FDTA spatial/temporal/identity refinement, DINO patch/EMAdistillation and JEPA prediction are relevant overlap. 3/30semantic audits complete; remaining27novelty audits and stronger baselines pending. |
|19. 最终是否形成独立方法论文的可靠证据？ | Not yet. Full A/B/C/Dcontrolled positive evidence, all-fold/three-seed/raw online performance and final novelty assessment remain required. |
|20. 最终推荐使用哪套架构，为什么？ | Final recommendation pending. Keep the frozen native baseline as current reference while developing visual/local evidence, genuine DenseJEPA, repaired joint competition and independent motion with matched controls. |

The left column preserves the user's20final questions. Continue source auditing, legal visual foundation, local identity learning, actual DenseJEPA, repaired joint training and independent motion, then selective combinations, all-fold/three-seed complete-video and raw online tests. Failed hypotheses stay recorded with discriminating follow-ups; neither partial audit nor seed0head results close the task.

The executed crop/retrieval/training evidence is hash-linked in `PHASE_VII_VISUAL_PILOT_MEASURED.json`; first-fold data and trained weights stay outside Git. No positive paper contribution or completed final-goal claim follows from these pilots.
