# Phase VII identity evidence: first measured decomposition

Status: IN_PROGRESS. Inherited raw/GRU and same-supported retained-cue metrics were recalculated; all six videos exactly match the original scientific metric payloads. This is actual accepted-history detector-feature diagnosis with inherited exposure, not completed GT/detector-crop ReID training or new online MOT evidence.

| Video | Known-positive queries | Raw mean R1 | GRU pure R1 | Geometry row R1 | Real blended row R1 | Cost reconstruction max error |
|---|---:|---:|---:|---:|---:|---:|
| BT-001 | 4223 | 58.5603 | 56.5712 | 60.2652 | 60.2889 | 1.19e-07 |
| BT-003 | 363 | 87.8788 | 87.3278 | 89.2562 | 89.2562 | 5.96e-08 |
| MSK-002 | 8537 | 45.4726 | 42.0288 | 55.9447 | 55.9447 | 1.19e-07 |
| PF-001 | 5742 | 60.5190 | 96.5343 | 60.6583 | 60.6757 | 1.19e-07 |
| SN-001 | 967 | 37.1251 | 73.0093 | 37.1251 | 36.9183 | 5.96e-08 |
| SN-015 | 10710 | 62.3623 | 35.8450 | 62.3249 | 62.2969 | 1.19e-07 |

SN-015 uses the same10,710 known-positive queries for the first pure-ranking contrast. The separate retained-cue support uses8,631 queries: mean76.989920%, GT-selected retained-cue oracle83.489746%. Oracle cue filtering is offline analysis only; unknown history does not prove wrong identity or irrecoverability.

The GRU pure rank is35.845005%, while real geometry/confidence blending gives62.296919%, close to geometry-only62.324930%. Reconstruction reproduces real recorded costs within1.2e-7 on SN-015. Feature norms are within about1.2e-7 of1, and normalized prototype similarities reproduce saved evidence. Thus normalization or a different checkpoint does not explain this contrast; the tracker uses a different score and the geometry prior can largely dominate these fixed-state row decisions. This does not prove that removing GRU preserves full endogenous tracking: Hungarian competition, thresholds, later stages, births and changed histories are outside row retrieval.

PF-001 pure GRU96.534309% versus mean60.519% contradicts a blanket claim that the GRU always discards useful identity evidence. The residual GRU also moves far from original normalized history: SN-015 mean cosine to last0.22285 and to FIFO0.19086. These are measured outputs, not proof of the unique causal source of degradation.

Training source tools/cfp_build_prediction_queries.py uses an earlier ordinary RGB-tracker producer, known unique current identities, positive-present filtering, sampled past cutoff gaps1/5/15/30 and at least two same-frame IDs. The current diagnostic uses historical GRU-produced actual state graphs, retains unknown candidates and evaluates naturally missing histories/candidates separately. SN-015 was calibration for the temporal model and seen by the frozen visual encoder. Candidate label support, generating tracker, requested temporal cutoff and scene exposure differ; their individual contributions remain to be discriminated.

Correct/mixed/unknown retained histories and common native appearance-region support have now been measured. The complete two-crop metadata index is available; the sparse paired-crop visual pilot is completed below; full-frame/new-encoder MOT and all-fold confirmation remain pending. Complete30semantic audits before final method/novelty selection. The independently implemented first joint/pair probes are explicit task-book controls, with their cost/threshold confound recorded separately. Do not begin large JEPA training from the pure-rank discrepancy alone. PairAUC/EER/mAP/R1/R5 and gap/density/overlap-proxy breakdowns are stored in PHASE_VII_A0_STARTUP_MEASURED.json. Overlap/gap are proxies, not ground-truth occlusion or turning labels.

Original checkpoint-matched temporal training query re-evaluation is complete on all6videos in PHASE_VII_A0_ORIGINAL_PRODUCER_MEASURED.json. On PF-001, GRU46.0896%on ordinary actual histories versus96.8361%on AnchorRecent; on SN-01545.1092%versus28.3419%. Clean FIFO/GRUare93.2634/60.3093%on PF-001and92.8724/71.0101%on SN-015. These original producer/pool/cutoff denominators differ from warm GRU-produced queries, so the change is evidence of scene/history dependence, not a paired causal producer effect or new visual encoder gain.

Further actual-history support and appearance-region diagnosis (v2)

| Video/support | Known-positive queries | FIFO R1 | GRU R1 |
|---|---:|---:|---:|
| BT-001/all_verified_correct | 1069 | 96.3517 | 88.6810 |
| BT-001/some_verified_correct | 2446 | 57.8496 | 56.9501 |
| BT-001/no_verified_correct_unknown_possible | 708 | 3.9548 | 6.7797 |
| BT-003/all_verified_correct | 178 | 99.4382 | 100.0000 |
| BT-003/some_verified_correct | 182 | 76.3736 | 75.2747 |
| BT-003/no_verified_correct_unknown_possible | 3 | 100.0000 | 66.6667 |
| MSK-002/all_verified_correct | 2986 | 79.1360 | 72.6390 |
| MSK-002/some_verified_correct | 4916 | 30.2278 | 28.1530 |
| MSK-002/no_verified_correct_unknown_possible | 635 | 5.1969 | 5.5118 |
| PF-001/all_verified_correct | 2993 | 99.9332 | 99.2315 |
| PF-001/some_verified_correct | 2259 | 20.8942 | 94.5994 |
| PF-001/no_verified_correct_unknown_possible | 490 | 2.4490 | 88.9796 |
| SN-001/all_verified_correct | 180 | 97.2222 | 96.1111 |
| SN-001/some_verified_correct | 709 | 25.6700 | 68.4062 |
| SN-001/no_verified_correct_unknown_possible | 78 | 2.5641 | 61.5385 |
| SN-015/all_verified_correct | 2716 | 99.5950 | 52.6510 |
| SN-015/some_verified_correct | 5915 | 66.6103 | 38.2079 |
| SN-015/no_verified_correct_unknown_possible | 2079 | 1.6354 | 7.1669 |

On SN-015 all2,716 actual histories whose retained proposal labels are verified correct, FIFO99.5950% versus GRU52.6510%. Contamination therefore cannot by itself explain the weak isolated GRU representation on this subset. A restricted native appearance region has7,081 positive-present queries, FIFO94.1110% versus GRU84.5078%. Changing the candidate pool changes both problem difficulty and denominator; these94/84 values must not be directly compared to the10,710-query62/35 values as a method improvement. The remainder is consistent with scene/target/producer distribution shift, but does not uniquely prove one mechanism.

## Completed sparse visual and crop pilot

All six training videos were encoded at preregistered frame residues1/2 modulo20, preserving same-frame sets and temporal galleries:47,751 current GT/detector observations,3.462GiB of features. This is sparse diagnosis, not all-frame runtime. Every selected crop feature is available. Historical FishReID was exposed to all six videos and is a diagnostic reference. Frozen ImageNet and every new fit use the exact ImageNet ResNet50 checkpoint SHA0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a. Matching old detector post-BN features preserves validity, with maximum component difference0.001208 and cosine difference3.756e-6; AMP batch-composition parity is numerical, not byte equality.

Rank@1 percent below uses past gap>=25 frames. Actual-view pools retain unknown detector identities; all representations share each view's queries and masks. GT and detector pools differ, so that table alone cannot isolate crop pixels.

| Video | GT ImageNet | GT FishReID BN | GT local8 visibility | Detector ImageNet | Detector FishReID BN | Detector local8 visibility |
|---|---:|---:|---:|---:|---:|---:|
| BT-001 | 20.952 | 43.016 | 41.587 | 14.744 | 20.513 | 23.237 |
| BT-003 | 12.727 | 97.727 | 87.273 | 17.514 | 40.678 | 42.373 |
| MSK-002 | 13.134 | 49.671 | 37.360 | 4.398 | 5.556 | 6.647 |
| PF-001 | 78.571 | 78.571 | 82.540 | 70.635 | 79.365 | 84.921 |
| SN-001 | 34.192 | 83.677 | 77.320 | 32.308 | 42.500 | 47.692 |
| SN-015 | 54.101 | 68.612 | 70.978 | 53.651 | 66.667 | 68.254 |

The paired comparison joins unique known(video,frame,fish)keys and uses identical keys, queries, galleries and temporal masks for both crop views. All six have zero excluded nonunique pairs. This is conditional on the original detector/GT physical-identity verification rule, not unrestricted perfect identity supervision.

| Video | Matched known keys | Queries | GT FishReID R1 | Detector FishReID R1 |
|---|---:|---:|---:|---:|
| BT-001 | 2229 | 640 | 42.468 | 26.442 |
| BT-003 | 200 | 200 | 97.175 | 53.672 |
| MSK-002 | 5683 | 3136 | 49.008 | 8.201 |
| PF-001 | 2872 | 128 | 86.508 | 81.746 |
| SN-001 | 561 | 560 | 85.385 | 50.000 |
| SN-015 | 5860 | 644 | 75.079 | 71.587 |

MSK002's large crop-view loss survives pool control. Padding, box extent, low resolution, foreground/background mixture and appearance changes are candidate mechanisms, not separately identified causes. No GT crop or physical label is supplied to online tracking.

Six real fits completed at fixed epoch8, seed0, outerSN015/innerBT001, fitting only BT003/MSK002/PF001/SN001. Four stopped-ImageNet adapters compare global, equal-two-projection dualglobal capacity control, simple local pooling and learned weighting. Two fresh ImageNet-initialized ResNet50 models actually update spatial backbone gradients and BN statistics only on the four fitting videos. They share the same supervision: composite physical-ID CE, known temporal positives and same-frame different-fish negatives.

| Representation | Inner BT001 GT | Inner BT001 detector | Outer SN015 GT | Outer SN015 detector | Allocated parameters | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| Frozen ImageNet raw2048 | 20.952 | 14.744 | 54.101 | 53.651 | 23,508,032 backbone | — |
| local_adapter_global | 17.619 | 13.622 | 46.530 | 43.175 | 1,084,609 | 19.91 |
| local_adapter_dual_global | 17.937 | 13.462 | 46.530 | 43.333 | 1,084,609 | 20.13 |
| local_adapter_simple | 16.825 | 12.500 | 47.319 | 45.079 | 1,084,609 | 19.52 |
| local_adapter_learned | 18.254 | 13.782 | 43.849 | 44.921 | 1,084,609 | 23.75 |
| visual_identity_global | 20.159 | 12.019 | 45.584 | 42.698 | 24,592,641 | 99.23 |
| visual_identity_learned | 20.317 | 13.942 | 43.849 | 46.667 | 24,592,641 | 101.68 |

No first-fold new fit beats frozen ImageNet on outer detector retrieval. A local-versus-global improvement within the trained CNN pair is insufficient to claim a retained baseline gain. This is one fold/seed, not all-fold/three-seed confirmation or new-ReID MOT evidence. Fit loss decreases do not certify generalization. Diagnose raw fine-tuned2048 versus projected/pre-normalization256 and LayerNorm256 on fitting/inner data, then crop stability/domain robustness. Exposed outer scores cannot select the next checkpoint or hyperparameters under an untouched-outer claim.

Full source/data/config/checkpoint hashes and all gap1/25/100 R1/R5/mAP, actual-pool AUC/EER, training updates/AMP skips and masks are retained in `PHASE_VII_VISUAL_PILOT_MEASURED.json`. Atomic model/optimizer/scheduler/AMP/all RNG checkpoints exist outside Git; actual interrupted-resume equivalence remains untested. A later empty-view/unavailable-gallery helper fix preserves historical run snapshots; all pilot features were available, so it does not alter these scores. The256-dimensional new representation cannot use the inherited512-dimensional GRU.
