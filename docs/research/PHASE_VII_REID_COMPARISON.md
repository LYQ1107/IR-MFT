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

Next: stratify correct/mixed/unknown retained histories and common candidate support, construct the explicit two-crop identity diagnostic platform, and complete30 source audits before choosing visual/local/joint training architectures. Do not begin large JEPA training from the pure-rank discrepancy alone. PairAUC/EER/mAP/R1/R5 and gap/density/overlap-proxy breakdowns are stored in PHASE_VII_A0_STARTUP_MEASURED.json. Overlap/gap are proxies, not ground-truth occlusion or turning labels.

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
