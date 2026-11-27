# Confidence/shape association: completed fixed factorial

Original RGB-only YOLOX-X+ByteTrack baseline stays29.5289706874 HOTA, full Test7/15,985 frames. Minimum34.5289706874, stretch39.5289706874. Fixed November DiffV2IR, ProtocolV2 and seed0; strongestRGB/crossarchitecture sweeps remain deferred.

Four fixed cases combine original/candidate-scale matching with original/no score fusion. Candidate-scale matching uses the predicted track center and current candidate width/height at all three matching stages. Original observations, score partitions, birth/update rules, template reliability, trained heads/calibrations/archive and PP publication remain fixed. Changed associations may change future states and identities. Exact-original-state/MOT equivalence applies to the reference; each alternative publisher retains its own main observations.

Three regressions and six real-Train64-frame checks passed, including literal reference state/MOT and32-frame prefix invariance under future-input corruption. Warm all27,251 Train frames, score its last6,814; full Dev covers4,834. Require validation gain>=0.1 over selected PP and no Dev degradation within1e-10, then rank validation HOTA and Dev HOTA/AssA/IDF1/identifier. Freeze before any new Test. Controls cannot be promoted from Test.

| Case | Forward Train HOTA | Gain over PP | Dev HOTA |
|---|---:|---:|---:|
| DA_reference | 46.227954728 | +0.000000000 | 52.113114092 |
| DA_without_score | 45.502757214 | -0.725197514 | 50.474963933 |
| DA_measured_scale | 45.342967623 | -0.884987105 | 50.012916071 |
| DA_both | 45.052514520 | -1.175440208 | 48.476434725 |

No candidate qualified; no new Test run occurred. Prior selected PP remains32.4955880280 (+2.9666173406), below+5.

The supplemental geometry check compares147 equal-size original/candidate-scale cost entries across three sizes and fractional/integer displacements: all match exactly in this runtime. The first probe had an incorrect prewritten convention-mismatch narrative even though both numeric touching-edge overlaps were0. That narrative is retracted, with the original note and corrected numeric check both retained. No inference code, thresholds or candidate selection changed.

Independent checks reparse8 original276-column CSV tables and1880 official47 values, validate32 canonical MOT streams and reconstruct original references, GT windows, cleanup and prospective selection. All17 jobs and controller have actualexit0. It is not a second visual producer or complete independent alternative-association/state replay.

No new independent IR information is introduced by these classical geometric variants. They cannot by themselves establish multimodal novelty. Historical Test and prior PP results were exposed; legacy encoder Train identities, inspected development windows and the older RGB detector’s Dev exposure remain disclosed. Generated pseudo-IR is RGB-derived, not physical thermal.

Only small code/configuration/official aggregate and per-sequence47 values/raw CSV/summary/logs/provenance are published. Canonical MOT, NN banks, images, weights and GT stay private. No new NN training or GPU work; physicalGPU4/5 maximum two if required. New private cap256MiB inside64GiB work cap with40GiB reserve. Cached costs exclude original neural/image/generation costs and are not end-to-end FPS.

**Goal remains active pending final assessment; independently supported new multimodal innovations:0.**
