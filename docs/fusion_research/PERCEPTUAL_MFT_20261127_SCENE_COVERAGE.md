# Scene-level coverage diagnostic: complete Train and Dev

The original RGB YOLOX-X + ByteTrack Test7 baseline remains **29.5289706874 HOTA**. The minimum target remains **34.5289706874**. The latest completed formal selected result remains **30.7170810665**, or **+1.1881103791** points. This diagnostic creates no new Test accuracy result.

All 27,251 Train frames and 4,834 Dev frames were processed. The unchanged original teacher and an independent low-threshold geometric shadow produce the current confirmed unused proposal mass. A deficit is the confirmed unused mass divided by that mass plus the original currently measured mass. Confidence sums are uncalibrated proxies, not expected fish counts or error probabilities.

Every sequence finishes its complete GT-free inference before GT is opened. The read-only Dev adapter cannot fit models, call the Train-only label interface on Dev, or label Test. Unknown/ignored proposals remain unknown. The analysis target is a local-IoU opportunity proxy; it is not a counterfactual HOTA gain.

At the preregistered deficit threshold 0.25, the target is a missed-opportunity ratio of at least 0.1. Columns below are frame counts.

| Split / sequence | Gate | Activated | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| train / BT-001 | count | 12 | 12 | 0 | 237 |
| train / BT-001 | RGB | 15 | 14 | 1 | 235 |
| train / BT-001 | paired | 15 | 14 | 1 | 235 |
| train / BT-001 | IR_roll | 16 | 15 | 1 | 234 |
| train / BT-003 | count | 4 | 4 | 0 | 26 |
| train / BT-003 | RGB | 4 | 4 | 0 | 26 |
| train / BT-003 | paired | 4 | 4 | 0 | 26 |
| train / BT-003 | IR_roll | 4 | 4 | 0 | 26 |
| train / MSK-002 | count | 894 | 870 | 24 | 568 |
| train / MSK-002 | RGB | 936 | 910 | 26 | 528 |
| train / MSK-002 | paired | 927 | 902 | 25 | 536 |
| train / MSK-002 | IR_roll | 908 | 886 | 22 | 552 |
| train / PF-001 | count | 55 | 2 | 53 | 0 |
| train / PF-001 | RGB | 54 | 2 | 52 | 0 |
| train / PF-001 | paired | 54 | 2 | 52 | 0 |
| train / PF-001 | IR_roll | 54 | 2 | 52 | 0 |
| train / SN-001 | count | 1 | 1 | 0 | 132 |
| train / SN-001 | RGB | 1 | 1 | 0 | 132 |
| train / SN-001 | paired | 0 | 0 | 0 | 133 |
| train / SN-001 | IR_roll | 1 | 1 | 0 | 132 |
| train / SN-015 | count | 0 | 0 | 0 | 213 |
| train / SN-015 | RGB | 0 | 0 | 0 | 213 |
| train / SN-015 | paired | 0 | 0 | 0 | 213 |
| train / SN-015 | IR_roll | 0 | 0 | 0 | 213 |
| dev / BT-005 | count | 0 | 0 | 0 | 19 |
| dev / BT-005 | RGB | 0 | 0 | 0 | 19 |
| dev / BT-005 | paired | 0 | 0 | 0 | 19 |
| dev / BT-005 | IR_roll | 0 | 0 | 0 | 19 |
| dev / SN-013 | count | 5 | 4 | 1 | 133 |
| dev / SN-013 | RGB | 0 | 0 | 0 | 137 |
| dev / SN-013 | paired | 0 | 0 | 0 | 137 |
| dev / SN-013 | IR_roll | 0 | 0 | 0 | 137 |

Paired confidence selects 927 MSK-002 frames, of which 902 meet the target, with 25 false positives and 536 false negatives. RGB selects 936 with 910 true positives. The paired signal is useful as a coverage proxy; the comparison does not establish a unique IR contribution. At threshold 0.25, paired and RGB select zero frames in both Dev sequences. Simple count selects five frames in SN-013. No sequence-specific threshold or Test observation enters the rule.

The independent audit recomputed scalar tables and count conservation for all 32,085 frames, verified producer/engineering actual exit zero and frozen source hashes, and checked the existing Train original-MOT/label hashes and opportunity counts. It did not independently replay the visual producer, frame GT matching, or tracker states. The private scalar table is only 165,054 compressed bytes and is excluded from Git. No new GPU computation was needed.

Engineering passed causal prefix checks on all six real Train sequences and literal RGB fallback checks. This supports the engineering properties only. The next closed-loop recall study has its own prospective selection rule: standalone forward Train validation of the new controller, followed by complete Dev noninferiority. It must not relax or reopen the previously failed conditional-quality study.

The earlier Test history is exposed. Existing ReID encoders have seen Train identities. The original RGB detector has legacy Dev exposure. Fixed November DiffV2IR remains a generated auxiliary view of RGB, not a physical thermal sensor. No new tracking, domain-generalization, innovation-count or runtime claim is established here.

**Goal remains active and unmet. Supported independent multimodal innovations: 0.**
