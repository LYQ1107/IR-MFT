# Coverage-triggered weak recall: complete registered study

The primary comparator remains the original direct-RGB YOLOX-X + ByteTrack, **29.5289706874 HOTA** over the complete historical Test7 / 15,985 frames. Minimum target **34.5289706874**; stretch target **39.5289706874**. Strongest-RGB and cross-architecture sweeps remain deferred under the user’s revised scope.

The new method uses the unchanged original teacher’s confirmed unused proposal mass to trigger recall. It retains original detector scores/boxes/order, completes existing primary/secondary and high-score pending associations first, then permits remaining proposals above 0.3 with at least three independent original-score shadow confirmations to start tracks. Recall-born pending tracks can confirm against remaining low detections. Extra tracks can change future candidate graphs. Ordinary thresholding, confirmation and ByteTrack extensions do not receive novelty credit by themselves.

The prospective standalone validation warms every Train sequence from frame 1 and scores only its last quarter: 6,814 scored frames and 27,251 past/current input frames. It excludes the old late-calibrated birth-quality/archive thresholds. Coverage classifiers and foreground directions were fitted on first-75-percent Train data; old ReID encoders have already seen Train identities. This tests the new controller forward in time, not an unseen encoder/domain. Full Dev uses the old selected BO_paired_q999_c3 parent and its existing Train calibration.

The selection rule was committed before new HOTA results: eligible paired cases need at least +0.1 standalone-validation HOTA and full Dev HOTA ≥52.11311409153856 (tolerance 1e-10). Exact Dev ties are prospectively allowed. Rank by validation HOTA, then Dev HOTA/AssA/IDF1 and identifier. RGB/duplicated/count/unconditional controls cannot be promoted after Test. The previously completed conditional-quality study remains closed.

| Case | Standalone validation HOTA | Gain | Full Dev HOTA | Eligible |
|---|---:|---:|---:|---|
| CR_reference | 44.920663037 | +0.000000000 | 52.113114092 | False |
| CR_unconditional | 39.716702212 | -5.203960826 | 38.714996234 | False |
| CR_count_t01 | 42.279332847 | -2.641330190 | 51.441007628 | False |
| CR_RGB_t01 | 43.158283956 | -1.762379081 | 51.789290252 | False |
| CR_paired_t01 | 42.687230207 | -2.233432830 | 51.789117208 | False |
| CR_duplicatedRGB_t01 | 43.158283956 | -1.762379081 | 51.789290252 | False |
| CR_count_t025 | 43.725548742 | -1.195114296 | 52.229091146 | False |
| CR_RGB_t025 | 43.763006185 | -1.157656853 | 52.113114092 | False |
| CR_paired_t025 | 43.652499391 | -1.268163646 | 52.113114092 | False |
| CR_duplicatedRGB_t025 | 43.763006185 | -1.157656853 | 52.113114092 | False |

No paired case satisfied the registered validation/Dev gate. No new Test selection was frozen and no new Test result was produced. The previous completed formal result remains 30.7170810665 (+1.1881103791).

The conservative paired 0.25 candidate increases standalone validation DetA from 52.029192635 to 56.915313397, while AssA falls from 38.918866532 to 33.617528553. This is a recall/identity tradeoff, not a positive total HOTA result.

| Train validation sequence | Reference HOTA | Paired 0.25 HOTA | Difference |
|---|---:|---:|---:|
| BT-001 | 48.675944850 | 48.675944850 | +0.000000000 |
| BT-003 | 57.223397240 | 57.223397240 | +0.000000000 |
| MSK-002 | 35.259059833 | 39.335847885 | +4.076788052 |
| PF-001 | 62.915975747 | 47.651509036 | -15.264466711 |
| SN-001 | 57.305998731 | 57.305998731 | +0.000000000 |
| SN-015 | 42.004260398 | 42.004260398 | +0.000000000 |

A read-only follow-up chose the worst Train validation sequence, PF-001, then its first three preregistered 0.25 need triggers (frames 1065, 1083, 1306). The unused high candidates retain unknown labels (-1); they are not relabeled as background. Their maximum fractions of area covered by an existing measured fish box are 0.89880944, 0.68543055, and 0.74809882. The first two have IoU below 0.5. Thus IoU-only duplicate checks can miss boxes that mostly depict part of an already measured fish. RGB and fixed pseudo-IR frame 1065 were visually inspected without copying or pushing the images. Three frames support this geometric explanation; they do not independently prove the cause of the full identity loss.

The next hypothesis is to keep the trusted teacher unchanged, handle unexplained proposals in an independent peripheral stream, and test spatial explanation plus one-to-one identity handoff. This is a proposed next study, not an implemented or validated result in this report. The completed failed study remains closed; it cannot be advanced by changing its gate retroactively.

Independent audit verified 29 real worker exits, reparsed 20 official tables (4700 aggregate/per-sequence fields), checked 80 original trajectory hashes/formats, and rebuilt 80 gate activation counts from the predecessor’s private scalar diagnostic bank. It reconstructed the prospective selection. Its scope excludes a second visual producer, independent Kalman/LAP replay and end-to-end runtime.

The original detailed 276-column CSVs and 39-field summaries are preserved byte-for-byte. Source hashes, actual execution receipts, small raw logs, every sequence and all 47 metrics are included. Weights, images, descriptor/scalar banks, validation GT copies and MOT trajectory dumps are excluded. CPU replay/evaluation time excludes original neural/image/generation costs; it must not be reported as end-to-end FPS.

Fixed November DiffV2IR and ProtocolV2 remain unchanged, seed0 only, at most physical GPU4/5. This study reuses existing banks on CPU. Historical Test exposure, old detector Dev exposure and ReID Train-identity exposure remain disclosed. Generated pseudo-IR is an RGB-derived auxiliary view, not a physical thermal sensor.

**Goal remains active pending numeric and scientific requirements. Independently supported multimodal innovations: 0.**
