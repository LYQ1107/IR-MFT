> [Latest complete Dev and causal ambiguity status](PERCEPTUAL_MFT_20261127_TRACKLET_DEV_AND_AMBIGUITY_STATUS.md): late RGB identity evidence scores50.429366, +0.314156 over the identical IR visual parent; extra IR confirmation has no gain and retention250 hurts. The frozen full7 Test continues on4/5. Nine fixed conditional association policies reuse the sealed banks on CPU, without selecting on new Test scores. The verified full-Test gain remains+0.270555 versus original RGB29.528971; the+5 target is unmet. Older status paragraphs below retain their original observation times.

> [Causal tracklet R2](PERCEPTUAL_MFT_20261127_TRACKLET_PERMANENCE_R2_STATUS.md) passed six CPU regressions and actual engineering on both physical4/5. Full Dev proposal/identity extraction is now running; the detector and fixed November IR remain unchanged, with two explicit existing ResNet50 RGB/IR identity encoders for association. No new accuracy or independent innovation is established.

> Latest Fisher full Dev is negative: all four nonzero cases40.838753–47.221266 are below the identical visual parent50.115210; no new Test was run. [Complete results and all47 metrics](PERCEPTUAL_MFT_20261127_FISHER_FULL_DEV.md). The fixed original-RGB+5 goal remains unachieved.

# IR-MFT Phase VI final fusion report

> Current fixed-November primary result: full7 Test HOTA **29.799525**, original RGB **29.528971**, gain **+0.270555 points**; the +5 target is unmet. The latest human priority defers extra RGB structures/removals and prioritizes +5, with +10 as a stretch. See [the current full-Test analysis](PERCEPTUAL_MFT_20261126_NATIVE_PRIMARY_FULL_TEST_ANALYSIS_ZH.md). [Memory V3 completed full Dev with identical metrics and MOT bytes at all three weights](PERCEPTUAL_MFT_20261127_NATIVE_MEMORY_NO_GAIN.md), so no new Test was run. [The Train-only Fisher projection and five fixed Dev cases](PERCEPTUAL_MFT_20261127_FISHER_TRAIN_AND_DEV_STATUS.md) completed full Dev negatively; the original +5 target remains unmet. The original RGB CSV is unchanged; only a seven-millionths copied constant was corrected from that same CSV, with the original aggregation failure preserved.

> Historical Phase VI archive, not completion of the current November DiffV2IR goal. F1/F2/F3 below use the causal-cache EMA luminance-difference auxiliary representation. These results must not be relabelled as experiments on the fixed November DiffV2IR images. The current reference is direct RGB-finetuned YOLOX-X + ByteTrack, Test HOTA 29.529; the active target is at least 34.529 with 2–3 independently supported multimodal contributions. The current goal is unachieved. See [current status](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS.md), [all six matched nine-epoch fits](PERCEPTUAL_MFT_20261126_NATIVE_ALL_SIX_NINE_EPOCHS.md), [the current auxiliary-modality audit and remaining requirements](CAUSAL_AUXILIARY_MODALITY_AUDIT.md), and [the all-Train workflow](PERCEPTUAL_MFT_20261124_FULL_TRAIN_WORKFLOW.md). The historical numbers below are retained without changing their scope.

This report records the reproducible Phase VI screening through F3. The
repository is on branch `research/causal-multimodal-fusion-v1`; external
weights, cache frames, per-frame predictions, and TrackEval artifacts are
under `/data3/liuyeqiang/IR-MFT_assets` and are intentionally not committed.
The 20-source audit is in
[`OPEN_SOURCE_20_CODE_AUDIT.md`](OPEN_SOURCE_20_CODE_AUDIT.md), with the
machine-readable version beside it.

## Protocol and baseline

The fixed reference is the V3 static RGB-T fusion model evaluated with the
same detector, ByteTrack thresholds, image size, causal-cache auxiliary mode,
and TrackEval aggregation. V3 scores are HOTA 46.086 / DetA 71.164 / AssA
29.869 / IDF1 52.886 / MOTA 92.538 / IDSW 96 on Dev (4,834 frames), and the
historical Test reference is HOTA 31.091 / DetA 49.575 / AssA 19.619 / IDF1
35.044 / MOTA 52.325 / IDSW 560 on all 15,985 Test frames. These are the
comparison numbers; a learned candidate is counted as an improvement only if
it is better on the complete fixed split.

The split protocol is Train 6 sequences / 27,251 frames, Dev 2 / 4,834, and
Test 7 / 15,985. Test sequences are `BT-002`, `BT-004`, `MSK-003`, `PF-002`,
`SN-009`, `SN-011`, and `SN-014`. All F3 runs used physical GPU3 only because
GPU2 was occupied by an unrelated job; this stays within the requested GPU2/3
limit.

## Candidate screening

| candidate | Dev HOTA | Test HOTA | Test DetA | Test AssA | Test IDF1 | Test MOTA | Test IDSW | assessment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V3 static reference | 46.086 | 31.091 | 49.575 | 19.619 | 35.044 | 52.325 | 560 | fixed reference |
| F1 dynamic quality | 13.049 | 16.529 | 27.106 | 10.328 | 12.562 | -48.971 | 2,422 | negative / gate collapse |
| F2a quality + spatial | 14.742 | 16.501 | 28.268 | 9.785 | 12.983 | -40.636 | 2,224 | negative |
| F2b quality + spatial | 14.332 | 15.679 | 27.131 | 9.264 | 12.002 | -47.678 | 2,430 | negative / gate collapse |
| F2c quality + spatial + redundancy | 13.544 | 16.270 | 29.073 | 9.283 | 12.467 | -34.716 | 2,432 | negative / redundancy collapsed |
| F3 temporal, stateful | 46.889 | 30.087 | 47.972 | 18.977 | 32.706 | 50.960 | 517 | Dev-only gain; Test negative |
| F3 temporal, disabled history | 46.888 | **30.306** | **48.029** | **19.243** | **33.315** | 50.883 | **514** | best F3 Test policy, still below V3 |
| F3 temporal, reset each frame | 46.888 | **30.306** | **48.029** | **19.243** | **33.315** | 50.883 | **514** | identical to disabled |

The complete Test result does not support a positive claim for the current
multimodal innovations. The best F3 state policy is below V3 by 0.785 HOTA,
1.546 DetA, 0.376 AssA, 1.729 IDF1, and 1.442 MOTA points, although it
reduces IDSW by 46. The stateful recurrent policy is another 0.219 HOTA and
0.609 IDF1 below the disabled/reset policies. On Dev, F3 stateful gains 0.803
HOTA and 0.949 AssA over V3, but loses 0.328 IDF1 and increases IDSW by 5;
the gain does not transfer to Test.

## F3 implementation and training

F3 is `TemporalComplementarityFusion` initialized from the V3 static fusion
checkpoint. It is an isolated temporal-contribution experiment, so it does
not claim to be a composition of temporal fusion with every earlier candidate.
The implementation adds:

- `yolox/data/datasets/mot_rgbt_clip.py`: sequence-safe contiguous clip
  windows, fixed clip padding, and a clip collate function;
- `tools/train_temporal_fusion.py`: three-epoch clip BPTT with state carried
  within clips and detached/reset between clips, while detector branches stay
  frozen;
- stateful/reset controls in `tools/eval_mft26ir.py` and forwarding helpers in
  the YOLOX/PAN-FPN model;
- regression tests for state graphs and clip-start coverage.

Training used clip length 4, stride 4, batch size 64, 3 epochs, and the
strict causal-cache auxiliary mode. All 27,251 Train frames are covered by
6,814 clips; the tail policy processes 27,256 frame positions per epoch
because five tail positions are deliberately repeated. Mean clip loss was
3.8414, 3.8356, and 3.8413 for epochs 1–3. The external final checkpoint is
748,350,655 bytes with SHA256
`17ad5f815407cd0941da2028a0c5e32cb9bccc67d58fee6eeedf499b409f0ff7`.

The state controls are interpreted precisely as follows. `stateful` carries
the hidden state through a sequence. `disabled` turns off history but still
applies the current-frame temporal correction. `reset_each_frame` applies the
same correction after clearing history before every frame. Disabled and reset
are exactly equal on Dev and Test at the reported precision, showing that the
recurrent history is not the source of the small Dev signal. F3 Test gate
means for stateful are 0.08927 / 0.00336 / 0.00278 at P3/P4/P5; disabled and
reset are 0.10258 / 0.00473 / 0.00282. Redundancy is zero in every F3 run.

## Reproducibility and storage policy

The lightweight, committed evidence is the code, tests,
[`PHASE_VI_EXECUTION_STATUS.md`](PHASE_VI_EXECUTION_STATUS.md), and
[`PHASE_VI_F3_RESULTS.json`](PHASE_VI_F3_RESULTS.json). The external result
directories contain the raw protocol JSON and TrackEval summaries for every
F3 state/split; the JSON summary records their exact paths. No checkpoint,
causal-cache frame, raw prediction file, or TrackEval output is tracked in
Git. Smoke artifacts were removed after validation, and available storage was
checked throughout the long runs.

Validation completed before submission:

- `pytest -q tests/test_qra_fusion.py` — 6 passed;
- Python byte-compilation of all modified tools and model/dataset modules;
- protocol checks for exactly 4,834 Dev frames and 15,985 Test frames;
- `git diff --check` and a repository scan excluding the external asset root.

## Interpretation and next experiment

Phase VI now has auditable negative evidence for F1/F2a/F2b/F2c and F3. The
main failure mode is not lack of a temporal API: true contiguous clips and
explicit state controls work, but the learned residual is weak and its
recurrent history does not generalize. The next experiment should therefore
start from the complete V3 detector and train a jointly calibrated,
uncertainty-weighted residual with a held-out sequence-level validation split,
rather than adding another unconstrained gate to the current detector.
