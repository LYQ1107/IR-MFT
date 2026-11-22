# Visual continuity and classical algorithms for new DiffV2IR MFT

Status: IN_PROGRESS. The target is at least +5 Test HOTA points over the direct
RGB fine-tune control, with two or three independently supported contributions.
No new accuracy result is established at registration.

The frozen [research plan](../../configs/perceptual_mft_20261122.json) retains
the November DiffV2IR images, Protocol V2, all seven Test sequences and the
existing official TrackEval. The current direct RGB control is YOLOX-X plus
ByteTrack, HOTA 29.529; the absolute minimum target is therefore 34.529. Any
further detector adaptation requires a matching RGB training-budget control.
Improvement from a generic backend must also be separated from the contribution
of the pseudo-IR input using the same backend with RGB and double-RGB controls.

The initial research questions concern figure-ground shape, motion continuity,
and object permanence during occlusion. They motivate structured RGB/pseudo-IR
descriptors, causal optical-flow transport with uncertainty, and competitive
association with protected appearance and shape memories. These are hypotheses,
not validated innovations. Each claim needs single-module, cumulative and removal
ablations, input interventions, and complete-sequence metrics.

Classical matching, Kalman filtering, Lucas-Kanade optical flow and appearance
EMA are prior art, not contributions by themselves. Relevant primary sources
are [OC-SORT](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.html),
[Deep OC-SORT](https://arxiv.org/abs/2302.11813), and
[OpenCV optical flow documentation](https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html).
Any proposed novelty must describe a concrete modality-specific mechanism and
its measured contribution beyond those ordinary controls.

The prior F1/F2 gate-collapse, F3 non-transferring temporal residual, and
CMUE/PLIM failures remain evidence against merely adding unconstrained gates
or calling generic identity memory a multimodal contribution. Historical
results using another pseudo-IR version are context, not substitutes for new
experiments on the November images.

Only physical GPU 2 and GPU 3 are permitted. The data3 workspace has about
1.5 GiB free at startup, so new assets go to the separately mounted local
filesystem at `/var/tmp/liuyeqiang_ir_mft_perceptual_20261122` (about 3.2 TiB
free), with a NAS archive. Existing images and detection caches are reused;
no full dataset copy is made. The work filesystem keeps a 40 GiB reserve,
and this campaign initially caps new assets at 64 GiB.

Selection uses Train and Dev. Every concrete configuration is recorded before
evaluation. A rejected hypothesis gets diagnosis and at least three targeted
repair rounds before the family is abandoned. Every frozen formal candidate
and required ablation gets the complete 15,985-frame historical Test, with
all official fields and per-sequence outputs. Test outcomes cannot choose
parameters or trigger selection of the best Test configuration.

The historical Test has already been exposed, and original RGB pretraining
included the current Dev videos. These limitations will remain explicit.
Only current and past images may enter inference; no GT input, future frames,
or revision of previously emitted outputs is allowed.

The first two Dev replay controls exactly reproduce HOTA 47.579 (RGB) and
46.236 (new DiffV2IR), each covering 4,834 frames. Both allowed GPUs passed
eight-frame extraction smoke checks. Nine mechanism tests cover known
translation, invalid flow, polarity-independent contrast, identity continuity,
future-evidence prefix invariance and supported single-gap output. The current
full repository suite passes 28 tests, including all nine mechanism tests.
See [validation](PERCEPTUAL_MFT_20261122_VALIDATION.json).

The [Dev diagnosis](PERCEPTUAL_MFT_20261122_DEV_DIAGNOSIS.json) separates
proposal recall, tracker output and identity discontinuities. At IoU 0.5,
BT-005 new-input proposal recall is 98.44% at confidence 0.1, 93.72% at 0.6
and 58.68% at 0.7; corresponding RGB values are 98.57/95.20/72.10%. The
0.7 birth threshold can therefore suppress candidate initialization even
when proposals exist. This does not by itself prove a new method: ordinary
lower-threshold controls are registered alongside current-image transport
support for a single missing observation. This support never updates identity
memory as though a detection existed, and never rewrites an earlier output.

Thirteen initial Dev configurations and seven diagnosis-driven repairs are
registered in `configs/perceptual_screen_20261122` and
`configs/perceptual_repair1_20261122`. Current features compare ordinary RGB
appearance, pseudo-IR versus RGB figure-ground crop views, separate shape
descriptors, RGB/IR adjacent-frame transport and protected-memory controls.
The preserved RGB identity encoder was fitted on the six Train videos/98
identities under the exact current protocol hash. Its ordinary appearance
benefit cannot be credited as a new multimodal contribution.

Human-vision motivation is limited to computational hypotheses: linking
successive observations of an object is motivated by
[object files](https://www.sciencedirect.com/science/article/pii/001002859290007O),
and comparing coherent motion is motivated by
[human motion-grouping evidence](https://pmc.ncbi.nlm.nih.gov/articles/PMC7533882/).
This work does not claim to reproduce the biological implementation. Ordinary
EMA, an appearance encoder, lower birth thresholds, or switching the tracker
alone receive no multimodal innovation credit.

Unlabeled Test feature preparation has started on GPUs 2/3 before any new
Test accuracy evaluation. It computes the same fixed current-frame views and
adjacent-frame motion as Dev, does not read GT, and does not select a model.
A separate fit bank recomputes 11,967 paired observations from 2,551 Train
frames/98 identities, with every Dev record explicitly excluded. It prepares
the registered train-only metric-adaptation family; its eventual fitting and
selection remain separate from Test feature computation.

The double-RGB descriptor view is an ordinary representation-capacity control,
not a retrained double-RGB detector. Same-proposal RGB backend controls retain
the fixed new-IR detector, whereas the full RGB-only control uses the actual
single-branch RGB detector. These scopes must remain distinct in conclusions.

The [ordinary metric control](../../configs/perceptual_metric_20261122.json)
is registered before fitting: matched RGB/IR inputs, seeds 0/1/2, at least
18 and at most 36 feature-training epochs, with complete Dev checks every six.
Its bounded residual starts at the preserved RGB representation and uses
Train-only normalization and identity prototypes. The RGB cue consumes neither
auxiliary views nor auxiliary quality. This family tests whether ordinary
metric learning, rather than a proposed multimodal mechanism, explains a gain.
The first full fit bank is complete: 11,967 observations, 2,551 frames, six
Train videos and 98 identities, zero Dev/Test fitting rows; SHA256
`803850d1a177a8528ffc166ffba4dbfa718e00254748b9565eb9ecc38bf9cc8f`.

The complete [first 20-row Dev screen](PERCEPTUAL_MFT_20261122_DEV_SCREEN.md)
is negative; no candidate exceeds direct RGB. The cost audit records 36
accepted zero-IoU pairs, including 17 proposal-level local identity changes
under the diagnostic GT mapping. These are diagnostic events, not TrackEval
IDSW. The custom backend also used a 30-frame age limit while original
ByteTrack scales its buffer to 25 at frame-rate 25. See the
[cost diagnosis](PERCEPTUAL_MFT_20261122_COST_DIAGNOSIS.json).

The next family preserves the original ByteTrack lifecycle and feasible
association graph and only ranks ambiguous existing edges with evidence.
Ordinary Train-only centering, PCA and Fisher controls test feature calibration.
The zero-cue wrapper and the unmodified-cue baselines exactly reproduce all
official Dev fields, not merely HOTA, on 4,834 frames. Thirty repository tests
pass. This is an engineering correction, not a claimed innovation. Twenty
concrete configurations were registered before their evaluations in
`configs/perceptual_safe_20261122`. Paired RGB/IR metric fits have separately
completed 24 epochs for each seed 0/1/2; the registered common stopping rule
now stops all six heads together. See the complete
[56 calibration and metric Dev controls](PERCEPTUAL_MFT_20261122_CALIBRATION_METRIC.md).
No new-input gain or multimodal innovation has been established.


The [foreground point family](../../configs/perceptual_transport_20261122.json)
replaces box-centre translation with direct correspondences between textured
body points. Lucas–Kanade operates on RGB colour; actual new pseudo-IR supplies
optional source/target foreground constraints. A local Lab-colour RGB mask is
a strong matched control. The detector frontend is fixed separately to RGB or
new-IR V3: RGB proposals plus new-IR geometric association still form a two-input
late-fusion MFT system. Original V3 remains a reported reference; switching its
frontend is not credited as a multimodal contribution.

Thirty point-based Dev configurations were registered before their first
accuracy evaluation. They separate box/RGB/IR/joint point ownership, both
source/target removals, one-lag versus multi-lag object observations, preservation
of previously co-visible objects, and matched 0.5/0.8/1.0 point weights. Rejected
IoU edges can be opened only by explicit confident current-image correspondence,
with those openings counted. This differs deliberately from the earlier safe
appearance ranker. No future image, GT or revised prior output enters inference.

Thirty-seven repository tests pass, including seven point geometry/tracker
checks. New checks establish exact original lifecycle with disabled or absent
point evidence, identity-correct synthetic crossing without box overlap,
future-cache prefix invariance, and preservation of two previously co-visible
objects under the original overlap duplicate rule. These are engineering
checks, not evidence of real-data HOTA improvement. Full Dev point extraction
uses only a bounded past-image buffer and compact vote arrays outside Git.


The [complete30-row point screen](PERCEPTUAL_MFT_20261122_POINT_SCREEN.md)
shows a generic RGB point/co-visible-memory Dev gain to48.486, but hard
pseudo-IR masks remain harmful. RGB-preserving conditional IR ownership and
supported candidate births are registered in56 paired Dev configurations.
The new residual zero-cue control must reproduce the original RGB point
tracker exactly; a tiny0.004-point change is not established contribution.

A separately registered [counterfactual pair-evidence family](../../configs/perceptual_pair_calibration_20261123.json)
addresses raw-vote miscalibration. A small shared function compares actual cue
evidence with its RGB-reference evidence, yielding a bounded signed log-odds
increment; cue-independent offsets cancel. The matched RGB capacity control
uses a second ordinary RGB ownership/crop view. Training uses only128 sampled
source frames in each of the six Train videos, four causal source/target lags,
and two levels of box perturbation. No Dev/Test label fits this model. Two
actual-new-IR Train extraction smoke checks passed on physical GPUs2/3.
This is a proposed mechanism, with no claimed tracking improvement yet.

Forty-nine repository tests now pass. Three inactive completed common12-epoch
detector continuation checkpoints have been SHA-verified on NAS and replaced
locally by symlinks, reclaiming5.27568464GiB. A bulk-copy I/O pause prioritized
the last Dev image producer, then resumed automatically; archival is complete.
The data3 filesystem now has about6.4GiB free. All new research arrays and
weights remain on the separate local work filesystem and outside Git.


The complete [56-row conditional-residual screen](PERCEPTUAL_MFT_20261123_RESIDUAL_DEV.md),
[48-row weak-recovery/cycle screen](PERCEPTUAL_MFT_20261123_RECOVERABILITY_DEV.md),
and [32-row ROI-registration screen](PERCEPTUAL_MFT_20261123_REGISTERED_DEV.md)
remain negative for a meaningful modality-specific gain. Registration improves
some IR-heavy variants, but not beyond matched strong RGB. Both RGB-only raw/
registered pairs have exactly equal output files, all39 parsed headline fields
and every276-column detailed CSV row. No tiny rounded HOTA difference is
counted as a contribution. Earlier shorthand references to40 official fields
should be read as all parsed official fields; the exact audit above reports
actual field counts.

The Train-only association bank is complete:2,863,358 candidate pairs,
87,121 positives from six Train videos. The [complete36-row first18-epoch
pair-calibration Dev family](PERCEPTUAL_MFT_20261123_PAIR18_DEV.md) shows
ordinary RGB-head improvement on the new-IR detector frontend (best49.499),
but unstable IR increments over raw point odds. Every head executed24 epochs;
the frozen lookback rule requires extending all six to30. This is a separate
ordinary control, with three head seeds and fixed detector/encoder weights.

The [fitted-RGB residual plan](../../configs/perceptual_strong_rgb_residual_v2_20261123.json)
keeps a Dev-selected fitted RGB scorer immutable and adds an error-focused cue
correction, paired against an equal-capacity extra-RGB correction. Its initial
RGB reference selection used all36 first18-epoch Dev rows and no Test values;
the selected new-IR frontend has mean48.961 across its three fixed RGB heads.
Later ordinary-head epochs remain separately reported controls. The new
Train-only appearance bank contains347,782 chronological candidate pairs and
10,590 positives; these sets are not simultaneous detector observations.
Absent point/spatial evidence is explicitly masked. No new image dataset is
created. All six residual heads must execute at least18 epochs.

All64 maintained repository tests pass, including exact preservation of the
entire fitted RGB scorer when the new residual is zero, disabled or unavailable,
strictly current-frame competition, and future-feature prefix invariance.
The [prebirth identity plan](../../configs/perceptual_prebirth_identity_20261123.json)
registers retained-ID consultation before any new ID is first returned.
An already emitted identity remains immutable. This is still a hypothesis.

All nine previously selected ordinary metric controls have complete7-sequence,
15,985-frame Test exports. Their accuracy values do not choose the later
families. New point and learned association families have not yet established
Test improvement. Supported multimodal innovations remain zero; the>=+5 HOTA
and2-3 independently supported contribution goal is unfinished. Only GPUs2/3
are used, new assets occupy about3.7GiB on the separate work filesystem,
and lightweight code, plans and analysis are pushed without arrays or weights.


All79 maintained tests pass. All six ordinary pair-calibration heads actually
completed36 epochs and all72 registered Dev snapshots finished. All12
frontend/cue/seed controls are frozen using complete Dev HOTA with earliest
checkpoint ties, then evaluated on all7 Test videos. Their Test outcomes are
not consulted for subsequent research choices. See [selection](PERCEPTUAL_MFT_20261123_PAIR_SELECTION.json).

The [complete36-row first18 fitted-RGB residual family](PERCEPTUAL_MFT_20261123_STRONG18_DEV.md)
contains a new-front IR result50.933 in seed0 but substantial seed variation.
That one result does not establish a robust modality benefit. Every residual
head executed18 epochs and the registered stopping rule forces all six to24.
[Matched shrinkage and current-RGB-uncertainty guards](../../configs/perceptual_residual_guard_20261123.json)
repair the same hypothesis; they are not additional innovations.

The [complete20-row prebirth diagnostic](PERCEPTUAL_MFT_20261123_PREBIRTH_DEV.md)
shows that large archives cause incorrect identity links. The new-front IR
parent50.933 exceeds its naive recovery variants. A separate
[missing-point repair](../../configs/perceptual_missing_point_repair_20261123.json)
addresses a specific unused-data path: the long appearance-only training regime
was unavailable to the inherited main matcher whenever valid point count fell
below4. Conservative existing-edge and explicitly counted opening controls are
registered before replay. No previous identity output can be changed.

[Actual paired crop views](PERCEPTUAL_MFT_20261123_CROP_DIAGNOSIS.png) show
examples of distorted generated geometry despite near-saturated contrast
quality. This is a visual diagnosis, not an estimate of dataset-wide frequency.
Frozen unmasked RGB feature pooling avoids pixel-mask distribution changes,
but its [six-video Train-only spatial audit](PERCEPTUAL_MFT_20261123_SPATIAL_TRAIN_AUDIT.json)
still shows no useful raw/counterfactual identity-AUC gain at gap128. These
Train identities were seen by the encoder, so even a positive result would
not be held-out tracking evidence. Part pooling is established prior art,
including [PCB/RPP](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Yifan_Sun_Beyond_Part_Models_ECCV_2018_paper.php)
and [part-aligned bilinear representations](https://openaccess.thecvf.com/content_ECCV_2018/html/Yumin_Suh_Part-Aligned_Bilinear_Representations_ECCV_2018_paper.html).

The [native new-IR representation repair](../../configs/perceptual_native_cue_20261123.json)
therefore registers a newly trained encoder on actual November IR pixels,
with an identical extra-RGB encoder control. Both start from the preserved
Train-only RGB weights and execute36 epochs with matching source draws,
identity/triplet losses and paired frozen-RGB feature alignment. This is not
an old constructed-IR checkpoint. Only compact derived Train crops are cached
outside Git; full image datasets and detector inputs remain unchanged.
Native encoders and ordinary distillation are controls/primitives, not
validated contributions by themselves.

[Official detailed metric tables](perceptual_20261122_official_tables/index.json)
retain every per-sequence/combined row and threshold-specific column for388
completed runs (about6.4MB), including the nine earlier fixed ordinary Test
controls. Arrays, crop banks, model weights and raw track predictions are
excluded. The>=+5 Test HOTA objective and2-3 independently supported modules
remain unmet; supported multimodal innovation count is still zero.


Native representation and explicit-null query update (2026-11-23): all six
actual-pixel RGB/new-IR encoder fits completed36 epochs. All36 sampling digests
match within each RGB/IR seed pair; AMP overflow skip counts are separately
reported in [the budget audit](PERCEPTUAL_MFT_20261123_NATIVE_BUDGET_AUDIT.json).
The [old feature-scale diagnosis](PERCEPTUAL_MFT_20261123_FEATURE_SCALE_DIAGNOSIS.json)
shows why raw pre-neck cosine0.5 does not reject wrong identities. The repaired
post-neck feature space is an ordinary control, not a new fusion contribution.

Exact reconstruction of the original2,863,358 local association edges preserves
every query, label, lag and geometry field. Native features use the actual
perturbed proposal crops and their backgrounds, rather than substituting GT
crop descriptors. The native [query-fitting policy](../../configs/perceptual_native_query_v2_20261123.json)
forbids additional hidden local target deletion: deleted proposals must never
remain in point competition or frozen RGB inputs. Long appearance-only masks
recompute all affected query competition and RGB scores.

All12 frontend/cue/seed native query heads completed the minimum18 epochs.
The complete42-configuration Dev replay is running:36 matched6/12/18 snapshots
plus six exact-zero controls, each covering both Dev videos. Runtime keeps the
complete frozen fitted RGB scorer and real/null prior when the native increment
is zero. It can consult a retained identity only before a new identity's first
returned output. Current/future native observations cannot alter an emitted
prefix. All90 maintained tests pass.

All six earlier fitted-RGB residual heads also completed36 epochs. The complete
[epoch24](PERCEPTUAL_MFT_20261123_STRONG_RGB24_DEV.md),
[epoch30](PERCEPTUAL_MFT_20261123_STRONG_RGB30_DEV.md), and
[epoch36](PERCEPTUAL_MFT_20261123_STRONG_RGB36_DEV.md) tables retain all seeds.
[Guarded residuals](PERCEPTUAL_MFT_20261123_GUARDED_DEV.md) and
[missing-point fallbacks](PERCEPTUAL_MFT_20261123_MISSING_POINT_DEV.md) do not
establish a robust multimodal benefit. The [strong residual controls](PERCEPTUAL_MFT_20261123_STRONG_RGB_SELECTION.json)
are frozen only from all72 completed Dev snapshots, preserving every matched
frontend/cue/seed combination for formal Test.

The work cache occupies about16GiB on the independent work filesystem, below
the64GiB soft cap. No models, arrays, crops, dataset copies or raw predictions
are committed. Only physical GPUs2/3 are used. Native Test processing currently
extracts unlabeled current-frame representations; no Test result chooses this
new family. The original Test baseline is29.529 and the requested target is
at least34.529. Neither that target nor2-3 independent fusion contributions
has yet been established.


The initial native-query Dev family exposes harmful association overreach;
it does not establish a new-IR identity advantage. The complete
[Dev identity audit](PERCEPTUAL_MFT_20261123_NATIVE_DEV_IDENTITY_AUDIT.json)
separately shows native IR is weaker than the equally trained RGB encoder on
these held-out identities, especially at long intervals. Preserve this failure.
A [counterfactual query repair](../../configs/perceptual_counterfactual_query_dev_20261123)
keeps the strong RGB point match by default, requires>=0.05 posterior improvement
from the added cue, and protects clear RGB body correspondence. Its first three
[zero-cue parity controls](PERCEPTUAL_MFT_20261123_COUNTERFACTUAL_ZERO_PARITY.json)
match the full RGB reference exactly on all39 official fields and both complete
Dev track files. Its full matched Dev curve is still running.

The registered [post-neck metric repair](../../configs/perceptual_native_neck_metric_20261123.json)
trains metric and frozen paired-RGB distillation losses in the actual normalized
post-neck inference space. Both RGB and actual November IR receive an extra36
epochs after the earlier36, for72 cumulative encoder epochs, all Train-only.
These fits are running; no improvement is claimed. Only one resumable latest
and one compact exact-FP32 endpoint are retained for each repair arm. All95
maintained tests pass, including gradient-space and counterfactual parity tests.


Further complete Dev families retain the negative evidence:
[native real/null queries](PERCEPTUAL_MFT_20261123_NATIVE_QUERY_DEV.md),
[counterfactual corrections](PERCEPTUAL_MFT_20261123_COUNTERFACTUAL_QUERY_DEV.md),
[route decomposition](PERCEPTUAL_MFT_20261123_COUNTERFACTUAL_ROUTES_DEV.md),
[positive-only/short memory](PERCEPTUAL_MFT_20261123_POSITIVE_QUERY_DEV.md),
[ordered RGB/native memory](PERCEPTUAL_MFT_20261123_ORDERED_QUERY_DEV.md),
[mismatch calibration](PERCEPTUAL_MFT_20261123_ROBUST_QUERY_DEV.md),
[two-observation consensus](PERCEPTUAL_MFT_20261123_CONSENSUS_QUERY_DEV.md),
[clean-loss controls](PERCEPTUAL_MFT_20261123_CONSENSUS_CLEAN_DEV.md), and
[Train-fitted reachability](PERCEPTUAL_MFT_20261123_REACHABLE_QUERY_DEV.md).
A seed0 mismatch-calibrated result50.079 is insufficient: the other seeds often
inherit harmful generic RGB memory links. Preserve the stronger point-only
reference49.385/49.382/49.499 and the complete RGB memory controls.

The same registered12-epoch lookback rule forces both12-head clean and robust
cohorts through24 and30 epochs: the complete ordered RGB-cue seed0 result49.776
at18 exceeds its6/12 best49.083 by0.693pp. Every earlier snapshot and stage
history remains available; final stopping/selection still uses complete Dev.

The native post-neck repair completed an extra36 encoder epochs per cue/seed,
72 cumulative. Its [fixed-seed held-out diagnosis](PERCEPTUAL_MFT_20261123_NATIVE_NECK_S0_DEV_IDENTITY_AUDIT.json)
still does not support a reliable IR identity advantage. Prepared neck-query
loaders/fit plans are infrastructure only; those downstream candidate/head
fits have NOT been executed. An approximate reconditioning audit of old spatial
tokens is also diagnostic only: missing per-part raw norms prohibit calling it
exact post-neck inference or a tracking result.

All12 [ordinary pair controls](PERCEPTUAL_MFT_20261123_FIXED_PAIR_TEST.json) and
all12 [earlier fitted-RGB residual controls](PERCEPTUAL_MFT_20261123_FIXED_STRONG_RGB_TEST.json)
cover all7 Test sequences and15985 frames after Dev-only selection. For the
ordinary pair controls, mean Test HOTA is30.610 with the RGB detector/RGB
association,31.378 with the fixed new-IR paired detector/RGB association, and
31.710 with that same paired detector/IR point association. These are control
results, still below the requested34.529. They do not establish2-3 innovations.
No current foveated-method Test has started, and no Test per-sequence outcome
chooses its crop policy or thresholds. Historical Test exposure remains a limit.

The [foveated observation pilot](../../configs/perceptual_foveated_mft_20261123.json)
uses coarse actual new-IR structure to allocate RGB high-resolution crop
observations, allowing spatially approximate cross-modal guidance instead of
assuming reliable IR identity texture. Thirty-eight complete Dev controls are
running: RGB/new paired global streams, equal RGB/IR attention crop counts,
half/quarter crop scales, uniform tiling, and resolution-weighted refinement
versus concatenation/NMS. Generic zoom/weighted-box averaging alone receives
no multimodal novelty credit. The primary RGB-global/IR-attention method and
its RGB-attention control share exact detector weights and crop counts; the
paired-global variant has extra detector cost and is reported separately.
Both real-image source checks passed on physical GPUs2/3. Cold loading of the
old1.19GB optimizer checkpoint was canceled before any crop forward; the
existing198MB compact EMA is proven to be exactly the same inference half()
conversion by its archived joint3 source-selection metadata. No model changed.

All109 maintained tests pass; focused foveation tests also pass after the
source-only multiscale/prefetch additions. Work assets occupy about27GiB on the
independent work filesystem, below64GiB, while the original data filesystem
retains about5.8GiB free. Dataset/image copies and large Git artifacts remain
excluded. Goal status stays IN_PROGRESS / NOT_MET.

The complete30-epoch matched Dev tables are now retained for
[robust queries](PERCEPTUAL_MFT_20261123_ROBUST30_DEV.md),
[robust consensus](PERCEPTUAL_MFT_20261123_CONSENSUS30_DEV.md),
[clean queries](PERCEPTUAL_MFT_20261123_ORDERED30_DEV.md), and
[clean consensus](PERCEPTUAL_MFT_20261123_CONSENSUS_CLEAN30_DEV.md).
The common lookback rule forces the final36-epoch cap for all24 clean/robust
heads: RGB-global/IR-cue seed1 improves46.585 to46.962 at24 (+0.377pp),
which exceeds its complete best through18 by>0.2pp. This is not a Test gain
or evidence for a multimodal contribution. Strong point-only controls remain.

A current-image visual diagnosis shows bright generated tank walls and
reflections can dominate whole-image saliency. The original complete38-policy
foveation experiment continues unchanged. A separately registered
[working-region experiment](../../configs/perceptual_foveated_regions_20261123.json)
applies the same current RGB scene-region constraint to both cue selectors
and normalizes contrast inside that region. Another38 full Dev controls are
running, with no new foveated Test yet. Exact lookup preprocessing accelerates
CPU input preparation while preserving the original FP16 tensor bit for bit;
it does not change inference weights or precision. Scene masking, zoom and
ordinary box averaging alone receive no multimodal novelty credit.

All111 maintained tests pass. Current work assets occupy26.42GiB on the
independent work filesystem, below64GiB. No image copies, model weights,
feature arrays or raw tracking outputs are included in Git. The original
>=+5 Test HOTA and2-3 contribution objective remains NOT_MET.
