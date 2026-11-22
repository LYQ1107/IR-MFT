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
