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
eight-frame extraction smoke checks. Six new mechanism tests cover known
translation, invalid flow, polarity-independent contrast, identity continuity,
future-evidence prefix invariance and supported single-gap output. The current
full repository suite passes 25 tests, including all six mechanism tests.
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
