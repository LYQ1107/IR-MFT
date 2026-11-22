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
