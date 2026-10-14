# Phase VII independent motion and occlusion experiments

Status: IN_PROGRESS. Existing motion is unchanged in all new C probes; no motion improvement is claimed.

SU-T exact479be89 now has complete source fields: its default custom19-sigma-point UKF applies linear9-D F/5-D H without fx/hx callbacks; the optional FilterPy branch has mismatched7/4 versus9/5 dimensions. FishIoU central_area1 uses overwritten intersection coordinates, and area/1000 is resolution-dependent. tools/su_tracker.py requests an absent evaluator export and evaluate_su_t constructs an unbound lowercase tracker alias. These are static source findings, not observed runtime failures or measured motion effects.

Kalman/UKF filtering, FishIoU/AGCIoU association and observable occlusion status must be isolated while retaining V3 proposals, feature coordinate, memory, native lifecycle and output policy. Importing a whole external tracker would mix those factors. Geometry/input equality, numerical finite behavior, equal zero-measurement handling and causal prediction-before-write ordering require checks. Gap/overlap/density remain proxies; no ground-truth occlusion or turning labels are invented.

Nonlinear motion learning, occlusion-conditioned dynamics and combinations with local/JEPA/joint evidence remain pending and will be selected by actual fitting/calibration evidence. Full native trajectories and complete online cost must validate any retained motion contribution independently of identity features.

TIDE exactfba27ae default track.py calls JDETracker, using XYAH Kalman, appearance/motion and ordinary IoU. Standalone AGCv1 is singular at zero center distance, AGCv2 has no return, and neither is invoked by the active matching helpers. STrack first writes the inherited shared BaseTrack.features list, then replaces it with an empty instance deque; initial per-track history is lost and the class list retains the initial features. This is not an absent-attribute failure. Paper AGCIoU, source-faithful inactive definitions, disclosed repairs and independent geometric controls must be distinguished. Reference code was not executed. See PHASE_VII_30_PAPERS_CODE_AUDIT.md and exact AST/source proofs.

## Additional eight source audit constraints

M4FT spatial eight-direction selective scan is current-image detector processing, not temporal fish motion; its README default tracker is BYTETracker, with SORT on a separate entry. PFTrack uses predicted center offsets and geometric assignment. DM3T adds six confidence association stages/velocity smoothing and random noise even in eval, so detection, lifecycle and stochasticity controls are required. SAMURAI uses a model-global SOT XYAH Kalman state and mask-quality gates; same-input MOT adaptation needs per-track state/reset and detector-origin prompts. None establishes a trained nonlinear fish-motion gain.

## Additional three source audit constraints

DiffMOT exacteada72e supplies a genuine five-record8-D history-conditioned nonlinear4-D next-box-delta predictor: six-layer history Transformer and one-step decoupled diffusion with weighted drift/noise SmoothL1. Training records need actual frame-spacing checks; default inference samples random initial noise. The public evaluation reads cached detections/ReID, has an uncached ReID-name mismatch and does not include full raw encoding. A motion-only port must preserve native inputs/lifecycle and compare matched deterministic regression/KF/UKF before combinations; no training/gain claimed from this audit.

## Final three source audit constraints

TrackTrack uses linear8DcxcywhKF with detector-score-scaled measurement covariance and corner-angle/confidence terms; default offline AFLink/gradientboosting interpolation is excluded. COVTrack motion cue in the active fusion is raw4Dbox position, not nonlinear dynamics. Source-name/paper language alone does not establish learned swimming/occlusion; motion must be isolated against same-input KF/UKF/geometry/lifecycle controls.

## Independent native linear UKF control: all six complete videos

The owned NativeUnscentedFilter uses17sigma points (alpha1,beta2,kappa0) with exactly the original8D XYAH constant-velocity F/H, same initialization and height-dependentQ/R. No jitter/eigenvalue clipping, changed observed state, noise tuning or nonlinear fish-motion claim. Gaussian transform tests and160prediction/update steps with observation gaps match the native linear KF numerically. Track activation/update/reactivation and lost-state velocity reset remain native; only the process-local filter implementation changes. It is an8D control, not officialSU-T9Dfulltracker reproduction.

All27251frames across6videos complete. Fresh KF andUKF predictions are exactbyte-equal to each other and each original GRU native reference, all6; pinned official metrics remain unchanged. Complete currentproposals/512RGBcoordinates/history/nativegraph/lifecycle/output remain fixed. Single-run CPUassociationtime (sharedmachine, notisolatedruntimebenchmark):

| Video | Frames | KF (s) | UKF (s) | Exact output bytes |
|---|---:|---:|---:|---|
| SN-001 | 684 | 6.70 | 10.35 | Yes |
| SN-015 | 6570 | 66.24 | 84.60 | Yes |
| PF-001 | 15000 | 61.52 | 73.66 | Yes |
| BT-001 | 3000 | 30.89 | 39.86 | Yes |
| BT-003 | 243 | 2.43 | 3.12 | Yes |
| MSK-002 | 1754 | 89.41 | 109.38 | Yes |

UKF alone with the same linear motion/observation model provides no nativeMOT improvement here. This isolates filtering from changes in state, nonlinear dynamics, appearance or lifecycle. It doesnot exclude a genuinely different nonlinear fish-motion model after error-source evidence. FishIoU/AGCIoU/source-faithful versus repaired definitions, observable overlap/density/conflict states and learned nonlinearity remain required. Neither this cacheassociationtime nor official metric evaluation is complete rawFPS. Full proof and parameter/runtime/source/input/outputSHA references are in PHASE_VII_LINEAR_UNSCENTED_MOTION_MEASURED.json.

## Completed isolated geometry controls: 96 complete videos

All16configurations×6videos (436016nativeframes) complete, original V3/512RGB/GRUseed0/nativeKF/proposal rows and scores/thresholds/lowstage/duplicates/lifecycle/measurementoutput fixed. Geometry-only weight0 and originalGRUweight0.3 are separate. Identitycost and its original inclusiveIoU eligibility remain fixed; only high/unconfirmed association geometrychanges. These deterministic source0controls are not threeindependent newencoderseeds.

Native IoU includes+1pixel; source FishIoU/AGCv1 usecontinuous coordinates. ContinuousIoU and ordinaryDIoU therefore distinguish convention/center-distance changes from sourceheuristics. OriginalFishcentral-area defect is preserved in fish_source; centralrepair alone isdisclosed and still retains the resolution-dependent area/1000factor. Fixed1.3IoU+.3 andIoU+.05controls isolate scorecompression/offset effects. AGCv1 is inactive/singular inTIDE source; our finiteclippedlimit is explicit and is not paperAGCIoU/fulltracker reproduction. MIT notices for both source projects are retained. Actual2000pair purefunction parity errors are0(Fish) and6.94e-18(finiteclippedAGC); zerocenter sourceexception isverified.

| GRU geometry | SN001 HOTA / IDF1 / SW | SN015 HOTA / IDF1 / SW | MSK002 HOTA / IDF1 / SW | PF001 HOTA / IDF1 / SW |
|---|---:|---:|---:|---:|
| native | 46.238 / 65.251 / 26 | 31.983 / 29.636 / 151 | 24.393 / 29.359 / 389 | 45.385 / 45.292 / 18 |
| diou | 42.571 / 60.151 / 24 | 32.505 / 29.569 / 156 | 24.232 / 28.654 / 388 | 45.888 / 45.294 / 18 |
| fish_central_repaired | 40.198 / 53.924 / 32 | 29.357 / 26.015 / 184 | 18.382 / 22.286 / 667 | 33.513 / 27.748 / 28 |
| fish_affine_control | 41.204 / 57.024 / 36 | 27.999 / 26.122 / 216 | 11.304 / 11.684 / 6152 | 34.210 / 31.796 / 46 |
| continuous_iou | 46.238 / 65.251 / 26 | 31.539 / 28.406 / 154 | 24.346 / 29.360 / 385 | 45.385 / 45.292 / 18 |
| fish_source | 40.198 / 53.928 / 32 | 29.627 / 25.982 / 179 | 18.175 / 22.364 / 662 | 33.376 / 26.753 / 35 |
| agc_clipped | 46.238 / 65.251 / 26 | 31.539 / 28.406 / 154 | 24.178 / 28.867 / 387 | 45.385 / 45.292 / 18 |
| iou_offset005 | 41.325 / 57.226 / 43 | 28.646 / 29.898 / 158 | 11.214 / 12.432 / 5895 | 45.385 / 45.292 / 18 |

Source/repairedFishIoU does not establish a stable benefit: large score offsets and sourcecentraldefect can change many nativefeasible edges and worsen continuity. DIoU hasheterogeneous effects, e.gPF001HOTA45.385→45.888 butSN00146.238→42.571, withnoIDF1improvement onPF. ClippedAGC doesnot consistently beat the continuousIoU control. OriginalIoU convention alone changesSN015GRUHOTA31.983→31.539; do not attribute that to a new fish method. All6/geometry-only/fullmetrics, native-threshold crossing counts, exactinputs/output/source/protocol/officialSHA evidence are in PHASE_VII_FISH_GEOMETRY_MEASURED.json. No per-video threshold selection, sourcefulltracker/paperAGCIoU/learnednonlinearity/trueocclusion/novelty/rawFPS claim. Paper-faithful AGCIoU and observableocclusion/nonlinearerror diagnosis remain required.

## Paper/source AGCIoU distinction confirmed

The [TIDE v1 paper](https://arxiv.org/html/2607.26412v1#S3.SS3) specifies normalized centroid distance, aspect/area alignment and an IoU<=0.7 activation with fixed constants. Pinned AGCv1 uses raw distance, lacks structural terms/activation, and is inactive in source ious; AGCv2 has no return. Earlier clipped AGCv1 metrics remain a source-function control. Average scale/epsilon and scores above1 need explicit reproduction choices or supplementary evidence before a paper-faithful claim. This discrepancy changes interpretation, not the recorded native outputs.


### Completed independent observable-overlap and native-motion diagnosis

All 42 full-video controls are complete: seven fixed configurations across six videos, each covering all 27,251 native frames. V3 proposals, RGB features, the legacy GRU seed 0, KF, history writes, lifecycle, and measured output policy are fixed. Disabled controls reproduce the original prediction bytes for every video. The intervention changes only the high-stage appearance weight: candidate quality is one minus its maximum continuous IoU with another current high-confidence detection. Constant-half, eligible-row mean, and three candidate-permutation controls distinguish weight reduction from candidate-specific information. Row means and gate histograms match on a fixed decision input; trajectories can subsequently diverge.

| Control | Macro HOTA | Macro IDF1 |
|---|---:|---:|
| Original | 39.692 | 48.132 |
| Constant half | 38.549 | 46.448 |
| Eligible-row mean | 39.250 | 47.391 |
| Candidate overlap | 39.721 | 47.982 |
| Permutation seed 0 | 39.389 | 47.641 |
| Permutation seed 1 | 39.697 | 47.874 |
| Permutation seed 2 | 39.670 | 47.926 |

Overlap weighting adds 0.029 HOTA points while losing 0.149 IDF1 points. BT-001 loses HOTA and IDF1 relative to the original. SN-001 has identical reported scores for row-mean, overlap, and all permutation controls. These results do not establish a stable independent contribution. Permutation seeds are placebo assignments, not three separately trained methods. All official per-video metrics, alpha arrays, source/input/GT/prediction hashes and threshold crossings are linked in PHASE_VII_OBSERVABLE_OCCLUSION_ALL42_MEASURED.json.

A separate observer records native predicted motion states and reproduces all six prediction files before loading physical labels. Retrospective analysis requires at least three entirely known, consistent recent accepted proposal identities; unknown, mixed, short, missing-reference, and absent/nonunique current GT cases are excluded. Current GT is used only to evaluate center error, normalized by its box diagonal.

| Video | Supported rows | Mean KF error | Mean two-measurement CV error |
|---|---:|---:|---:|
| SN-001 | 4,070 | 0.0552 | 0.0581 |
| SN-015 | 52,658 | 0.0537 | 0.0590 |
| PF-001 | 27,636 | 0.0352 | 0.0362 |
| BT-001 | 15,618 | 0.0653 | 0.0671 |
| BT-003 | 1,249 | 0.0685 | 0.0700 |
| MSK-002 | 37,782 | 0.0619 | 0.0644 |

The four fitting videos contribute 70,737 supported rows, with pooled KF/CV errors 0.05122/0.05313. KF and two-point CV have different history capacity, and the KF state can retain earlier pollution. The latest KF measurement and identity-write timestamps are checked separately; they agree in these supported samples. This comparison does not justify a broad learned-motion replacement or explain failures in excluded histories.

Long accepted-measurement gaps have larger errors: MSK-002 has 302 rows with KF/CV means 0.715/0.720, SN-015 has 322 rows with 0.480/0.962, and inner BT-001 has 45 rows with 0.288/0.274. These are observed gaps, not true occlusion or turn labels. Among known true candidates in this support, exclusion by the original appearance graph occurs in 1/3,912, 79/51,670, 0/27,506, 2/14,908, 0/1,191, and 34/33,536 rows for the six videos in the table order. Missing detections and unclassified histories remain separate limitations. Full support, exclusions, and conditional error distributions are in PHASE_VII_NATIVE_MOTION_ERROR_MEASURED.json.

Primary-source check of [TIDE v1, section3.3](https://arxiv.org/html/2607.26412v1#S3.SS3) reconfirms the normalized-centroid/aspect/area formula with its fixedactivationconstants. The manuscript names averageboxscale withouta uniqueexplicitdefinition or epsilonvalue; itsreference tosupplementaryderivations doesnotresolve thosechoices inthatHTMLversion. ExistingAGCv1sourcecontrols remainseparate; fullpaperAGCIoU reproduction andcompleteTIDE association pipeline arestillunverified.

Measured PNG/SVG/PDF figures are retained outside Git at /data3/liuyeqiang/IR-MFT_assets/cfp/phase7/observable_motion_figures_v1. The figure compares conditional center errors and macro metric changes; it does not infer physical occlusion, nonlinear causes, or statistical significance.
