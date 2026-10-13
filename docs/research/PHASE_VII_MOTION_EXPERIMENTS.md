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
