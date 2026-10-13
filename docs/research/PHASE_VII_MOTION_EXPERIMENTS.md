# Phase VII independent motion and occlusion experiments

Status: IN_PROGRESS. Existing motion is unchanged in all new C probes; no motion improvement is claimed.

Partial SU-T source inspection finds real sigma-point UKF machinery, but its default unsetfx propagates by a constant-velocity F matrix. A UKF name does not establish learned nonlinear swimming dynamics. The active FishIoU/default entry and alternative FilterPy state dimensions still need tracing. Partial TIDE matching source includes inverse-distance AGC variants; exact active variant and zero-distance behavior need verification before a faithful metric control.

Kalman/UKF filtering, FishIoU/AGCIoU association and observable occlusion status must be isolated while retaining V3 proposals, feature coordinate, memory, native lifecycle and output policy. Importing a whole external tracker would mix those factors. Geometry/input equality, numerical finite behavior, equal zero-measurement handling and causal prediction-before-write ordering require checks. Gap/overlap/density remain proxies; no ground-truth occlusion or turning labels are invented.

Nonlinear motion learning, occlusion-conditioned dynamics and combinations with local/JEPA/joint evidence remain pending and will be selected by actual fitting/calibration evidence. Full native trajectories and complete online cost must validate any retained motion contribution independently of identity features.
