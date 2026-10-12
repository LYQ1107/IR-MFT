# Phase VI initial whole-video risk controls

Status: WARMSTART_DIAGNOSTICS_COMPLETE; Phase VI IN_PROGRESS. 12 actual eight-epoch seed0 fits: six outer validation videos, four fitting videos and one distinct calibration video in each fold. 64 steps/epoch, batch128, video-uniform sampling, identical query set/seed/optimization budget. Context model 1,825 parameters, ordinary same-input MLP 1,793 (difference 1.8%). Frozen original GRU teacher states are NOT genuinely OOF for its fitting videos; detector/FishReID training exposure remains. These results isolate new risk fitting only.

| Outer video | Model | Accepted-edge risk AUC | Calibrated ECE | Local N01 | Local N10 | Net |
|---|---|---:|---:|---:|---:|---:|
| BT-001 | ordinary MLP | 0.661016 | 0.343016 | 28 | 44 | -16 |
| BT-001 | context risk | 0.655954 | 0.347584 | 26 | 47 | -21 |
| BT-003 | ordinary MLP | 0.926922 | 0.232816 | 17 | 24 | -7 |
| BT-003 | context risk | 0.929445 | 0.247419 | 17 | 26 | -9 |
| MSK-002 | ordinary MLP | 0.659261 | 0.219820 | 3 | 36 | -33 |
| MSK-002 | context risk | 0.638621 | 0.355212 | 1 | 44 | -43 |
| PF-001 | ordinary MLP | 0.988523 | 0.061905 | 630 | 0 | 630 |
| PF-001 | context risk | 0.989997 | 0.100533 | 585 | 0 | 585 |
| SN-001 | ordinary MLP | 0.931793 | 0.267466 | 142 | 3 | 139 |
| SN-001 | context risk | 0.928674 | 0.298070 | 164 | 4 | 160 |
| SN-015 | ordinary MLP | 0.597887 | 0.127519 | 211 | 2195 | -1984 |
| SN-015 | context risk | 0.557306 | 0.138529 | 201 | 2303 | -2102 |

N01/N10 compare local known-positive candidate rankings to original known accepted choices on identical fixed histories, not changed Hungarian assignments, sustained online recovery events or MOT metrics. Unknown labels and unknown newly ranked candidates are excluded; absent-positive rows do not contribute rank metrics. All unknown exclusion/coverage, calibration bins, Brier, risk-coverage curves, fixed accept/reject counts and uncalibrated controls are retained in the measured JSON.

Observed failure: context pooling does not consistently improve the same-input MLP; SN-015 loses many previously correct local choices, while PF-001 and SN-001 have positive static signals. Calibration learned on another entire video often transfers poorly (ECE substantial). Root-cause alternatives: current scalar cues lack reliable identity evidence, historical teacher-state shift dominates, score ranking may reward suppressing motion, or calibration changes acceptance under different errors. No overall MOT benefit or innovation is claimed.

Discriminating next experiments: run actual complete-video risk gates/scoring against same-input MLP, fixed confidence/margin/entropy and simple protected writes; regenerate video-excluded teacher states before genuine OOF claims; inspect score/assignment changes and video-conditioned shift without per-video tuned thresholds. Independently evaluate K1/K2 states and matched candidate-attention/EMA objectives. Consider raw token/candidate relationships if scalar risk is insufficient.

Engineering repair: initial v1 was stopped before the first epoch because each per-row NPZ member lookup re-decompressed whole arrays. Loader now materializes arrays once; 40,886 queries loaded in measured 0.781 s. Preserved v1 logs/source/directories and immutable loadrepair_v2 retain the same scientific configs. Exact configured epoch3 stop/resume to epoch8 now matches continuous student, optimizer, RNG, loss/sample order, calibration and inference payloads; see PHASE_VI_RISK_RESUME_PROOF.json. This proves epoch-boundary restoration, not arbitrary minibatch SIGTERM recovery.
