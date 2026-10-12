# Phase V baseline audit

**BASELINE_REPRODUCED**. Existing detector/feature/weight/GT hashes verified; seven preserved runs independently reevaluated with pinned official TrackEval. All metric arrays and metric-source hashes exactly match prior records. No SGD, detector/cache changes or historical test evaluation.

| System | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| ByteTrack | 49.297 | 34.012 | 71.509 | 58.877 | 92.529 | 86.000 |
| BoT-SORT | 52.482 | 37.498 | 73.518 | 57.477 | 93.552 | 113.667 |
| RGB-GRU | 51.885 | 37.716 | 71.433 | 62.298 | 92.562 | 89.000 |

Available storage at start: 171.6GiB; minimum reserve40GiB; maximum4 physical GPUs.100 original tests pass. Both pinned detector cu117/cuDNN8.5 and encoder cu118/cuDNN8.7 stacks verified. Detailed system snapshot, every video/seed metric, actual commands and input hashes: PHASE_V_BASELINE_AUDIT.json.

Baseline worktree began clean at4f1b226, live remote confirmed identical; new branch research/identity-repair-jepa-v1 created without overwriting history. Current prototype source commit: 4f1b226343335d219af015d6c2954c0c81d28c7a

The two dev videos isolate new identity modules only; historical V3 trained on all8. Biological identity overlap unverified; dense MSK excluded dev. Previous online numeric boundary remains: canonical4834frame detector bank ends withbatch2, while online uses8current copies; last2frames differ in coordinates/confidence, principal metrics/identity decisions equal. See RESEARCH_OUTCOME_20261012.json. Full per-sequence metrics remain in the machine audit; three-seed variation is not a cross-scene confidence interval.

Test invocation: `MFT26IR_ROOT=/data3/liuyeqiang/datasets/mft26-ir-causal-v1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /data3/liuyeqiang/.venvs/cfp-fishmot/bin/python -m pytest -q tests`. Initial CFP-only glob collected99tests; a full-suite invocation without MFT26IR_ROOT failed the legacy pairing test at its stale default path. Both failed audit attempts/logs are retained; explicit dataset env restores all100tests without changing data, source or global environments.
