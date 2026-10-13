# Phase VII hypothesis ledger

Status: IN_PROGRESS. Preserve failed methods and discriminate alternatives; no fixed NO_GO count.

| ID | Question | First discriminating evidence | Status |
|---|---|---|---|
| VII-H0 | Can inherited results and guards be reproduced? | Old rank/cue measurements and baseline source/input/hash checks | REGISTERED |
| VII-H1 | Why does raw history rank better than compressed GRU while tracking is strong? | Same-pool normalized retrieval, native geometry/blend, confidence, actual history and calibration exposure | REGISTERED |
| VII-H2 | Do local visual tokens preserve identity beyond global features? | GT/detector crop retrieval, hard same-frame negatives, frozen encoder/probe and simple local controls | PROPOSED |
| VII-H3 | Does dense prediction add identity information beyond ordinary local temporal learning? | Matching local Transformer/contrastive/noEMA/no-prediction/no-ID controls | PROPOSED |
| VII-H4 | Does full-set competition resolve track/candidate conflicts and supervised NULL? | Same-input pair/full-set/MOTIP-adapted controls, random local ID remapping, bounded residual and full decoder routes | PROPOSED |
| VII-H5 | Do motion/occlusion cues add independent benefit? | Isolated Kalman/UKF/FishIoU/AGCIoU and observable occlusion controls | PROPOSED |
| VII-H6 | What dominates complete online time, and which changes preserve precision? | Component profiler and batch8/batch128 same-input parity | PROPOSED |

Startup: 10 A100-SXM4-40GB devices idle; no compute processes listed; 162.883GiB free. Initial work is CPU/source acquisition and offline diagnostics. Preserve old assets and reference commits; no large JEPA training before explaining VII-H1.

## VII-H1 first measured result

All six-video inherited pure ranks and same-supported-cue metrics were exactly recalculated. SN-015 FIFO62.3623% versus GRU35.8450% does not imply weak full tracker behavior: actual native blended row scores rank62.2969%, near geometry-only62.3249%. The geometry+confidence+GRU reconstruction matches recorded costs within1.2e-7; raw input norms are essentially1. This excludes normalization/checkpoint mismatch as an explanation for that fixed-state contrast, and demonstrates why pure-rank and actual score are different measurements.

PF-001 reverses the contrast: pure GRU96.5343% versus FIFO60.5190%. A uniform GRU-compression-failure conclusion is unsupported. Temporal training and current actual-state diagnostic use different historical tracker producers, known/unknown candidate supports and temporal cutoff sampling; SN-015 has calibration/backbone exposure. Registered next experiment stratifies all/some/no verified retained cues and common native appearance-region support, reporting changed denominators explicitly. Geometry dominance in fixed rows does not prove full Hungarian/endogenous trajectories remain unchanged without the GRU.

## Source audit execution boundary

The v2 follow-up completed all six videos. On SN-0152,716 all-verified-correct actual histories, FIFO99.5950% versus GRU52.6510%; history contamination alone is insufficient. Restricting retrieval to native appearance-region candidates yields7,081 positive-supported queries and FIFO94.1110% versus GRU84.5078%. This narrows but does not eliminate the discrepancy, with a different denominator explicitly disclosed. The smaller pool is a diagnostic condition, not a deployed eligibility change or claimed tracking improvement. Training/producer shift and scene-specific supervision remain live explanations.

The30-target registry is complete as an inventory, with15 source-ready entries at v2. Semantic audit remains incomplete for every entry; initial concrete model/runtime/mask/shape findings for MOTIP/DINOv3/SU-T/TIDE are written as partial audits. Intermittent TLS and incomplete checkout errors are preserved. v1 aborted on an old external symlink guard; v2 safely inventoried all30; v3 repairs only our partial clones after materialized-file hash checks. No old reference is pulled or reset and no external project code or model weights are executed/downloaded.
