# EvidenceFishMOT — Phase VII final research goal

Status: IN_PROGRESS. User task book: PHASE_VII_TASK_BOOK.md, all 1,723 lines.

Learning Fine-Grained Identity Evidence and Joint Association for Online Multi-Object Fish Tracking. The objective is stable, reproducible identity evidence and multi-track association improvement with reasonable complete online cost. A/B/C/D are hypotheses, not preselected successful contributions. A scoped scientific conclusion must be evidence based; initial failures or a fixed experiment count do not terminate the goal.

Start from verified Phase VI 042608779d936bcd442d8c266734a6d4d47636c1, on research/evidence-fishmot-v1. Preserve Phase I–VI assets, strong ordinary RGB GRU/current-measurement and BoT-SORT references. The seven historical test videos are excluded from research selection.

Required work: source-level audit of all 30 exact-commit reference projects; explain the raw-history/GRU ranking discrepancy before large JEPA training; build common GT-crop and actual-detector-crop identity diagnostics; study frozen/appropriately adapted visual features and local tokens; implement joint identity competition with supervised NULL/newborn handling; train and independently test object-centric Dense JEPA and matched ordinary local temporal controls; test motion/occlusion separately; combine mechanisms only to answer evidence-supported questions.

New identity encoders and association modules must exclude the outer evaluation video and inner calibration video from fitting. Old frozen encoders that saw outer videos yield conditional association evidence only. Every comparison needs matching seed/data/proposals/output/evaluator rules. Physical identity is (video_id, physical_track_id); unknown labels cannot become wrong/NULL supervision.

Protocol A: identity retrieval and hard same-frame negatives, clean/actual/corrupted history, GT/detector crop differences and time/occlusion/density breakdowns. Protocol B: complete frozen-V3 current-measurement MOT with all official metric vectors and per-video results. Protocol C: complete raw-frame strictly online pipeline with component profiler, actual FPS/latency/memory/parameters/countable operations, independent module overhead and numerical-output parity.

Ordinary controls must separate local evidence from extra parameters, JEPA prediction from ordinary Transformer/contrastive learning, EMA from noEMA, and set competition from independent pair scorers. Preserve geometry in the bounded joint-residual route; test full learned ID decoding separately. No teacher, GT label or future frame may reach online inference.

Maintain the experiment registry, hypothesis ledger and all task-book documents. Real failures/interruption and next discriminating experiments are recorded. Keep >=40 GiB free, use <=4 currently idle physical GPUs, do not affect others, save resumable training states including optimizer/scheduler/all RNG/data/source/config guards. Reference source lives outside IR-MFT; inspect licenses before copying or using weights. Do not download all models/data, bypass permissions, or upload data/weights/features/caches to GitHub. Push small code/config/results at completed stages and independently verify the remote ref.

Final audit covers all 20 final scientific questions and each explicit implementation, control, test, artifact and command in the original task book. Pending experiments remain IN_PROGRESS; no invented positive claims or early project-wide NO_GO.
