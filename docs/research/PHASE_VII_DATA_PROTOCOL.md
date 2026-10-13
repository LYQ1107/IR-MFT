# Phase VII data protocol

Status: IN_PROGRESS. Frozen base: configs/cfp/protocol_v2.json.

Train: BT-001, BT-003, MSK-002, PF-001, SN-001, SN-015. Outer leave-one-video-out with distinct inner calibration. New encoder fitting excludes both; track-producer/module fitting matches the same exclusions and coordinate space. Seed-matched baselines are evaluated on identical outer videos. New training features require new guarded, immutable banks and compatible ordinary predictors; old GRU weights cannot be silently moved into new feature coordinates.

BT-005/SN-013 remain exposed development references, with detector exposure disclosed. The seven historical test videos are excluded from model/threshold search. Existing six-video FishReID features have encoder exposure and cannot demonstrate new-encoder unseen-video generalization.

Identity key: (video_id, physical_track_id). Same-frame different known fish are the primary hard negatives; unknown proposal identity stays unknown. Duplicate/ambiguous annotation handling and derived GT hashes follow protocol_v2. Labels are offline supervision/diagnosis only. Actual accepted histories remain uncorrected.

Two crop protocols are required: supervised GT crops and identical current V3 detector crops. GT crops are training/diagnostic inputs only. Current output boxes must remain current measured detector boxes. Clean/actual/corrupted histories use identical current candidate pools and report denominators, missing candidates, unknowns and available past support.

Natural occlusion/turn/visibility labels require verified annotation support; box overlap, gap and crowding may be reported explicitly as proxies. No semantic anatomical token names without labels. External MFT25/MFT-Edge/BrackishMOT-M4FT/FishMAE data require permission, identity/cadence/split, video-overlap and modality audits before acquisition.

Protocol A/B/C and full metric list follow PHASE_VII_TASK_BOOK.md. Synthetic causal IR is derived from RGB and is not an independent infrared sensor.
