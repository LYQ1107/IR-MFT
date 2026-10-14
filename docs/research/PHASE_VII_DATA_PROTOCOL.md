# Phase VII data protocol

Status: IN_PROGRESS. Frozen base: configs/cfp/protocol_v2.json.

Train: BT-001, BT-003, MSK-002, PF-001, SN-001, SN-015. Outer leave-one-video-out with distinct inner calibration. New encoder fitting excludes both; track-producer/module fitting matches the same exclusions and coordinate space. Seed-matched baselines are evaluated on identical outer videos. New training features require new guarded, immutable banks and compatible ordinary predictors; old GRU weights cannot be silently moved into new feature coordinates.

BT-005/SN-013 remain exposed development references, with detector exposure disclosed. The seven historical test videos are excluded from model/threshold search. Existing six-video FishReID features have encoder exposure and cannot demonstrate new-encoder unseen-video generalization.

Identity key: (video_id, physical_track_id). Same-frame different known fish are the primary hard negatives; unknown proposal identity stays unknown. Duplicate/ambiguous annotation handling and derived GT hashes follow protocol_v2. Labels are offline supervision/diagnosis only. Actual accepted histories remain uncorrected.

Two crop protocols are required: supervised GT crops and identical current V3 detector crops. GT crops are training/diagnostic inputs only. Current output boxes must remain current measured detector boxes. Clean/actual/corrupted histories use identical current candidate pools and report denominators, missing candidates, unknowns and available past support.

Natural occlusion/turn/visibility labels require verified annotation support; box overlap, gap and crowding may be reported explicitly as proxies. No semantic anatomical token names without labels. External MFT25/MFT-Edge/BrackishMOT-M4FT/FishMAE data require permission, identity/cadence/split, video-overlap and modality audits before acquisition.

Protocol A/B/C and full metric list follow PHASE_VII_TASK_BOOK.md. Synthetic causal IR is derived from RGB and is not an independent infrared sensor.

The new two-crop metadata index is complete for all27,251six-training-video frames:217,201valid nonambiguous GT observations,258,309actual frozen proposals and98composite physical identities. Unknown detector identities are retained; per-video known GT/proposal coverage and source hashes are recorded in PHASE_VII_TWO_CROP_INDEX_MEASURED.json. No duplicate raw image or dense crop cache is created. This is index completion, not visual feature evaluation or model fitting.

The first C probe fits4videos(BT001/BT003/MSK002/PF001), reports innerSN001 and conditional outerSN015after fixed last epoch8. Its encoder and historical GRU have prior exposure to those videos; the new-head holdout cannot be represented as encoder-independent unknown-scene generalization. The exact full-video evaluation subset preserves original GT hashes and declared memberships, and combines inner/outer roles only for descriptive aggregates. Three-seed/all-fold and new outer/inner-excluded visual training remain pending.

## Executed visual fitting split and paired crop support

The all-frame index feeds a frozen every20frame residues1/2 pilot with47,751GT/detector observations. New adapters and ImageNet-initialized backbone fits exclude both outerSN015 and innerBT001; fitting videos are BT003/MSK002/PF001/SN001. Composite(video,fish)identities and real crowded-frame anchors prevent cross-video label collision. Known positives require>=25frame distance; unsupported short histories are excluded from temporal-positive loss rather than fabricated. Unknown detector labels stay in retrieval galleries but never become labeled negatives. GT pairing is offline only. The9.38GB normalized float16 crop cache preserves exact feature row order and all validity bits; no crops or features are Git-tracked. All-fold/three-seed and external-data identity/license/overlap audits remain pending.

## Initial external fish-data primary metadata audit

[BrackishMOT creator metadata](https://www.kaggle.com/api/v1/datasets/view/maltepedersen/brackishmot) version4 declares CC BY-SA4.0 and2,234,753,716bytes, while a secondary index names different terms; the primary response is locally hash-preserved. [The project](https://vap.aau.dk/brackishmot/) describes98real clips and additional synthetic training data. Actual split/cadence/identity continuity and overlap with MFT25/MFT26/BT-sequences remain unverified; it is not yet independent external data.

[FishTrack23's primary paper](https://openaccess.thecvf.com/content/WACV2024/papers/Dawkins_FishTrack23_An_Ensemble_Underwater_Dataset_for_Multi-Object_Tracking_WACV_2024_paper.pdf) describes optical provider subsets annotated at5/10Hz despite higher native rates. Per-provider permissions, exact clip/track IDs and overlap need inspection; source web-open403 is retained. [Fish4Knowledge trajectory release](https://groups.inf.ed.ac.uk/vision/DATASETS/FISH4KNOWLEDGE/WEBSITE/GROUNDTRUTH/BEHAVIOR/) provides5fps trajectory metadata and explicitly no raw images, so it cannot directly supply cropReID. Reuse terms and multi-trajectory video linkage are not verified.

[The institutional3D zebrafish record](https://tore.tuhh.de/entities/product/01d8baa1-b6a5-4163-924a-01f40ff3bc81) declares stereoRGB, CC BY-NC4.0 and67.89GBzip; raw payload acquisition is deferred while preserving running-fit disk reserves. Split by recording/physical fish across cameras, exact annotation/native cadence and video overlap must be verified. No external frames/models have been acquired or trained on in this audit. PHASE_VII_EXTERNAL_DATA_AUDIT_MEASURED.json distinguishes exact verified fields, transport limitations and required pre-use work. This is preliminary metadata auditing, not completed external validation.

LocalJEPA v1 reuses exact frozen DINO local/CLS/fraction arrays, native-frame counts and original20frame residues1/2; empty sampled scenes are explicitly restored from index provenance. All9controls share exact per-epoch sampled-input hashes. Training/currentGT targets and classification images are separate from strictly past predictor input. Unknown candidates are not labeled negatives; `target_absent` means no verifiedpositive in the current gallery, not proven physicalabsence or NULL. Offline physicalGT/verified-detector histories are diagnostic references, not deployed corrected histories. Student forward contains no physicalIDs/GTtargets/teacher. Masked target supervision withholds the entire currentframe; cached past tokens remain globally mixed within their own crop, so absent-part/pixelmask isolation or anatomy/occlusion labels are not claimed.

## Priority external data audit

Creator primary metadata for MFT25 is acquired and hash pinned: HF repository2950cefee5d8dd199693f166105dbc8bc0257567 declares Apache2, auto gating, train/test zip names and1,830,282,058storage bytes. The gate asks for contact-information sharing; no account acceptance or image payload occurred. Exact TIDE source lists ten MFT-Edge sequences and seven MFT25 test sequences. Shared BT/MSK/PF/SN family names require content/lineage verification before calling them independent external data. MFT-Edge data permission/cadence/IDs and FishMAE laboratory data access/license remain unresolved; code license does not supply image permission. Snapshot TLS failures are retained. Full primary metadata/source report is PHASE_VII_PRIORITY_EXTERNAL_DATA_MEASURED.json; no external generalization claim follows.

Primary sources: [MFT25 creator metadata](https://huggingface.co/datasets/Vranlee/MFT25), [MFT25 project](https://vranlee.github.io/SU-T/), [MFT-Edge project](https://vranlee.github.io/TIDE/), [FishMAE source](https://github.com/Wynne-cc/FishMAE).
