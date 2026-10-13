# Phase VII30-reference source audit

Status: IN_PROGRESS. Source readiness and exact Git commit inventory do not constitute a completed semantic audit. New sparse snapshots omit weights/media and live outside IR-MFT. Failed checkout/network attempts are preserved. No external project code has been executed.

| Project | Exact available commit | Source acquisition | Semantic audit |
|---|---|---|---|
| [FishMAE](https://github.com/Wynne-cc/FishMAE) | cffcdeaf61c8c3a9768711a4615ff4af3899ca30 | ACQUISITION_FAILED | NOT COMPLETE |
| [Fish_Re_Identification](https://github.com/msamdk/Fish_Re_Identification) | 1976bb16ef7af49412b5fd690b4cbb526a91a762 | ACQUISITION_FAILED | NOT COMPLETE |
| [SU-T](https://github.com/vranlee/SU-T) | 479be89a27129f6879c62710a83abac6a8b261ee | SOURCE_READY | NOT COMPLETE |
| [TIDE](https://github.com/vranlee/TIDE) | fba27aefd1da11bfc7f946240dd66d85ea02a39c | SOURCE_READY | NOT COMPLETE |
| [M4FT](https://github.com/vranlee/M-4-FT) | b120e568c07145891e043675892890ed9fe9279d | SOURCE_READY | NOT COMPLETE |
| [DM3T](https://github.com/vranlee/DM-3-T) | 662790ba213b4460f19a1bdbefb9714416eb2b5c | SOURCE_READY | NOT COMPLETE |
| [TFMFT](https://github.com/vranlee/TFMFT) | d4fbcc0d4df685cc01f5f5d7c9510601a1c9e5b2 | SOURCE_READY | NOT COMPLETE |
| [PFTrack](https://github.com/wqw123wqw/PFTrack) | 73d4dc4c5e69e530f7a77d974bff7d2040720728 | SOURCE_READY | NOT COMPLETE |
| [FDTA](https://github.com/Spongebobbbbbbbb/FDTA) | b3b3b778acf93fa4269663b5ea1fd1d5ff8c6730 | ACQUISITION_FAILED | NOT COMPLETE |
| [TransReID](https://github.com/damo-cv/TransReID) | dec55046fcdfadee14e2c28e2df89305d8f7557a | ACQUISITION_FAILED | NOT COMPLETE |
| [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID) | eb1898b72c882875f478bebfc6d41644eece0a5d | ACQUISITION_FAILED | NOT COMPLETE |
| [DINOv3](https://github.com/facebookresearch/dinov3) | 6876159a11b4df116f30f667f8c9888617df0751 | SOURCE_READY | NOT COMPLETE |
| [DINOv2](https://github.com/facebookresearch/dinov2) | 7764ea0f912e53c92e82eb78a2a1631e92725fc8 | SOURCE_READY | NOT COMPLETE |
| [I-JEPA](https://github.com/facebookresearch/ijepa) | 52c1ae95d05f743e000e8f10a1f3a79b10cff048 | ACQUISITION_FAILED | NOT COMPLETE |
| [V-JEPA2](https://github.com/facebookresearch/vjepa2) | 204698b45b3712590f06245fbfba32d3be539812 | SOURCE_READY | NOT COMPLETE |
| [DINO-Tracker](https://github.com/AssafSinger94/dino-tracker) | UNAVAILABLE | ACQUISITION_FAILED | NOT COMPLETE |
| [GOT-JEPA](https://github.com/chenshihfang/GOT) | 5f382645ccb7331280a7bf07f3c5709d1072caf7 | SOURCE_READY | NOT COMPLETE |
| [CoTracker3](https://github.com/facebookresearch/co-tracker) | 82e02e8029753ad4ef13cf06be7f4fc5facdda4d | ACQUISITION_FAILED | NOT COMPLETE |
| [SAM2](https://github.com/facebookresearch/sam2) | 2b90b9f5ceec907a1c18123530e92e794ad901a4 | ACQUISITION_FAILED | NOT COMPLETE |
| [SAMURAI](https://github.com/yangchris11/samurai) | 76ba195984892b0d1e3db5d9c9f90bb62175680a | ACQUISITION_FAILED | NOT COMPLETE |
| [MOTIP](https://github.com/MCG-NJU/MOTIP) | ffc0e905ac196a603027eca8d18fb0dff48c8bcc | SOURCE_READY | NOT COMPLETE |
| [COVTrack](https://github.com/zekunqian/COVTrack) | UNAVAILABLE | ACQUISITION_FAILED | NOT COMPLETE |
| [TrackTrack](https://github.com/kamkyu94/TrackTrack) | ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da | ACQUISITION_FAILED | NOT COMPLETE |
| [TCEI](https://github.com/1941Zpf/TCEI) | 145d1b8431398156f8d9f854430e306fdee39eaa | SOURCE_READY | NOT COMPLETE |
| [MeMOTR](https://github.com/MCG-NJU/MeMOTR) | eb7a177b9cbcb89742ec69b2545ab3af2ea31a80 | ACQUISITION_FAILED | NOT COMPLETE |
| [MOTRv2](https://github.com/megvii-research/MOTRv2) | 1aac7c3beb093d4b523419a197e9531597e01798 | ACQUISITION_FAILED | NOT COMPLETE |
| [DiffMOT](https://github.com/ShuCvlab/DiffMOT) | eada72e74e54c153b30674d277b97882edc568b8 | SOURCE_READY | NOT COMPLETE |
| [HybridSORT](https://github.com/ymzis69/HybridSORT) | 396f8d30db13304c0cbaf1dcf2e16ded93ce1701 | SOURCE_READY | NOT COMPLETE |
| [BoT-SORT](https://github.com/NirAharon/BoT-SORT) | 251985436d6712aaf682aaaf5f71edb4987224bd | SOURCE_READY | NOT COMPLETE |
| [ByteTrack](https://github.com/FoundationVision/ByteTrack) | d1bf0191adff59bc8fcfeaa0b33d3d1642552a99 | SOURCE_READY | NOT COMPLETE |

Initial inspected code findings (partial audits)

MOTIP atffc0e905ac196a603027eca8d18fb0dff48c8bcc: training train.py (train_engine/train_one_epoch/prepare_for_motip), inference submit_and_evaluate.py and models/runtime_tracker.py. models/motip/id_decoder.py concatenates feature and local-ID embeddings, uses per-frame detection self-attention after the first layer, then cross-attends all past track/time tokens. Its causal mask rejects trajectory time>=current query time. models/motip/id_criterion.py uses masked CE or focal ID loss; official training also includes detector loss. configs/r50_deformable_detr_motip_dancetrack.yaml has feature256+ID256,6 layers,50 IDs+newborn,30-frame clips and interval<=4. data/naive_sampler.py enumerates legal clips and data/transforms.py randomly binds physical objects to local dictionary IDs and simulates occlusion/switches/newborns. RuntimeTracker.update processes current image then updates historical memory; frame detector input is replaced if adapting to frozen V3 proposals. Top-level LICENSE identifies Apache2.0; dependencies/individual subcomponents and pretrain license/access still require completion. The full official system and a fixed-V3 head adaptation must be reported separately; full-set attention/local dictionary competition overlap our proposed C contribution.

DINOv3 at6876159a11b4df116f30f667f8c9888617df0751: dinov3/hub/backbones.py defines vits16 (patch16,dim384,12 layers,6 heads,4 storage tokens); dinov3/models/vision_transformer.py::forward_features returns normalized CLS/storage/patch tokens. At128x256 this implies128 spatial patch tokens before additional CLS/storage tokens, not anatomical parts. dinov3/train/train.py and dinov3/train/ssl_meta_arch.py expose multi-crop/masking/EMA and DINO/iBOT/Gram-related objectives. LICENSE.md is the custom DINOv3 agreement dated2025-08-19; code and weight availability must not be treated as unrestricted Apache licensing. No weights downloaded or access terms accepted on behalf of the user. Full sampling/loss/dependency audit and lawful weight-access verification remain pending.

SU-T at479be89a27129f6879c62710a83abac6a8b261ee: README identifies a2025 preprint/AAAI2026 paper; tools/train.py and tools/su_tracker.py are entry candidates. trackers/SUT_UKF.py uses custom9-state/5-measurement filtering by default and includes a separate FilterPy UKF branch; trackers/UKF.py actually computes sigma points, but propagation falls back to linear F when fx is unset. The named fx in the tracker is a constant-velocity matrix, so a UKF label alone does not establish learned/nonlinear swimming dynamics. trackers/association.py::fish_iou_batch and association routes need active entry/default configuration tracing before porting. MIT top-level license verified; external FilterPy attribution remains separate. Isolate filter choice, geometry metric and lifecycle/output; copying the whole tracker would mix them. Training/loss/pretrained-data/feature audit is still incomplete.

TIDE atfba27aefd1da11bfc7f946240dd66d85ea02a39c: README points to2026 arXiv and MFT-Edge; src/track.py and src/lib/tracker/mc_bot_sort.py form a current-frame tracker route. src/lib/tracker/matching.py defines AGC variants, including AGCv1 adding inverse center-distance to IoU. Zero-distance numerical behavior and which AGC variant is actually called must be checked before a faithful geometry control. README training/reid64 and dual-branch network are not equivalent to our frozen V3/RGB pipeline. MIT top-level license verified; full network/active metric/loss/dataset audit remains pending.

FishMAE atcffcdeaf61c8c3a9768711a4615ff4af3899ca30 (public blobs verified against exact Git tree): mae-fish/train_mae.py calls imported engine_pretrain.py::train_one_epoch with4 arguments, but the engine requires epoch/loss_scaler and reads args.accum_iter; the raw example is not runnable as written. models_mae.py base uses patch16/embed768/depth12; normal pretraining adds patch positional embeddings, but mae-fish/fish_reid_mae/model.py manually bypasses forward_encoder and omits patch positional embedding. The ReID classifier applies BN/linear to CLS features; train.py optimizes CE+normalized batch-hard triplet, not dense local identity evidence. sampler.py stores p without using it, iterates identity chunks then globally shuffles indices; it does not enforce the advertised P×K batch structure. Pretrain exports both raw state dictionaries and wrapped resume dictionaries; MAEReID.load_state_dict(strict=False) must not silently accept the wrong container. The exact tree has no LICENSE/COPYING file and README has no verified paper citation. No external code was copied/executed; paper identity, lawful reuse, dataset construction and evaluation remain unresolved. A controlled FishMAE-style experiment must use an independently implemented correct sampler/forward/checkpoint adapter and verified permissible base, not claim faithful runnable original training.

All other projects, and unresolved fields above, require actual source reading and primary paper/weight references. No row is marked AUDIT_COMPLETE.
