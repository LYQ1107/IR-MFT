# Phase VII30-reference source audit

Status: IN_PROGRESS. Source readiness and exact Git commit inventory do not constitute a completed semantic audit. New sparse snapshots omit weights/media and live outside IR-MFT. Failed checkout/network attempts are preserved. Source acquisition does not execute downloaded training scripts. Licensed inherited native tracker code is used by fixed baselines; the separate Apache-licensed DINOv2 random CPU forward is recorded below. No newly acquired external pretrained weights are used.

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
| [DINOv2](https://github.com/facebookresearch/dinov2) | 7764ea0f912e53c92e82eb78a2a1631e92725fc8 | SOURCE_READY | SOURCE AUDIT COMPLETE |
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
| [BoT-SORT](https://github.com/NirAharon/BoT-SORT) | 251985436d6712aaf682aaaf5f71edb4987224bd | SOURCE_READY | SOURCE AUDIT COMPLETE |
| [ByteTrack](https://github.com/FoundationVision/ByteTrack) | d1bf0191adff59bc8fcfeaa0b33d3d1642552a99 | SOURCE_READY | SOURCE AUDIT COMPLETE |

Initial inspected code findings (partial audits)

MOTIP atffc0e905ac196a603027eca8d18fb0dff48c8bcc: training train.py (train_engine/train_one_epoch/prepare_for_motip), inference submit_and_evaluate.py and models/runtime_tracker.py. models/motip/id_decoder.py concatenates feature and local-ID embeddings, uses per-frame detection self-attention after the first layer, then cross-attends all past track/time tokens. Its causal mask rejects trajectory time>=current query time. models/motip/id_criterion.py uses masked CE or focal ID loss; official training also includes detector loss. configs/r50_deformable_detr_motip_dancetrack.yaml has feature256+ID256,6 layers,50 IDs+newborn,30-frame clips and interval<=4. data/naive_sampler.py enumerates legal clips and data/transforms.py randomly binds physical objects to local dictionary IDs and simulates occlusion/switches/newborns. RuntimeTracker.update processes current image then updates historical memory; frame detector input is replaced if adapting to frozen V3 proposals. Top-level LICENSE identifies Apache2.0; dependencies/individual subcomponents and pretrain license/access still require completion. The full official system and a fixed-V3 head adaptation must be reported separately; full-set attention/local dictionary competition overlap our proposed C contribution.

DINOv3 at6876159a11b4df116f30f667f8c9888617df0751: dinov3/hub/backbones.py defines vits16 (patch16,dim384,12 layers,6 heads,4 storage tokens); dinov3/models/vision_transformer.py::forward_features returns normalized CLS/storage/patch tokens. At128x256 this implies128 spatial patch tokens before additional CLS/storage tokens, not anatomical parts. dinov3/train/train.py and dinov3/train/ssl_meta_arch.py expose multi-crop/masking/EMA and DINO/iBOT/Gram-related objectives. LICENSE.md is the custom DINOv3 agreement dated2025-08-19; code and weight availability must not be treated as unrestricted Apache licensing. No weights downloaded or access terms accepted on behalf of the user. Full sampling/loss/dependency audit and lawful weight-access verification remain pending.

SU-T at479be89a27129f6879c62710a83abac6a8b261ee: README identifies a2025 preprint/AAAI2026 paper; tools/train.py and tools/su_tracker.py are entry candidates. trackers/SUT_UKF.py uses custom9-state/5-measurement filtering by default and includes a separate FilterPy UKF branch; trackers/UKF.py actually computes sigma points, but propagation falls back to linear F when fx is unset. The named fx in the tracker is a constant-velocity matrix, so a UKF label alone does not establish learned/nonlinear swimming dynamics. trackers/association.py::fish_iou_batch and association routes need active entry/default configuration tracing before porting. MIT top-level license verified; external FilterPy attribution remains separate. Isolate filter choice, geometry metric and lifecycle/output; copying the whole tracker would mix them. Training/loss/pretrained-data/feature audit is still incomplete.

TIDE atfba27aefd1da11bfc7f946240dd66d85ea02a39c: README points to2026 arXiv and MFT-Edge; src/track.py and src/lib/tracker/mc_bot_sort.py form a current-frame tracker route. src/lib/tracker/matching.py defines AGC variants, including AGCv1 adding inverse center-distance to IoU. Zero-distance numerical behavior and which AGC variant is actually called must be checked before a faithful geometry control. README training/reid64 and dual-branch network are not equivalent to our frozen V3/RGB pipeline. MIT top-level license verified; full network/active metric/loss/dataset audit remains pending.

FishMAE atcffcdeaf61c8c3a9768711a4615ff4af3899ca30 (public blobs verified against exact Git tree): mae-fish/train_mae.py calls imported engine_pretrain.py::train_one_epoch with4 arguments, but the engine requires epoch/loss_scaler and reads args.accum_iter; the raw example is not runnable as written. models_mae.py base uses patch16/embed768/depth12; normal pretraining adds patch positional embeddings, but mae-fish/fish_reid_mae/model.py manually bypasses forward_encoder and omits patch positional embedding. The ReID classifier applies BN/linear to CLS features; train.py optimizes CE+normalized batch-hard triplet, not dense local identity evidence. sampler.py stores p without using it, iterates identity chunks then globally shuffles indices; it does not enforce the advertised P×K batch structure. Pretrain exports both raw state dictionaries and wrapped resume dictionaries; MAEReID.load_state_dict(strict=False) must not silently accept the wrong container. The exact tree has no LICENSE/COPYING file and README has no verified paper citation. No external code was copied/executed; paper identity, lawful reuse, dataset construction and evaluation remain unresolved. A controlled FishMAE-style experiment must use an independently implemented correct sampler/forward/checkpoint adapter and verified permissible base, not claim faithful runnable original training.

All other projects, and unresolved fields above, require actual source reading and primary paper/weight references. Three rows now have completed source-level field audits with explicit reuse/access restrictions;27 remain partial or unaudited. Source audit completion does not claim a reproduced pretrained system.

## Follow-up acquisition and primary verification

The bounded exact-commit public-blob follow-up has terminated with all30 requested entries recorded:29 exact heads are available, while COVTrack's GitHub API request failed with preserved TLS errors. Of the16 new snapshots,5 selected-entry sets are complete,10 are partial, and1 failed; this count is about the selected24-file entry sets, not full clones or completed semantic audits. All14 historical source trees were reused without mutation. The manifest and per-file Git-blob/SHA256 proofs live at `/data3/liuyeqiang/IR-MFT_assets/cfp/phase7/reference_audit_http_v1.json`. The repository registry records successful and missing files independently from its prior sparse-checkout status. COVTrack's public official repository is verified to exist; network failure is not evidence that the project is unavailable.

[FishMAE's publisher page](https://www.sciencedirect.com/science/article/abs/pii/S0144860926000622) confirms the paper title, Aquacultural Engineering114(2026)102746, DOI10.1016/j.aquaeng.2026.102746, and links this source repository. This resolves paper identity/year, not absent code-license terms or dataset/weight access. The paper studies long-term individual retrieval, which differs from same-frame crowded online MOT.

FDTA atb3b3b778acf93fa4269663b5ea1fd1d5ff8c6730: [the primary preprint](https://arxiv.org/abs/2512.02392) is submitted2025-12-02; its repository reports CVPR2026. Actual source read covers train.py, submit_and_evaluate.py, models/fdta/{fdta,Temporal_Adapter,trajectory_modeling,id_decoder,id_criterion}.py, models/Spatial_Adapter/depth_predictor.py, models/runtime_tracker.py, models/deformable_detr/deformable_detr.py and DanceTrack/BFT configs. The default trajectory shape is[B=1,G=6,T=30,N,256], with256-dimensional ID embedding concatenated to512; vocab50+unmatched and6decoder layers. BFT usesT20. Temporal_Adapter applies6layers/8heads with missing-frame and lower-triangular causal masks, but squeezes batch0 and assumesB1. ID cross-attention masks memorytime>=querytime. Runtime detects current observations, predicts IDs from earlier memory and then appends current writes; the default object-max and optional Hungarian routes differ and should not be silently conflated.

FDTA trains detector classification/box/GIoU plus depth, quality-filtered instance contrastive and masked CE/focal ID objectives. Identity Adapter selects detector matches withIoU>=0.5, maps instance IDs and passesIoU weights to its contrastive criterion; complete positive/negative normalization still requires further reading. DepthPredictor merges projected multiscale feature maps and predicts discrete depth; original backbone features are detached for that auxiliary path. tools/gen_depthmaps.py reads the entire training video and normalizes targets by its global maximum, so those maps are offline training supervision. RuntimeTracker.update passes the current image without external depthmaps: do not incorrectly label its RGB-only learned depth prediction as future input. Faithful raw online accounting includes that learned branch; precomputed maps cannot be treated as free causal observations if an adaptation uses them at inference.

FDTA's top-level license is MIT with2025YuqingShao copyright. Inherited MOTIP/DeformableDETR/TrackEval and VideoDepthAnything/weights remain separately audited. The declared environment is Python3.12, torch2.4.0/torchvision0.19.0, accelerate and a compiled multiscale deformable-attention extension; current torch2.0 compatibility is unverified. Its detector-native256 embeddings, depth targets and joint detector training are not a frozenV3/RGB512 baseline. Spatial/temporal/identity adapters and contrastive instance refinement overlap broad evidence-learning claims; an adapter port alone is not novelty. No external code or weights were executed/copied. The detailed audit remains partial.

DINOv2 at7764ea0f912e53c92e82eb78a2a1631e92725fc8:13actual source/model-card/license files were read and independently checked against the exact Git tree's blob hashes. [The primary paper](https://arxiv.org/abs/2304.07193) was submitted2023/revised2024; general image/pixel features are its scene/contribution scope, not fish identity supervision. dinov2/hub/backbones.py::dinov2_vits14 uses384dim/12blocks/6heads/patch14, distilled from the large pretrained teacher on LVD142M. forward_features returns normalized CLS/register/spatial patches. Direct PatchEmbed.forward asserts H/W multiples14:128×256 cannot be passed unchanged even though MODEL_CARD describes truncation. A declared126×252 resize gives162spatial patches; a register variant adds4storage tokens. Padding/resizing must be matched across controls, never silently treated as the original crop.

Actual training train.py/SSLMetaArch.forward_backward uses2global+8local views, an infinite sharded image sampler and stopped teacher with EMA. DINO matches sharpened teacher distributions across global/local views; iBOT uses masked patch-distribution CE with inverse masked-count weighting; KoLeo spreads normalized features from each global-crop chunk by negative log nearest-neighbor distance, excluding self and separately treating the other global crop. No fish-ID PK sampler or same-frame cross-fish supervision is present. Input features are current image crops; this backbone has no online future-frame requirement.

DINOv2's LICENSE and explicit [model card](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md) identify Apache2. The official small backbone URL is `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth`; the live HEAD attempt failed TLS, so no pretrained weights are acquired. A separate licensed random-initialized CPU source smoke has now executed. Official requirements include torch2.0/torchvision0.15/xformers0.0.18; SSLMetaArch explicitly requires xformers for its distributed FSDP/AMP/CUDA training route. Torch2CPU single-tensor inference fallback succeeds without xformers;140×266padding yields190patch tokens and CLS384. LVD142M's private curated corpus prevents a verified zero-overlap claim against fish data. Port frozenCLS/patch features and independently supervised fish/local adapters as matched A1/A2 controls; new384coordinates require a new matching producer, not old512GRU. EMA/patchdistillation/local tokens are established prior mechanisms. Weight delivery and pretrained crop metrics remain pending; source-field audit is complete with explicit access/external-overlap limits. Official training is not reproduced.

## Completed source-field audit records

Source-level completion covers the11requested fields at exact commits, including explicit exclusions where pretrained weights/data cannot be verified. It does not certify an executed official pretrained system. Proof paths/hashes remain in PHASE_VII_REPOSITORY_REGISTRY.json.

### ByteTrack: d1bf0191adff59bc8fcfeaa0b33d3d1642552a99

- **paper_year_scene_contribution**: {'primary': 'https://arxiv.org/abs/2110.06864', 'title': 'ByteTrack: Multi-Object Tracking by Associating Every Detection Box', 'year': 'Preprint2021/revised2022; ECCV2022 per official repository', 'scene': 'PedestrianMOT17/MOT20, HiEve and BDD100K demonstrations; not a fish identity encoder', 'contribution': 'Associate high-confidence detections, then retain low-confidence detections for recovery rather than dropping them.'}

- **license_and_weight_access**: {'code': 'MIT copyright2021YifuZhang; imported/framework/inherited notices remain separately applicable. No external code copied by this new audit.', 'weights': 'README public GoogleDrive/Baidu model links; no separate verified weight license/access artifact. Full official detector weight reproduction excluded until lawful provenance verified; frozenV3 tracking-only control needs no new ByteTrack weights.'}

- **training_entry_and_sampler**: tools/train.py main ->get_exp ->Trainer.train; exps/example/mot/yolox_x_ablation.py get_data_loader: MOTDataset/MosaicDetection/InfiniteSampler/YoloBatchSampler, image-level shuffled infinite sampling. No association-head or physicalID PKtraining. MOTDataset carries track_id but detector losses read class and box columns only.

- **inference_entry_and_temporal_scope**: tools/track.py ->MOTEvaluator.evaluate ->BYTETracker.update current detector boxes. Predict8stateKalman first, high stage confirmed+lost, low stage remaining active only, unconfirmed/birth/removal afterward. No future frames or GT in tracker inputs. Scalar confidence exactly equal to track_thresh excluded from both high/low due strict comparisons; preserve faithful boundary semantics.

- **core_model_files_and_shapes**: byte_tracker.py/matching.py/kalman_filter.py: boxes[N,5] or tensor boxes/objectscore/classscore; XYAH+4velocities mean8/cov8x8, measurement4; cost[Ntracks,Ndets], outputconfirmedSTrack list. YOLOXhead emits multiscale strides8/16/32, perlocation4box+objectness+class channels; num_classes1 in MOT exp, targetsclass/xywh plusignoredtrackingID, image800x1440 nominal dynamicresize.

- **loss_and_positive_negative_construction**: Tracking association is untrained IoU/LAPJV, no ReID or IDclassification loss. YOLOX detection uses5*IoU+BCEobjectness+BCEclass+optionalL1, SimOTA dynamicK from top10IoU sum/clamp>=1, conflicts resolved by lowestcost peranchor. Positives are detector-location/GT matches, not samephysicalfish identity pairs; background objectness negatives and IoUweighted class targets.

- **pretrained_sources**: COCO YOLOX initialization; ablationCrowdHuman+MOT17halftrain, fullMOT17 addsCityperson/ETHZ, MOT20 usesCrowdHuman+MOT20. Dataset/model licenses independent. No new data/weights downloaded. Official full-detector comparisons differ from fixedV3 candidate-only adaptation.

- **GPU_CUDA_dependencies**: Tracking aloneCPU: NumPy/SciPy/lap/Cythonbbox/OpenCV; np.float aliases need isolated compatibility on NumPy>=1.24. Detector torch>=1.7/torchvision>=0.10, trainingNCCL/AMP, optionalYOLOX C++extension in setup.py; CUDAneededforofficialdetector training/inference, notIoU matching.

- **IR_MFT_compatibility**: Useful frozenV3 geometry/lowconfidence baseline; no crop identity embeddings or local tokens. Native8stateXYAH differsfromotherUKF/BoTstate; wholetracker swaps change lifecycle/output/birth and must remain separate from metric/filter ablations.

- **mechanisms_to_port_and_exclude**: Retain known lowconfidence recovery/observed-current/past lifecycle controls. ExistingIR-MFT fork alreadyinherits ByteTrack; do not recopy entiretracker or trainnewdetector for first fixedV3 control. Defaulttrack threshold0.6/tools, birthtrack_thresh+0.1, lowIoUcostthreshold0.5, unconfirmed0.7; currentproject0.58/0.97 are explicitly different configurations.

- **novelty_overlap**: Low-score recovery, Kalman/IoU assignment, lost-track reactivation and confidence fusion are established. They cannot be credited as novel EvidenceFishMOT identity learning.


### BoT-SORT: 251985436d6712aaf682aaaf5f71edb4987224bd

- **paper_year_scene_contribution**: {'primary': 'https://arxiv.org/abs/2206.14651', 'title': 'BoT-SORT: Robust Associations Multi-Pedestrian Tracking', 'year': '2022primarypreprint', 'scene': 'MOT17/MOT20pedestrians, camera motion/appearance/motion fusion; not fishReID', 'contribution': 'XYWHKalmanstate, cameramotion compensation, gatedappearance and IoU fusion plus ByteTrack-style lowconfidence recovery'}

- **license_and_weight_access**: {'code': 'TopMIT2022NirAharon; nestedfast_reid/LICENSEApache2. Inherited ByteTrack/YOLOX and imported dependencies need preserved notices. No externalsource copiedinthisaudit.', 'weights': 'PublicMOT17/MOT20SBS-S50GoogleDrive links andByteTrack/YOLOXdetectors inREADME; no verified independent weightlicense/access proof. Fullweightreproduction excludeduntilresolved; existingfixedV3/fish-coordinate geometry/lifecycle control doesnotneednewofficialweights.'}

- **training_entry_and_sampler**: Officialdetectortraining refersByteTrack/YOLOX; FastReID fast_reid/tools/train_net.py with MOT17/sbs_S50.yml ->Base-SBS ->Base-bagtricks. DefaultNaiveIdentitySampler, B64/K16 identitygroupedcroppatches, not samephysicalfish/sameframe constrained sampling. SBS60epochs, Adam3.5e-4, AMP, initialbackboneFREEZE_ITERS1000; no learnableassociationhead.

- **inference_entry_and_temporal_scope**: tools/track.py or tools/demo.py ->BoTSORT.update ->predict ->GMC ->high association ->low/unconfirmed/birth/write. OnlycurrentRGBand earliertrack states/EMA required forliveGMC. tools/track.py default cmc=file and perMOTsequence thresholdoverrides are officialbenchmark choices; fileGMC reads unvalidated offline rows and no generationcausalityproofisprovided, sofilemode excludedfrom strictonlineclaim. sparseOptFlow reads previous/current and updatesprev aftermotionestimate.

- **core_model_files_and_shapes**: tracker/bot_sort.py STrack: mean8/cov8x8 XYWH+4velocities, measurement4; lost width/heightvelocities zeroed. GMC2x3 affine liftedto8x8 withtranslatedcenters. PaircostNtracks xMdetections. SBS-S50 build_resnest_backbone defaultdepth50x, feature2048, GeM andBNneck, evalafter-neck normalizedbyFastReIDInterface; input384x128 stretchRGBcrops, currentfeatures+normalizedEMA0.9.

- **loss_and_positive_negative_construction**: Association is untrainedLAPJV. SBS: CircleSoftmax classifier(scale64/margin.35) + CElabel_smoothing.1 + batchhardTriplet(margin0 softform, nonnormalizedfeatures). Positives samepersonIDwithK16samples; negatives otherpersonIDs, notvideo/sameframe hardfish construction. Trackscorecost min(fusedIoU, cosinedistance/2) withrawIoUproximitymask andappearance cutoff; confidence fusion disabledforMOT20.

- **pretrained_sources**: ImageNet/ResNeStFastReID backbone initialization and separateMOT17/MOT20 personpatchIDmodels, ByteTrackYOLOX COCO/pedestrian detectorweights. RGBsinglemodality; learned fish/IR signal unavailablebydefault. OfficialfullsystemandfixedV3/head/fish-coordinate adaptation aredifferentprotocols.

- **GPU_CUDA_dependencies**: CPUtrackers requireNumPy/SciPy/lap/Cythonbbox/OpenCV; np.float/np.float_ compatibility onmodernNumPy mustbeisolated. FastReID torch/torchvisionCUDAhalf batch8 orCPUfloat; yacs/pyyaml/h5py/etc, detectorCUDA/AMP. LiveGMC addscurrentRGB opticalflow/featureextraction cost; precomputedfiles notfreeend-to-endonlinecost.

- **IR_MFT_compatibility**: SamefrozenV3proposals andfish featurecoordinate cancompare nativecost/lifecycle separately. XYWHstate cannotbeinstalled asXYAHfilterwithoutSTrack conversion changes; existingPhaseV CFPlifecycle source alreadydefines matchedcoordinate/native controls. S50personweights/stretch384x128 notequivalenttofish letterbox128x256.

- **mechanisms_to_port_and_exclude**: PortisolatedXYWHmotion, gatedappearance/IoUmin and causalGMC ablations withfixedfeatures/writes/output. Do notrecopywholetracker orclaimGMCfilecausality; keepdata-specificthresholdoverrideandnativebirth/duplicate rules explicit. Nativehigh .6/low .1/new .7, rawIoUdistanceproximity .5, appearance distance/2 cutoff .25 aredefaults, notour .58/.97 policy.

- **novelty_overlap**: Confidence/geometryappearancefusion, normalizedEMA, motion-statechange andcamera compensation areestablishedpriors. Anygainfromlifecyclethresholds/GMC cannotbeclaimed asnewfine-grainedidentitylearning.


### DINOv2: 7764ea0f912e53c92e82eb78a2a1631e92725fc8

- **paper_year_scene_contribution**: DINOv2: Learning Robust Visual Features without Supervision; primary arXiv2304.07193, submitted2023/revised2024. General image/pixel features, not individual fish labels.

- **license_and_weight_access**: Apache2 code/license and modelcard. OfficialDINOsmall URL repeatedlyfailsTLS. HFMeta namespace convertedSmall Apache2file atrevision150c8e7bb7cef2d30ec31b13a517af14840ee3f7 haspublisherSHA256ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1 andsize88249960. OfficialHFAPIandpublicmirror requestsfail; no weights acquired. Pretrainedexperiment access pending, not an audited pretrained result.

- **training_entry_and_sampler**: dinov2/train/train.py main; SSLMetaArch.forward_backward student global2/local8 views, teacher no-grad and EMA; defaultsharded-infinite sampler, no fishID PK/hard-frame sampling.

- **inference_entry_and_temporal_scope**: dinov2/hub/backbones.py dinov2_vits14 ->models/vision_transformer.py forward_features; currentimage, no futureorphysicallabels. Apache-licensed actualCPUforward on currentTorch2.0 succeedswithoutxformers using singleTensor fallback, butrandominitializationonly.

- **core_model_files_and_shapes**: Small14 dim384/depth12/heads6, CLS+patch tokens; actual22056576parameters.128x256failsassertion, mean-normalizedpaddingto140x266 yields190spatialtokens+CLS verified. Registersareseparatevariant excludedfromSmall14pilot.

- **loss_and_positive_negative_construction**: DINO teacher-distribution CE for other global/current local views; iBOT teacher-stopped masked patch CE with inverse mask-count weighting; KoLeo normalized batch nearest-neighbor negative-log distance excludes self and same-image otherglobal view by cropchunk. No physical-ID positives/negatives or cross-frame association loss.

- **pretrained_sources**: ViTS14 distilled from pretrainedViTg; LVD142M curated142Mimages. Exact fishdataset overlap cannot be verified from private corpus; do not claim zero external overlap.

- **GPU_CUDA_dependencies**: SSLMetaArchrequiresxformers/FSDP/CUDA/AMP forofficialtrain; dense singleTensor inference actuallysucceeds onTorch2.0CPU withXFORMERS_DISABLED=1. No package installation ormodelweightdownload. GPUforward/pretrainedquality yetunmeasured.

- **IR_MFT_compatibility**: Current-only RGB backbone fits frozen-global/local crop controls, new384dimension coordinate must create matched producer and exclude outer/inner videos for learned fish adapters. Old512GRU cannot be reused across latent spaces.

- **mechanisms_to_port_and_exclude**: Use frozenCLS/patchTokens as A1/A2 controls, then independent supervised fish adapters/local pooling. Do not repeat142M-pretraining or download giant/register/depthheads unnecessarily.

- **novelty_overlap**: Teacher EMA, patchdistillation and localtoken features are prior mechanisms; merely adoptingDINOpatches or EMAis not new identity prediction.
