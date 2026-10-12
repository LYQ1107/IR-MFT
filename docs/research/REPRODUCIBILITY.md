# Reproduction and current evidence

Research checkout: `/data3/liuyeqiang/IR-MFT`, branch `research/cfp-fishmot-v1`. Historical source is `e0cf4c724ff249f2d259cd4c4006c0c872f800d1`; all original tracked source files remain unchanged. Large assets stay outside Git. At most four GPUs permitted; actual runs use only physical GPU 2, with GPU 0/1 reserved.

## Verified historical runtime

Isolated `/data3/liuyeqiang/.venvs/cfp-fishmot-cu117`: Python 3.10.21, Torch 2.0.0+cu117, TorchVision 0.15.0+cu117, cuDNN 8.5, NumPy 1.24.3, OpenCV 4.10.0.84. Full environment freeze is `IR-MFT_assets/cfp/phase0/pip_freeze_cu117.txt`. cu118 reproduces the tracker given identical old detections but drifts during detector inference. Use cu117 for canonical V3 caches.

From the checkout:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
 /data3/liuyeqiang/.venvs/cfp-fishmot-cu117/bin/python tools/eval_mft26ir.py \
 --exp_file exps/yolox_mft26ir_rgbt_v3.py \
 --ckpt /data3/liuyeqiang/IR-MFT_assets/checkpoints/mft26ir_rgbt_v3_best_single_gpu.pth.tar \
 --data-root /data3/liuyeqiang/datasets/mft26-ir \
 --trackeval-root /data3/liuyeqiang/IR-MFT_assets/vendor/VUP-PFTrack \
 --output /data3/liuyeqiang/IR-MFT_assets/cfp/phase0/v3_reproduction_cu117_20261012 \
 --detections-output /data3/liuyeqiang/IR-MFT_assets/cfp/phase0/v3_reproduction_cu117_20261012/detections \
 --batch-size 8 --workers 8 --device cuda:0 --fp16 \
 --track-thresh .58 --track-buffer 45 --match-thresh .97 --min-box-area 100 --seed 0
```

These output paths already contain verified evidence; use fresh output paths for another run. Full cache/track parity is recorded in `IR-MFT_assets/cfp/phase0/full_cu117_parity.json`. HOTA 31.091, AssA 19.619, DetA 49.575, IDF1 35.044, MOTA 52.325, IDSW 560. Use TrackEval counts consistently: the separate motmetrics path gives 558 switches. Historical inputs/thresholds have prior test exposure and some noncausal synthetic inputs; this is reproduction, not blind final reporting.

## New experiment protocol and inputs

`configs/cfp/protocol_v2.json` freezes six association-train and two dev videos, seeds 0/1/2, identity namespaces and audited derived GT. Its old inherited informational IR/preprocessing fields are superseded by the explicit `evaluation_gt_policy` and separate `ir_provenance.json`/`causal_ir_v1.json`; do not silently edit the frozen split file. New official evaluations require `--derived-gt-root /data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt`. Detector V3 previously saw dev images; disclose this exposure.

`cfp/pseudo_ir.py` imports hash-pinned original generators from `IR-MFT_assets/scripts`, without running their old-path mains. Fixed temporal response + past-only EMA background is a separate input version. Geometry caching/component lookup are pixel-equivalent optimizations; offline chunk scheduling uses exact prefix-state checkpoints and does not introduce future observations. Full generation covers all 32,085 original train frames; sequential/chunk byte parity and detector-prefix checks pass. Generator runtime must be counted in an online speed report.

## Frozen identity diagnosis

The project analysis runtime `/data3/liuyeqiang/.venvs/cfp-fishmot` is isolated from the base environment. `tools/cfp_extract_identity_features.py` requires an explicit `--input-regime legacy_exploratory` or `causal`; causal mode rejects incomplete/probe generation manifests. Shared frozen ImageNet ResNet50 extracts aligned normalized RGB/IR embeddings. `tools/cfp_diagnose_features.py` uses `configs/cfp/feature_diagnosis_v1.json`, no historical test partition, temporally separated oracle-gallery queries and offline matched pair interventions. These are representation measures, not HOTA or deployed association.

## Tests

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
 /data3/liuyeqiang/.venvs/cfp-fishmot/bin/python -m pytest -q \
 tests/test_cfp_protocol.py tests/test_cfp_identity_data.py \
 tests/test_cfp_causal_ir.py tests/test_cfp_feature_diagnosis.py \
 tests/test_cfp_reid.py tests/test_cfp_association.py \
 tests/test_cfp_detection_features.py tests/test_cfp_control_views.py \
 tests/test_cfp_external_association.py tests/test_cfp_clear_events.py \
 tests/test_cfp_crop_views.py
```

All77 tests pass with MFT26IR_ROOT=/data3/liuyeqiang/datasets/mft26-ir; log: `/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/pytest_research_77_20261012.log`. The old default dataset path is retained in original code and overridden through its existing environment variable. External generator tests require the recovered scripts; single-frame adapter tests also require PyWavelets. Three seed0 CMUE training/evaluation rounds exist; sixteen temporal models (12cleanfirst-round/4actualround2) aretrained; final method validation/test outcomes remain outstanding.

## Actual canonical association commands

The default paired features for all three seeds were extracted with the same arguments below, using `--rgb-checkpoint .../reid_rgb_seedS/epoch12.pt`, `--ir-checkpoint .../reid_ir_causal_seedS/epoch12.pt`, batch 128 and eight workers. Do not overwrite existing evidence outputs; substitute fresh directories.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
 /data3/liuyeqiang/.venvs/cfp-fishmot/bin/python tools/cfp_cache_detection_features.py \
 --data-root /data3/liuyeqiang/datasets/mft26-ir-causal-v1 \
 --detections /data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_dev_detections \
 --protocol configs/cfp/protocol_v2.json --partition dev \
 --rgb-checkpoint /data3/liuyeqiang/IR-MFT_assets/cfp/phase1/reid_rgb_seed0/epoch12.pt \
 --ir-checkpoint /data3/liuyeqiang/IR-MFT_assets/cfp/phase1/reid_ir_causal_seed0/epoch12.pt \
 --output /data3/liuyeqiang/IR-MFT_assets/cfp/phase1/causal_dev_detection_features_seed0

PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /data3/liuyeqiang/.venvs/cfp-fishmot/bin/python tools/cfp_replay_association.py \
 --detections /data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_dev_detections \
 --features-root /data3/liuyeqiang/IR-MFT_assets/cfp/phase1/causal_dev_detection_features_seed0 \
 --protocol configs/cfp/protocol_v2.json --partition dev \
 --config configs/cfp/association_baseline_v1.json --mode rgb --appearance-weight .3 \
 --output /data3/liuyeqiang/IR-MFT_assets/cfp/phase1/causal_dev_rgb_seed0_w0.3

PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /data3/liuyeqiang/.venvs/cfp-fishmot/bin/python tools/cfp_evaluate.py \
 --data-root /data3/liuyeqiang/datasets/mft26-ir-causal-v1 \
 --derived-gt-root /data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt \
 --protocol configs/cfp/protocol_v2.json --partition dev \
 --predictions /data3/liuyeqiang/IR-MFT_assets/cfp/phase1/causal_dev_rgb_seed0_w0.3 \
 --output /data3/liuyeqiang/IR-MFT_assets/cfp/phase1/causal_dev_rgb_seed0_w0.3_evaluation \
 --trackeval-root /data3/liuyeqiang/CFP-FishMOT_references/TrackEval
```

External association CLI/config and each pinned reference are in `cfp_replay_external_association.py` / `external_association_v1.json`. It consumes the same feature banks and proposals; no native FastReID weights are claimed. `cfp_pair_identity_features.py` verifies matched seed/config/index/features before offline diagnosis. `cfp_compose_rgb_control_features.py` composes the predeclared independent RGB seed pairs; it does not claim a measured encoder runtime for cache composition.

Crossfit crop views are prepared by `cfp_prepare_crossfit_crops.py` at `/data3/liuyeqiang/IR-MFT_assets/cfp/phase2/crossfit_crops/fold{0,1}`. Their `protocol.json` declares training/heldout groups and no canonical dev fitting. Crops are symlinked, labels are local train classes, opposite-fold/canonical dev labels remain -1. Use original ImageNet initialization. New training checkpoints preserve optimizer/scaler plus Torch CPU/CUDA, sampler and augmentation RNG state. `--resume` is limited to an incomplete run's last completed epoch with identical input/config/source/runtime hashes; legacy completed checkpoints are preserved and cannot be relabeled as deterministically resumable.

## Representation controls, event traces and resume proof

`cfp_export_reid_features.py --embedding-stage post_neck` exports fixed weights without SGD. Proposal extraction accepts the same stage, and `--crop-variant` must match training metadata (legacy manifests mean `original`). Long-axis crops are created separately with `cfp_cache_identity_crops.py --crop-variant long_axis`; all six aligned encoders are trained; all54declaredassociation runs arecomplete andthevariant isnotretained.

`cfp_trace_clear_events.py --evaluation-root RUN_evaluation --derived-gt-root .../protocol_v2/gt --trackeval-root .../TrackEval --output FRESH.json` reads exact official matching events and writes full matched-row sidecars. It verifies all CLEAR counts, prediction/GT/metric hashes, and excludes historical-test diagnosis. `cfp_render_switch_cases.py` produces static figure artifacts with source hashes.

`cfp_verify_reid_resume.py --pretrained .../resnet50-0676ba61.pth --output FRESH` ran on GPU 2 with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Actual proof: `phase0/reid_resume_verification/verification.json`, status PASS. It interrupts only its dedicated fixture child, then compares model/optimizer/RNG states to uninterrupted real CUDA training. Completed runs remain immutable.

The ordinary-association campaign runner saves exact executed command vectors and full official metrics and resumes only complete, provenance-matching jobs. Actual post-neck capacity campaign:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /data3/liuyeqiang/.venvs/cfp-fishmot/bin/python tools/cfp_run_association_campaign.py \
 --plan configs/cfp/postneck_rgb_repeat_campaign_v1.json \
 --assets-root /data3/liuyeqiang/IR-MFT_assets/cfp/phase1 \
 --data-root /data3/liuyeqiang/datasets/mft26-ir-causal-v1 \
 --protocol configs/cfp/protocol_v2.json \
 --derived-gt-root /data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt \
 --trackeval-root /data3/liuyeqiang/CFP-FishMOT_references/TrackEval \
 --output /data3/liuyeqiang/IR-MFT_assets/cfp/phase1/causal_dev_rgb_repeat_postneck_campaign_summary.json
```


## Measured CMUE and causal PLIM module

Utility queries: `cfp_build_utility_queries.py --partition train --feature-roots phase2/oof_detection_features_postneck_fold0_seed0 phase2/oof_detection_features_postneck_fold1_seed0`, canonical protocol and frozen trial v1/v2. Fitting: `cfp_train_utility.py --kind {cmue,ordinary,quality} --seed0 --config configs/cfp/cmue_trial_v{1,2}.json --queries MATCHED_QUERIES --output FRESH`. Use proper separate option tokens (`--seed 0`). Replay: `cfp_replay_utility.py --kind KIND --checkpoint MATCHED/epoch8.pt --appearance-weight {.1,.3,.5} --config MATCHED_TRIAL --execution-config MATCHED_EXECUTION` with fixed causal dev bank and post-neck seed0 features. Actual argument vectors are in `phase2/cmue_{first,second}_seed0_campaign_summary.json`. This utility CLI permits canonical dev only; historical-test tuning is unavailable. Full logs/evaluation hashes are in registry.

`cfp_build_prediction_queries.py` requires main frozen original/post-neck feature space, `--config configs/cfp/plim_trial_v1.json --query-config configs/cfp/plim_query_v1.json`, canonical train/dev banks and protocol. It saves clean/actual strictly past indices and stopped current same-frame targets; GT/purity never become online model inputs. Predictive training iscomplete for twelve first-round/three-seed models plusfouractual-history round2 models; seed0tracking/write andalloffline modeldiagnoses arecomplete, seeds1/2 validation isunderway. `cfp/predictive.py` tests cover causal order, stopped gradients, empty histories, bounded memory and collapse-sensitive objectives.

New executions snapshot the exact small source/config bytes into each output's `source_snapshots`, in addition to immutable data/checkpoint hashes and checkout commit. Completed runs generated before source archiving retain their actual hashes; verified reconstructed originals are stored separately where available. Never rewrite a run's recorded commit or replace adverse outputs.

`cfp_train_predictor.py --architecture {gru,transformer} --objective {predictive,association} --history clean --seed 0 --device cuda --config configs/cfp/plim_trial_v1.json --query-config configs/cfp/plim_query_v1.json --training-config configs/cfp/plim_training_v1.json --queries TRAIN_QUERY_ROOT --output FRESH` trains only canonical-train queries. CUDA_VISIBLE_DEVICES=2 and CUBLAS_WORKSPACE_CONFIG=:4096:8 are required for this declared first trial. Eight epochs, identical video/identity-uniform budget; report SN-015 clean/actual and last/EMA/FIFO diagnostics after fixed epoch8. Canonical dev cannot fit or select a checkpoint. Online predictor replay iscomplete forfirstseed0 andinprogress forseeds1/2; the independentwrite policy isevaluatedseed0 andnotretained asaprovenpredictive benefit.

Two new predictor-association tests also pass (68 tests total across validated modules). `cfp_run_predictive_campaign.py --plan configs/cfp/plim_campaign_v1.json` records the30-run shared-candidate/equal-budget campaign. `cfp_diagnose_predictor.py` permits canonical dev frozen supervised diagnostics only, never fitting. Actual predictor queue command vectors/metadata: `phase3/predictor_seed0_clean_campaign_summary.json`.


## Thirdutility target andindependent write toggle

`cfp_derive_ranking_utility_queries.py --source phase2/utility_train_queries_full_context_seed0 --config configs/cfp/cmue_trial_v3.json --output FRESH` preservescompletecandidate/history/evidence/physical labels andpriorassociationutility, changingonlyrankingtargets. Runexistinggatefit/replay withv3 target/execution configs; allrealcommands arein `phase2/cmue_third_seed0_campaign_summary.json`. `utility_uncertainty_control_v1.json` declares parameter-free entropyrelativeIRweight oncompletecontext; sourceCLI supports `--kind uncertainty`, withnomodelcheckpoint.

`cfp_run_predictive_campaign.py` accepts `--features-root` forseed1/2 andchecksplan/execution/seed/checkpoint/GT hashes. `--execution-config configs/cfp/plim_actual_execution_v2.json` and`plim_actual_campaign_seed0_v2.json` selectactual-history training models atunchanged scoringbudget. Runsource bytes aresnapshotted beforemeasurement.

`cfp_replay_predictor.py --write-policy-config configs/cfp/plim_write_policy_v1.json --save-write-audit` isonlyanindependentpostgeometrywrite toggle atdeclaredweight.3. `plim_write_campaign_seed0_v1.json` comparesthree identicalmemory/models withpolicy on/off; allthree offresults matchtheearlierfullvideo filesbyte-for-byte. SavedJSONL decisions containcurrentobservables only. `cfp_audit_identity_writes.py` laterloadsGTtoauditphysicalpurity/gaps, withoutaffectingthesaveddecisions; unclear/initialanchors areexcludedfromexistingwrite precision. Auditcommands/hashes andallmetrics: `phase3_write_policy_results_20261012.json`.

Oneadditional entropy-control test alsopasses afterthe76-test suite. Purepolicy/gradient/target/physicalaudits aretested; exactCUDA interruptedresume proof remainsforReID, notyetforpredictor/utility trainers. No fullonline cost, finaltestor complete-goalclaim ismade.

First-round90-run temporal validation, actual-history12-run comparison andthree entropy-control runs are complete and registered. Model/data/source hashes and selection are in `configs/cfp/plim_first_three_seed_selection_v1.json`; no final-test authorization is implied.
