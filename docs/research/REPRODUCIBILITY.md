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

`cfp/pseudo_ir.py` imports hash-pinned original generators from `IR-MFT_assets/scripts`, without running their old-path mains. Fixed temporal response + past-only EMA background is a separate input version. Geometry caching/component lookup are pixel-equivalent optimizations; offline chunk scheduling uses exact prefix-state checkpoints and does not introduce future observations. Generator runtime must be counted in an online speed report. Full generation is not yet complete at this milestone.

## Frozen identity diagnosis

The project analysis runtime `/data3/liuyeqiang/.venvs/cfp-fishmot` is isolated from the base environment. `tools/cfp_extract_identity_features.py` requires an explicit `--input-regime legacy_exploratory` or `causal`; causal mode rejects incomplete/probe generation manifests. Shared frozen ImageNet ResNet50 extracts aligned normalized RGB/IR embeddings. `tools/cfp_diagnose_features.py` uses `configs/cfp/feature_diagnosis_v1.json`, no historical test partition, temporally separated oracle-gallery queries and offline matched pair interventions. These are representation measures, not HOTA or deployed association.

## Tests

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
 /data3/liuyeqiang/.venvs/cfp-fishmot/bin/python -m pytest -q \
 tests/test_cfp_protocol.py tests/test_cfp_identity_data.py \
 tests/test_cfp_causal_ir.py tests/test_cfp_feature_diagnosis.py
```

24 passed at this milestone. External generator tests require the recovered scripts; single-frame adapter tests also require PyWavelets. No CMUE/PLIM training or new-method final-test outcome exists yet.
