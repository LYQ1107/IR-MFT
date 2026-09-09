# IR-MFT code audit

This document records the interfaces inspected before the progressive RGB-IR
implementation. The original visible-only ByteTrack source is kept intact in
the parent directory; this repository is the development copy.

## Dataset and batch contract

The target annotation files are:

```text
/data1/LWR/vranlee/CELL/PFTrack/data/mft26-ir/annotations/train.json
/data1/LWR/vranlee/CELL/PFTrack/data/mft26-ir/annotations/test.json
```

The MFT26-IR image records use these fields:

```text
vi_file_name: BT-001/visible/000001.jpg
ir_file_name: BT-001/infrared/000001.jpg
frame_id, video_id, height, width, prev_image_id, next_image_id
```

Annotations contain `bbox`, `category_id`, `track_id`, `conf`, `iscrowd`, and
`area`. The original PFTrack JSON category list contains stale `pedestrian`
and `car` entries, but the actual annotations use category 1 only.

The inherited `yolox.data.datasets.mot.MOTDataset` has this contract:

```text
pull_item(index) -> image, res, img_info, image_id
__getitem__(index) -> image, padded_labels, img_info, image_id
```

`res` and `padded_labels` use six columns:

```text
[x1, y1, x2, y2, class_index, track_id]
```

After `TrainTransform`, the image is a float32 CHW array and labels are
`[class, cx, cy, w, h, track_id]`. `Trainer.train_one_iter` moves both to CUDA,
keeps `targets[:, :, :5]` for YOLOX loss, and retains column 5 only as an
unused track-ID tensor. The paired dataset therefore preserves this shape and
adds the two images as a six-channel input.

## Training path

`tools/train.py` obtains an experiment through `yolox.exp.get_exp`, constructs
`yolox.core.Trainer`, and calls `Trainer.train()`.

The relevant calls are:

```text
exp.get_model()
exp.get_data_loader(batch_size, is_distributed, no_aug)
DataPrefetcher.next() -> inps, targets
model(inps, targets)
```

`DataPrefetcher` does not impose a three-channel assumption; it transfers the
input tensor as received from the dataset. The inherited `YOLOX.forward` is:

```python
forward(self, x, targets=None)
```

It calls `backbone(x)`, then calls the head as
`head(fpn_outs, targets, x)` during training and `head(fpn_outs)` during
inference.

## Model path

`YOLOPAFPN.forward(input)` returns a three-element tuple:

```text
(pan_out2, pan_out1, pan_out0)  # P3, P4, P5
```

The inherited `YOLOXHead.forward(xin, labels=None, imgs=None)` consumes this
tuple. During training it returns the six loss values used by `YOLOX`; during
inference it returns decoded boxes in network-input coordinates with columns
`[cx, cy, w, h, objectness, class_confidence, class_index]`.

The V1 backbone therefore keeps the original head and returns the same three
FPN tensors after averaging the independent RGB and IR branches.

The detector checkpoint used for initialization is:

```text
/data1/LWR/vranlee/CELL/ByteTrack/YOLOX_outputs/yolox_mft26ir_visible/last_epoch_ckpt.pth.tar
```

It has top-level keys `start_epoch`, `model`, and `optimizer`. The visible-only
model state has 894 entries and matches the original YOLOX-X experiment model
exactly. For V1, each source `backbone.*` key is copied to both
`backbone.rgb_branch.*` and `backbone.ir_branch.*`; `head.*` keys are copied
unchanged.

## Augmentation contract

The original `TrainTransform` applies color distortion, horizontal mirroring,
resize, normalization, and padding to one image. Its random operations are not
safe to call independently for RGB and IR. V1 uses a new paired transform with
shared geometric decisions and disables Mosaic and MixUp so that the two files
remain pixel-aligned.

## Tracking and evaluation contract

`MOTEvaluator.evaluate` calls:

```python
tracker.update(output_results, img_info, img_size)
```

The inherited `BYTETracker.update` expects postprocessed detections and rescales
their network-input coordinates back to the original image using
`min(img_size[0] / original_height, img_size[1] / original_width)`. It then
performs the existing high-confidence IoU/fused-score association followed by
the low-confidence association.

The evaluator resets the tracker when `frame_id == 1` and writes one MOT text
file per sequence. `postprocess` performs confidence filtering and class-aware
batched NMS before tracking.

The repository's inherited `tools/track.py` prints MOT metrics against a
hard-coded `datasets/mot/train` path. It is not a valid MFT26-IR evaluator
without replacing the GT/result path with the seven target test sequences.

## Evaluation invariants for progressive experiments

All V1-V5 evaluations must use the same sequence list, GT files, detector
confidence, NMS threshold, tracker thresholds, and metric implementation. GT
files are read from:

```text
/data1/LWR/vranlee/CELL/PFTrack/data/mft26-ir/test/*/gt/gt.txt
```

No annotation or GT file is modified by this repository.
