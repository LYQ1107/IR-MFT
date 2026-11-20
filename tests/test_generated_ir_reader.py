import json

import cv2
import numpy as np
import pytest

from yolox.data.datasets.mot_rgbt import MOTDatasetRGBT


def _dataset(tmp_path):
    data = tmp_path / "data"
    generated = tmp_path / "generated"
    (data / "annotations").mkdir(parents=True)
    (data / "train/BT-005/visible").mkdir(parents=True)
    cv2.imwrite(str(data / "train/BT-005/visible/000001.jpg"), np.full((8, 12, 3), 31, np.uint8))
    annotation = dict(images=[dict(id=1, width=12, height=8, frame_id=1, video_id=1,
                                      vi_file_name="BT-005/visible/000001.jpg",
                                      ir_file_name="BT-005/infrared/000001.jpg")],
                      annotations=[], categories=[dict(id=1, name="fish")])
    (data / "annotations/train.json").write_text(json.dumps(annotation))
    for split, intensity in [("train", 19), ("dev", 147)]:
        dest = generated / split / "BT-005/infrared_diffv2ir"
        dest.mkdir(parents=True)
        cv2.imwrite(str(dest / "000001.png"), np.full((8, 12), intensity, np.uint8))
    return MOTDatasetRGBT(data_dir=str(data), json_file="train.json", name="train",
                          auxiliary_mode="diffv2ir", generated_ir_root=str(generated),
                          generated_ir_split="dev")


def test_generated_dev_uses_dev_namespace_and_replicates_scalar_intensity(tmp_path):
    dataset = _dataset(tmp_path)
    rgb, auxiliary, *_ = dataset.pull_item(0)
    assert rgb.shape == auxiliary.shape == (8, 12, 3)
    np.testing.assert_array_equal(auxiliary, np.full((8, 12, 3), 147, np.uint8))


def test_missing_generated_ir_does_not_fall_back_to_another_split(tmp_path):
    dataset = _dataset(tmp_path)
    (tmp_path / "generated/dev/BT-005/infrared_diffv2ir/000001.png").unlink()
    with pytest.raises(FileNotFoundError, match="DiffV2IR"):
        dataset.pull_item(0)
