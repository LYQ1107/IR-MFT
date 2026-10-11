import importlib.util
from pathlib import Path

import numpy as np
import torch

from cfp.reid import temporal_batch_hard

spec = importlib.util.spec_from_file_location('cfp_reid_cli', Path(__file__).parents[1] / 'tools/cfp_train_reid.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_identity_sampler_never_uses_unseen_dev_ids_and_spans_lifespan():
    records = [dict(partition='train', train_label=i, frame_id=frame) for i in range(8) for frame in range(1, 129)]
    records += [dict(partition='dev', train_label=-1, frame_id=1)]
    sampled = module.IdentitySampler(records, 0).sample()
    assert len(sampled) == 64
    assert len(np.unique(sampled)) == 64
    assert all(records[i]['partition'] == 'train' for i in sampled)
    for identity in range(8):
        times = [records[i]['frame_id'] for i in sampled if records[i]['train_label'] == identity]
        assert len(times) == 8 and min(times) <= 16 and max(times) > 112


def test_triplet_excludes_adjacent_positives_and_easy_cross_video_negatives():
    embedding = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
    labels = torch.tensor([0, 0, 1, 1])
    loss, coverage = temporal_batch_hard(embedding, labels, torch.tensor([1, 2, 1, 2]), torch.zeros(4, dtype=torch.long))
    assert coverage == 0 and loss.item() == 0
    loss, coverage = temporal_batch_hard(embedding, labels, torch.tensor([1, 30, 1, 30]), torch.tensor([0, 0, 1, 1]))
    assert coverage == 0 and loss.item() == 0
    loss, coverage = temporal_batch_hard(embedding, labels, torch.tensor([1, 30, 1, 30]), torch.zeros(4, dtype=torch.long))
    assert coverage == 1 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(embedding.grad).all()


def test_rgb_crop_reader_does_not_require_any_ir(tmp_path):
    import cv2
    from cfp.identity_data import SingleIdentityDataset
    folder = tmp_path / 'train/a'
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / 'rgb.png'), np.full((16, 32, 3), 127, np.uint8))
    records = [dict(image_id=1, rgb_file='a/rgb.png', ir_file='must_not_read.png', bbox_xywh=[0, 0, 32, 16])]
    image, position = SingleIdentityDataset(tmp_path, records, 'rgb')[0]
    assert image.shape == (3, 128, 256) and position == 0


def test_frozen_representation_candidate_changes_metric_without_classifier_or_state():
    from cfp.reid import FishReID
    torch.manual_seed(0)
    model = FishReID(8, 32).eval()
    model.neck.running_mean.copy_(torch.linspace(-1, 1, 32))
    inputs = torch.randn(2, 3, 64, 64)
    before = model.neck.running_mean.clone()
    with torch.inference_mode():
        pre, logits_pre = model(inputs)
        post, logits_post = model(inputs, embedding_stage='post_neck')
    assert torch.equal(logits_pre, logits_post)
    assert torch.equal(before, model.neck.running_mean)
    assert not torch.allclose(pre, post)
    for embedding in (pre, post):
        assert torch.isfinite(embedding).all()
        assert torch.allclose(embedding.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_epoch_resume_restores_independent_sampler_and_augmentation_streams():
    import copy
    records = [dict(partition='train', train_label=i, frame_id=frame) for i in range(8) for frame in range(1, 65)]
    sampler = module.IdentitySampler(records, 0)
    rng = np.random.default_rng(0)
    config = dict(horizontal_flip_probability=.5, erase_probability=.8, erase_area_fraction=[.05, .2])
    images = torch.arange(8 * 3 * 16 * 32).reshape(8, 3, 16, 32).float()
    sampler.sample()
    module.augment(images.clone(), rng, config)
    saved = copy.deepcopy(dict(sampler=sampler.rng.bit_generator.state, augmentation=rng.bit_generator.state))
    expected_positions = sampler.sample()
    expected_images = module.augment(images.clone(), rng, config)
    restored_sampler = module.IdentitySampler(records, 999)
    restored_rng = np.random.default_rng(999)
    restored_sampler.rng.bit_generator.state = saved['sampler']
    restored_rng.bit_generator.state = saved['augmentation']
    assert np.array_equal(expected_positions, restored_sampler.sample())
    assert torch.equal(expected_images, module.augment(images.clone(), restored_rng, config))
