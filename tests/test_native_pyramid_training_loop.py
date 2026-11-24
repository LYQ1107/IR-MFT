"""Restart equivalence, preserved address lineage, and bounded disk writes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.native_pyramid_train_data import PixelClip, TrainClip, address_bytes, epoch_clips, schedule_proof
from tools.native_pyramid_training_loop import (
    guard_storage, prefix_checkpoint, restore_delivered_prefix,
    run_training_to_epoch, state_sha256, validate_memory_at_prefix,
)
from tools.run_pseudo_ir_comparison import sha256
from yolox.models.native_pyramid_fusion import NativePyramidFusion, NativeTemporalState
from yolox.models.yolo_head import YOLOXHead


class TinyProvider:
    """Small synthetic images, strictly training-loop engineering; no dataset claim."""
    lengths = dict(A=5, B=1)
    epoch = None

    def begin_epoch(self, epoch):
        assert self.epoch is None
        self.epoch, self.prepared, self.digest = epoch, 0, hashlib.sha256()
        self.used = {s: [] for s in self.lengths}
        self.expected_next = iter(epoch_clips(self.lengths, epoch))

    def prepare(self, clip, cue):
        assert next(self.expected_next) == clip
        self.prepared += 1
        self.digest.update(address_bytes(clip))
        self.used[clip.sequence].extend(clip.frames)
        rng = np.random.default_rng([0 if clip.sequence == 'A' else 1, clip.frames[0]])
        pixels = torch.from_numpy(rng.normal(size=(len(clip.frames), 3, 64, 64)).astype(np.float32))
        target = torch.zeros(len(clip.frames), 500, 5)
        target[:, 0, 1:] = torch.tensor([32., 32., 16., 8.])
        return PixelClip(clip, pixels, pixels.clone(), target,
            torch.ones(len(clip.frames), 1, 64, 64, dtype=torch.bool), tuple(True for _ in clip.frames),
            tuple(np.zeros((16, 16), np.uint8) for _ in clip.frames))

    def finish_epoch(self):
        assert next(self.expected_next, None) is None
        proof = schedule_proof(self.lengths, self.epoch)
        assert proof['address_sha256'] == self.digest.hexdigest()
        for s, n in self.lengths.items(): assert self.used[s] == list(range(1, n+1))
        self.epoch = self.expected_next = None
        return dict(proof, actual_address_sha256=self.digest.hexdigest(), complete_all_Train_frames=True)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Conv2d(3, c, 1, stride=s)
                                          for c, s in [(8, 8), (16, 16), (32, 32)]])

    def forward(self, pixels):
        return tuple(layer(pixels) for layer in self.layers)


def test_verified_resume_prefix_restores_history_without_reading_images():
    provider = TinyProvider()
    provider.begin_epoch(1)
    first = next(epoch_clips(provider.lengths, 1))
    provider.prepare(first, 'ir')
    prefix = prefix_checkpoint(provider)
    recovered = TinyProvider()
    result = restore_delivered_prefix(recovered, prefix)
    assert result['reread_images'] == 0
    assert prefix_checkpoint(recovered) == prefix
    assert recovered.used == provider.used
    assert next(recovered.expected_next) == next(provider.expected_next)
    corrupted = dict(prefix, actual_address_sha256='0'*64)
    with pytest.raises(ValueError, match='digest'):
        restore_delivered_prefix(TinyProvider(), corrupted)


def test_temporal_memory_must_belong_to_exact_last_delivered_frame():
    clip = list(epoch_clips(TinyProvider.lengths, 1))[0]
    value = torch.ones(1, 1, 2, 2)
    state = NativeTemporalState(clip.sequence, clip.frames[-1], (value,), (value,), (value.bool(),))
    gray = np.zeros((16, 16), np.uint8)
    validate_memory_at_prefix(state, gray, TinyProvider.lengths, 1, 1)
    wrong = NativeTemporalState(clip.sequence, clip.frames[-1]+1, (value,), (value,), (value.bool(),))
    with pytest.raises(ValueError, match='last delivered'):
        validate_memory_at_prefix(wrong, gray, TinyProvider.lengths, 1, 1)


def test_storage_cap_fails_before_larger_writes(tmp_path):
    (tmp_path / 'small').write_bytes(b'x'*100)
    assert guard_storage(tmp_path, cap_bytes=100, reserve_bytes=0)['work_file_bytes'] == 100
    with pytest.raises(RuntimeError):
        guard_storage(tmp_path, cap_bytes=99, reserve_bytes=0)


def test_actual_optimizer_restart_matches_uninterrupted_and_saves_no_rgb_weights(tmp_path):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    def initialize():
        torch.manual_seed(0)
        encoder = TinyEncoder().eval().requires_grad_(False)
        head = YOLOXHead(1, width=.125, in_channels=[64, 128, 256]).eval().requires_grad_(False)
        module = NativePyramidFusion(channels=(8, 16, 32), mode='temporal')
        optimizer = torch.optim.AdamW(module.parameters(), lr=.0003, weight_decay=.0001)
        return module, encoder, head, optimizer
    module, encoder, head, optimizer = initialize()
    config = dict(budget_and_Dev=dict(stages=[1, 2]),
        architecture=dict(temporal_parameters=sum(p.numel() for p in module.parameters())),
        training_input_hw=[64, 64], Train_delivery=dict(clips_per_epoch=3),
        fit=dict(learning_rate=.0003, weight_decay=.0001, lambda_reliable_RGB=.02,
                 warmup_updates=2, horizon_updates=12, minimum_lr_fraction=.03, gradient_clip_norm=5),
        storage=dict(work_cap_GiB=1, minimum_independent_filesystem_free_GiB=0))
    config_path = tmp_path / 'synthetic_engineering_config.json'
    config_path.write_text(json.dumps(config))
    metadata = dict(fit_config_sha256=sha256(config_path), seed=0, source_hashes={},
                    cue='ir', fit_id='synthetic_engineering_only')
    complete = tmp_path / 'uninterrupted'
    history_full = run_training_to_epoch(module, encoder, head, optimizer, TinyProvider(), config,
        metadata, config_path, complete, 2, tmp_path)
    expected_model = state_sha256(module.state_dict())
    expected_optimizer = optimizer.state_dict()
    restarted = tmp_path / 'restarted'
    module, encoder, head, optimizer = initialize()
    run_training_to_epoch(module, encoder, head, optimizer, TinyProvider(), config,
        metadata, config_path, restarted, 1, tmp_path)
    # New objects deliberately model a fresh process, including lost temporal state.
    module, encoder, head, optimizer = initialize()
    history_resumed = run_training_to_epoch(module, encoder, head, optimizer, TinyProvider(), config,
        metadata, config_path, restarted, 2, tmp_path)
    assert state_sha256(module.state_dict()) == expected_model
    assert history_resumed == history_full
    for key, values in optimizer.state_dict()['state'].items():
        for name, value in values.items():
            assert torch.equal(value.cpu(), expected_optimizer['state'][key][name].cpu())
    saved = torch.load(restarted / 'latest.pt', map_location='cpu', weights_only=False)
    assert saved['completed_epochs'] == 2 and saved['delivered_clips_total'] == 6
    assert saved['actual_optimizer_updates'] == 6 and saved['temporal_state'] is None
    assert 'encoder' not in saved and 'head' not in saved and 'RGB_model' not in saved
    assert set(saved['module']) == set(module.state_dict())
