from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.correlation_gate_observer import CorrelationGateObserver
from tools.correlation_recovery_inference import CorrelationCaseDetector, LegacyCaseTrackWriter
from tools.native_pyramid_stream_tracks import mot_line
from yolox.models.local_correlation_repair import CorrelationRepair
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker


def test_private_counters_reproduce_original_two_video_tracks(tmp_path):
    videos = {'A': 6, 'B': 4}
    ids = ['first', 'second']
    observations = {}
    for name in ids:
        for seq, frames in videos.items():
            arrays = []
            for frame in range(1, frames+1):
                rows = [[frame, 0, frame+24, 24, .9]]
                if frame <= 2 or (name == 'first' and frame >= 4):
                    rows.append([100+frame, 80, 124+frame, 104, .85])
                arrays.append(np.array(rows, np.float32))
            observations[name, seq] = arrays
    expected, old_count = {}, BaseTrack._count
    try:
        for name in ids:
            BaseTrack._count = 0
            for seq, frames in videos.items():
                tracker = BYTETracker(SimpleNamespace(track_thresh=.6, track_buffer=30,
                    match_thresh=.9, mot20=False), frame_rate=25)
                rows = []
                for frame, array in enumerate(observations[name, seq], 1):
                    rows += [(frame, t.tlwh.copy(), t.track_id, t.score)
                             for t in tracker.update(array.copy(), (1, 1), (1, 1))
                             if t.tlwh[2]*t.tlwh[3] > 100]
                expected[name, seq] = ''.join(map(mot_line, rows))
    finally:
        BaseTrack._count = old_count
    counters = {name: [0] for name in ids}
    for seq, frames in videos.items():
        writer = LegacyCaseTrackWriter(tmp_path, seq, ids, counters)
        for frame in range(1, frames+1):
            writer.append(frame, {name: observations[name, seq][frame-1] for name in ids})
        result = writer.close(frames)
        for name in ids:
            assert Path(result[name]['track_path']).read_text() == expected[name, seq]
    assert min(counters[name][0] for name in ids) > 2


def test_gate_observer_preserves_half_outputs_and_missing_has_no_gate_sample():
    torch.manual_seed(0)
    module = CorrelationRepair(channels=(4, 6, 8), latent=3, radius=1).eval()
    for layer in module.levels:
        nn.init.normal_(layer.output.weight, std=.2)
    rgb = [torch.randn(2, c, h, h).half() for c, h in [(4, 4), (6, 2), (8, 1)]]
    aux = [torch.randn_like(value) for value in rgb]
    score = [torch.full_like(value[:, :1], .2) for value in rgb]
    for value in score:
        value[0, :, 0, 0] = .8
    available = torch.tensor([True, False])
    with torch.no_grad():
        expected = module(rgb, aux, score, available)
    observer = CorrelationGateObserver()
    observer.attach('case', module)
    try:
        with torch.no_grad():
            observed = module(rgb, aux, score, available)
            missing = module(rgb, None, score, torch.zeros(2, dtype=torch.bool))
        assert all(torch.equal(a, b) for a, b in zip(expected, observed))
        assert all(torch.equal(a, b) for a, b in zip(rgb, missing))
        rows = observer.summary('case')
        assert all(row['frames'] == 4 and row['available_frames'] == 1 and
                   row['missing_frames'] == 3 for row in rows)
        assert all(row['learned_sigmoid_gate']['count'] == value.shape[-1]**2
                   for row, value in zip(rows, rgb))
        assert all(row['reliable_changed_feature_cells'] == 0 for row in rows)
        assert any(row['nonzero_feature_correction_fraction'] > 0 for row in rows)
    finally:
        observer.close()


class SmallDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def backbone(self, values):
        return tuple(F.adaptive_avg_pool2d(values.float(), size) for size in [4, 2, 1])

    def head(self, features):
        values = []
        for feature in features:
            batch, _, height, width = feature.shape
            probability = feature.mean(1).flatten(1).sigmoid()
            index = torch.arange(height*width, device=feature.device).float()
            box = torch.stack([index*7+10, index*5+10, torch.full_like(index, 12),
                               torch.full_like(index, 12)], -1).expand(batch, -1, -1)
            values.append(torch.cat([box, (.4+.5*probability)[..., None],
                                     torch.full_like(probability[..., None], .9)], -1))
        return torch.cat(values, 1)


def test_batch_neighbors_missing_pixels_and_canonical_priority_are_isolated():
    torch.manual_seed(0)
    cases = [dict(id=name, intervention=intervention,
        source=dict(cue='ir', output_policy=policy)) for name, intervention, policy in [
        ('raw', 'normal', 'raw'), ('priority', 'normal', 'priority'),
        ('roll', 'IR_image_spatial_roll', 'raw'), ('off', 'unavailable', 'raw')]]
    cases += [dict(id=kind, baseline=True, convention=kind+'_B4') for kind in ['fresh', 'canonical']]
    adapters = {row['id']: None if row.get('baseline') else
                CorrelationRepair(channels=(3, 3, 3), latent=3, radius=1).eval() for row in cases}
    for module in filter(lambda value: value is not None, adapters.values()):
        for layer in module.levels:
            nn.init.normal_(layer.output.weight, std=.05)
    calibration = dict(statistics={cue: dict(mean=100., std=30.) for cue in ['gray', 'ir']})
    model = SmallDetector().eval()
    engine = CorrelationCaseDetector(model, cases, adapters, (32, 48), calibration)
    rng = np.random.default_rng(0)
    rgb = [rng.integers(0, 256, (40, 60, 3), dtype=np.uint8) for _ in range(4)]
    ir = [rng.integers(0, 256, (40, 60), dtype=np.uint8) for _ in range(4)]
    canonical = [np.array([[200, 200, 220, 220, .99]], np.float32) for _ in range(4)]
    result, fresh = engine.batch(rgb, ir, canonical)
    for index in range(4):
        np.testing.assert_array_equal(result['off'][index], fresh[index])
        np.testing.assert_array_equal(result['canonical'][index], canonical[index])
        assert any(np.array_equal(value, canonical[index][0]) for value in result['priority'][index])
    changed_rgb, changed_ir = list(rgb), list(ir)
    changed_rgb[3], changed_ir[3] = 255-rgb[3], 255-ir[3]
    changed, _ = engine.batch(changed_rgb, changed_ir, canonical)
    for name in result:
        for index in range(3):
            np.testing.assert_array_equal(changed[name][index], result[name][index])
    off_cases = [row for row in cases if row['id'] in ['off', 'fresh']]
    off = CorrelationCaseDetector(model, off_cases,
        {r['id']: adapters[r['id']] for r in off_cases}, (32, 48), calibration)
    assert not off.needs_IR
    # There are no auxiliary pixels at all, rather than a zero image embedding.
    unavailable, source = off.batch(rgb[:1], None, canonical[:1])
    np.testing.assert_array_equal(unavailable['off'][0], source[0])
