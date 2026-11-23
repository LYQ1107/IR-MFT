"""The real legacy loop's input bindings are private and frame synchronous."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools import train_structural_relative as legacy
from tools.structural_stream_loop import StreamingLoopBridge


class Provider:
    def __init__(self):
        self.schedule = {f'Train{i}': np.arange(1, 24 * 192 + 1).reshape(24, 192)
                         for i in range(6)}
        self.prepared = 0

    def addresses(self, draw, epoch):
        return [(seq, int(self.schedule[seq][epoch - 1, slot])) for seq, slot in draw]

    def prepare(self, draw, cue, epoch):
        frames = self.addresses(draw, epoch)
        self.prepared += 1
        return SimpleNamespace(rgb=frames, auxiliary=cue, target=epoch,
                               structure=(frames, cue, epoch), frames=frames)


def test_full_pass_records_actual_addresses_and_preserves_legacy_globals():
    provider = Provider()
    before = {name: legacy.run_fit.__globals__[name]
              for name in ['batches', 'tensors', 'structural_tensors']}
    state = {}
    bridge = StreamingLoopBridge(provider, list(provider.schedule), state)
    function = bridge.bind()
    assert function.__code__ is legacy.run_fit.__code__
    assert function.__globals__ is not legacy.run_fit.__globals__
    assert all(legacy.run_fit.__globals__[k] is v for k, v in before.items())
    for epoch in [1, 2]:
        for draw in bridge.batches(bridge.sequences, bridge.banks, 0, epoch):
            rgb, cue, target = bridge.tensors(bridge.banks, draw, 'ir')
            descriptors = bridge.structural_tensors(bridge.structure, draw, 'ir')
            assert (rgb, cue, target) == descriptors
            assert all((epoch - 1) * 192 < frame <= epoch * 192 for seq, frame in rgb)
    assert len(state['live_frame_passes']) == 2
    assert all(r['complete_input_pass'] and r['prepared_batches'] == 288
               for r in state['live_frame_passes'])
    assert state['live_frame_passes'][0]['actual_frame_draw_sha256'] != state['live_frame_passes'][1]['actual_frame_draw_sha256']


def test_wrong_draw_descriptors_and_unconsumed_batch_fail():
    provider = Provider()
    bridge = StreamingLoopBridge(provider, list(provider.schedule), {})
    draws = bridge.batches(bridge.sequences, bridge.banks, 0, 1)
    draw = next(draws)
    bridge.tensors(bridge.banks, draw, 'gray')
    with pytest.raises(RuntimeError, match='identical draw'):
        bridge.structural_tensors(bridge.structure, draw[::-1], 'gray')
    with pytest.raises(RuntimeError, match='not consumed'):
        next(draws)
    assert not bridge.metadata['live_frame_passes'][0]['complete_input_pass']


def test_diagnostic_generator_close_records_only_delivered_frames():
    provider = Provider()
    bridge = StreamingLoopBridge(provider, list(provider.schedule), {})
    draws = bridge.batches(bridge.sequences, bridge.banks, 0, 1)
    for _ in range(2):
        draw = next(draws)
        bridge.tensors(bridge.banks, draw, 'rgb')
        bridge.structural_tensors(bridge.structure, draw, 'rgb')
    draws.close()
    row = bridge.metadata['live_frame_passes'][0]
    assert row['prepared_batches'] == 2 and not row['complete_input_pass']
    assert sum(row['unique_frames'].values()) == 8
