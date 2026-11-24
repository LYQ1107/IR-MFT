from types import SimpleNamespace

import numpy as np
import pytest

from tools.structural_full_train_schedule import make_full_train_schedule, table_digest
from tools.structural_full_train_inputs import FullTrainInputs, FullTrainLoopBridge


LENGTHS = {'BT-001': 3000, 'BT-003': 243, 'MSK-002': 1754,
           'PF-001': 15000, 'SN-001': 684, 'SN-015': 6570}


def test_every_real_Train_frame_is_seen_in_each_of_three_logical_epochs():
    sequences, table, summaries = make_full_train_schedule(LENGTHS)
    assert table.shape == (72, 288, 4, 2)
    assert len(summaries) == 3
    for row in summaries:
        assert row['unique_Train_frames'] == 27251
        assert row['sample_exposures'] == 27648 and row['repeated_exposures'] == 397
        assert all(row['coverage'][s]['unique_frames'] == n for s, n in LENGTHS.items())
    paired = table.reshape(-1, 2, 2)
    assert np.all(paired[:, 0, 0] == paired[:, 1, 0])
    assert np.all(paired[:, 0, 1] != paired[:, 1, 1])
    s2, t2, _ = make_full_train_schedule(dict(reversed(list(LENGTHS.items()))))
    assert table_digest(sequences, table) == table_digest(s2, t2)
    with pytest.raises(ValueError, match='Budget'):
        make_full_train_schedule(LENGTHS, macro_passes=1)
    with pytest.raises(ValueError, match='seed0'):
        make_full_train_schedule(LENGTHS, seed=1)


def test_input_addresses_reject_heldout_videos_and_frame_overflow():
    inputs = object.__new__(FullTrainInputs)
    inputs.lengths, inputs.config = LENGTHS, dict(epochs=72)
    assert inputs.addresses([('PF-001', 15000)], 72) == [('PF-001', 15000)]
    for draw, epoch in [([('PF-001', 15001)], 1), ([('BT-005', 1)], 1), ([('SN-001', 0)], 1),
                        ([('BT-001', 1)], 73)]:
        with pytest.raises(ValueError):
            inputs.addresses(draw, epoch)


def test_private_loop_delivers_registered_full_Frame_macro_pass():
    sequences, table, _ = make_full_train_schedule(LENGTHS)
    class Provider:
        schedule = dict.fromkeys(sequences)
        config = dict(epochs=72)
        lengths = LENGTHS
        prepared = 0
        def addresses(self, draw, epoch):
            return draw
        def prepare(self, draw, cue, epoch):
            self.prepared += 1
            return SimpleNamespace(rgb=draw, auxiliary=cue, target=epoch, structure=(draw, cue, epoch), frames=draw)
    metadata = {}
    bridge = FullTrainLoopBridge(Provider(), sequences, metadata, table)
    for draw in bridge.batches(sequences, bridge.banks, 0, 1):
        result = bridge.tensors(bridge.banks, draw, 'ir')
        assert result == bridge.structural_tensors(bridge.structure, draw, 'ir')
    assert metadata['live_frame_passes'][0]['complete_input_pass']
    assert metadata['live_frame_passes'][0]['prepared_batches'] == 288
    bad = table.copy()
    bad[0, 0, 1] = bad[0, 0, 0]
    with pytest.raises(ValueError, match='distinct frames'):
        FullTrainLoopBridge(Provider(), sequences, {}, bad)
