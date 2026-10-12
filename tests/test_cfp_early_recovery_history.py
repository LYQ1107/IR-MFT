import numpy as np
import pytest

from cfp.history import early_recent_history
from cfp.lifecycle_crossover import load_bot_reference, make_cfp_lifecycle


def test_three_early_cues_are_retained_with_unique_sorted_eight_token_budget():
    early = [(1, 'wrong'), (2, 'a'), (3, 'b')]
    recent = [(i, str(i)) for i in range(10, 18)]
    selected = early_recent_history(recent, early)
    assert [row[0] for row in selected] == [1, 2, 3, 13, 14, 15, 16, 17]
    assert selected[0][1] == 'wrong'  # Retention does not assert correctness.
    assert early_recent_history(early, early) == early
    with pytest.raises(ValueError):
        early_recent_history([(0, None)], [])


def test_early_sidecar_is_updated_after_writes_and_reset_per_tracker():
    native = load_bot_reference('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT')
    tracker = make_cfp_lifecycle(native, 'gru', memory_mode='fifo', read_view='early3_recent5')
    box = np.array([[0, 0, 30, 20, .9]], np.float32)
    feature = np.array([[1., 0.]], np.float32)
    for frame in range(1, 12):
        tracker.update(box, feature, np.array([frame]))
    track = tracker.tracked_stracks[0]
    assert [row[0] for row in track.early_identity_history] == [1, 2, 3]
    assert len(track.identity_history) == 8 and track.last_identity_frame == 11
    fresh = make_cfp_lifecycle(native, 'gru', memory_mode='fifo', read_view='early3_recent5')
    assert fresh.frame_id == 0 and not fresh.tracked_stracks
