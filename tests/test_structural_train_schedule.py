import numpy as np
import pytest
from tools.structural_train_schedule import make_schedule


def test_first_pass_reproduces_old_frames_and_all_later_passes_are_distinct():
    original = {'short':[1,3], 'long':[2,9]}
    schedule = make_schedule({'short':3,'long':11},original,passes=4,samples=2)
    for key, rows in schedule.items():
        np.testing.assert_array_equal(rows[0],original[key])
        assert all(len(set(map(int,row))) == 2 for row in rows)
    assert len(np.unique(schedule['short'])) == 3
    assert len(np.unique(schedule['long'])) == 8


def test_schedule_is_shared_across_cues_and_stable_to_dictionary_order():
    a = make_schedule({'a':5,'b':7},{'a':[1,5],'b':[2,7]},passes=3,samples=2)
    b = make_schedule({'b':7,'a':5},{'b':[2,7],'a':[1,5]},passes=3,samples=2)
    for key in a:
        np.testing.assert_array_equal(a[key],b[key])


def test_rejects_extra_seeds_and_invalid_cached_frame_addresses():
    with pytest.raises(ValueError):
        make_schedule({'a':4},{'a':[1,4]},samples=2,seed=1)
    for first in [[1,1],[0,3],[1,5]]:
        with pytest.raises(ValueError):
            make_schedule({'a':4},{'a':first},samples=2)
