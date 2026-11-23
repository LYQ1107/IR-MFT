"""Verify that the handoff cannot overlap the existing GPU2/3 jobs."""
from tools.perceptual_gpu_handoff import future_gpus, previous_gpu_work_complete


def _dev(bt='COMPLETE', sn='COMPLETE', status='RUNNING_CPU_REPLAYS'):
    return dict(status=status, producers={'BT-005': {'status': bt}, 'SN-013': {'status': sn}})


def test_future_pair_is_exactly_the_user_requested_two_devices():
    assert future_gpus() == ('4', '5')


def test_handoff_waits_for_both_current_producers():
    fits = {'status': 'COMPLETE'}
    assert not previous_gpu_work_complete(_dev(bt='RUNNING'), fits)
    assert not previous_gpu_work_complete(_dev(sn='RUNNING'), fits)
    assert not previous_gpu_work_complete({'status': 'RUNNING'}, fits)
    assert previous_gpu_work_complete(_dev(), fits)


def test_handoff_requires_completed_training_and_preserves_failures():
    assert not previous_gpu_work_complete(_dev(), {'status': 'RUNNING'})
    assert not previous_gpu_work_complete(_dev(), {})
    assert not previous_gpu_work_complete(_dev(status='FAILED'), {'status': 'COMPLETE'})
    assert previous_gpu_work_complete(_dev(status='COMPLETE'), {'status': 'COMPLETE'})
