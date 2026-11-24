"""Backfill must never bypass unfinished GPU work or accept foreign predictions."""
import json

import pytest

from tools.structural_legacy_backfill import predecessor_ready, check_producer
from tools.run_pseudo_ir_comparison import sha256


def completed_predecessor():
    fits = dict(status='COMPLETE', jobs={str(i): dict(status='COMPLETE', pid=i + 100) for i in range(6)})
    dev = dict(status='COMPLETE', jobs={str(i): dict(status='COMPLETE') for i in range(50)})
    advance = dict(status='COMPLETE_FULL_TRAIN_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT')
    test = dict(status='COMPLETE', producers={str(i): dict(status='COMPLETE', pid=i + 200) for i in range(7)},
                jobs={'one': dict(status='COMPLETE')})
    return fits, dev, advance, test


def test_requires_actual_gpu_process_exit():
    states = completed_predecessor()
    assert predecessor_ready(*states, finished=lambda pid: True)
    assert not predecessor_ready(*states, finished=lambda pid: pid != 203)


@pytest.mark.parametrize('stage', [0, 1, 3])
def test_complete_label_cannot_hide_missing_registered_work(stage):
    states = completed_predecessor()
    key = 'producers' if stage == 3 else 'jobs'
    states[stage][key].pop('0')
    assert not predecessor_ready(*states, finished=lambda pid: True)


def test_does_not_start_before_official_full_train_export():
    states = completed_predecessor()
    states[2]['status'] = 'RUNNING_ALL_FROZEN_FULL_TRAIN_TEST'
    assert not predecessor_ready(*states, finished=lambda pid: True)


def test_failed_predecessor_preserved_instead_of_bypassed():
    states = completed_predecessor()
    states[1]['status'] = 'FAILED_PRESERVE_DEV'
    with pytest.raises(RuntimeError, match='do not steal GPUs'):
        predecessor_ready(*states, finished=lambda pid: True)


def test_prediction_bytes_are_checked_before_accuracy(tmp_path):
    packet = tmp_path / 'some_packet.npz'
    packet.write_bytes(b'completed-original-prediction')
    meta = dict(status='SMOKE_COMPLETE', frames=8, GT_read=False, future_frame_input=False, config_sha256='fixed-config',
                source_hashes={'math.py': 'fixed-source'}, fresh_RGB_parity=[dict(array_exact=True)] * 8,
                outputs={'one': dict(path=str(packet), sha256=sha256(packet))})
    (tmp_path / 'complete.json').write_text(json.dumps(meta))
    spec = dict(config_sha256='fixed-config', source_hashes={'math.py': 'fixed-source'}, source_zero_streams=1)
    check_producer(tmp_path, spec, 8, source=True)
    packet.write_bytes(b'different-prediction')
    with pytest.raises(AssertionError):
        check_producer(tmp_path, spec, 8, source=True)


def test_foreign_source_code_rejected_before_accuracy(tmp_path):
    meta = dict(status='SMOKE_COMPLETE', frames=8, GT_read=False, future_frame_input=False, config_sha256='fixed-config',
                source_hashes={'math.py': 'changed-source'}, outputs={'one': {}},
                fresh_RGB_parity=[dict(array_exact=True)] * 8)
    (tmp_path / 'complete.json').write_text(json.dumps(meta))
    spec = dict(config_sha256='fixed-config', source_hashes={'math.py': 'fixed-source'}, source_zero_streams=1)
    with pytest.raises(AssertionError):
        check_producer(tmp_path, spec, 8, source=True)
