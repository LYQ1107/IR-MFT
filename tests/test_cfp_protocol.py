"""Guard data leakage, immutable provenance and online MOT file contracts."""
import importlib.util
import json
from pathlib import Path

import pytest

from cfp.provenance import config_hash, record_run


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_data():
    return {'videos': [{'id': 1, 'file_name': 'a'}, {'id': 2, 'file_name': 'b'}],
            'images': [{'id': 1, 'video_id': 1, 'frame_id': 1}, {'id': 2, 'video_id': 2, 'frame_id': 1}],
            'annotations': [{'image_id': 1, 'track_id': 1}, {'image_id': 2, 'track_id': 1}]}


def test_composite_identity_and_video_isolation():
    check = module('cfp_build_protocol').check_partition
    assert check(tiny_data(), ['a'], ['b']) == {'train': 1, 'dev': 1}
    with pytest.raises(ValueError):
        check(tiny_data(), ['a', 'b'], ['b'])
    with pytest.raises(ValueError):
        check(tiny_data(), ['a'], [])


def test_causal_annotation_order():
    data = tiny_data()
    data['images'][0]['frame_id'] = 2
    with pytest.raises(AssertionError):
        module('cfp_build_protocol').check_partition(data, ['a'], ['b'])


@pytest.mark.parametrize('content', [
    '1,1,0,0,10,10,1\n1,1,1,1,10,10,1\n',
    '2,1,0,0,10,10,1\n1,1,0,0,10,10,1\n',
    '1,1,nan,0,10,10,1\n', '1,1,0,0,0,10,1\n',
    '3,1,0,0,10,10,1\n', '1,0,0,0,10,10,1\n',
    '1.5,1,0,0,10,10,1\n',
])
def test_invalid_online_outputs_rejected(tmp_path, content):
    p = tmp_path / 'seq.txt'
    p.write_text(content)
    with pytest.raises(ValueError):
        module('cfp_evaluate').validate_mot(p, 2)


def test_valid_and_empty_mot(tmp_path):
    p = tmp_path / 'seq.txt'
    p.write_text('1,1,0,0,10,10,.9\n2,1,2,3,10,10,.8\n')
    assert module('cfp_evaluate').validate_mot(p, 2) == 2
    p.write_text('')
    assert module('cfp_evaluate').validate_mot(p, 2) == 0


def test_registry_cannot_silently_overwrite_or_record_nan(tmp_path):
    record = dict(run_id='run_0', hypothesis='baseline', git_commit='abc', seed=0,
                  config={'threshold': .5}, detector_sha256='hash', reid_sha256=None,
                  split='dev', epochs=0, checkpoint_selection='none', metrics=None,
                  log='log', output='out', conclusion=None, status='RUNNING')
    p = record_run(tmp_path, record)
    assert json.loads(p.read_text())['config_sha256'] == config_hash(record['config'])
    with pytest.raises(FileExistsError):
        record_run(tmp_path, record)
    with pytest.raises(ValueError):
        record_run(tmp_path, dict(record, run_id='../unsafe'))
    with pytest.raises(ValueError):
        config_hash({'metric': float('nan')})


def test_frozen_real_protocol_has_no_video_overlap():
    protocol = json.loads((ROOT / 'configs/cfp/protocol_v1.json').read_text())
    assert not set(protocol['train']) & set(protocol['dev'])
    assert not (set(protocol['train']) | set(protocol['dev'])) & set(protocol['historical_test'])
    assert sum(protocol['frame_counts'].values()) == 32085
    crossfit = protocol['crossfit_groups']
    assert not set(crossfit[0]) & set(crossfit[1])
    assert set(crossfit[0]) | set(crossfit[1]) == set(protocol['train'])
