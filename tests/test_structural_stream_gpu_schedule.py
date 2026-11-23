import json

import pytest

from tools.probe_structural_stream_optimizer_gpu import dependency_ready


def stages(work, producers):
    for name in ['structural_factorial_fit_state.json', 'structural_factorial_dev_state.json',
                 'structural_factorial_runtime_schedule.json']:
        (work / name).write_text(json.dumps(dict(status='COMPLETE')))
    (work / 'structural_factorial_test_state.json').write_text(json.dumps(
        dict(status='RUNNING_REPLAYS', producers=producers)))


def test_diagnostic_gpu_cannot_start_while_any_registered_sequence_still_runs(tmp_path):
    assert not dependency_ready(tmp_path)[0]
    producers = {f'Sequence{i}': dict(status='COMPLETE') for i in range(7)}
    producers['Sequence6']['status'] = 'FULL_RUNNING'
    stages(tmp_path, producers)
    assert not dependency_ready(tmp_path)[0]
    producers['Sequence6']['status'] = 'COMPLETE'
    stages(tmp_path, producers)
    assert dependency_ready(tmp_path)[0]


def test_failure_is_preserved_instead_of_bypassing_dependencies(tmp_path):
    stages(tmp_path, {f'Sequence{i}': dict(status='COMPLETE') for i in range(7)})
    (tmp_path / 'structural_factorial_dev_state.json').write_text(json.dumps(dict(status='FAILED')))
    with pytest.raises(RuntimeError, match='Preserve failed'):
        dependency_ready(tmp_path)
