"""Physical GPU handoff authorized by the user; no CUDA initialization."""
import json
from pathlib import Path

POLICY_PATH = Path(__file__).resolve().parents[1]/'configs/perceptual_gpu_handoff_20261123.json'


def future_gpus():
    policy = json.loads(POLICY_PATH.read_text())
    devices = tuple(policy['future_physical_gpus'])
    assert devices == ('4', '5')
    assert policy['maximum_concurrent_physical_gpus'] == 2
    return devices


def previous_gpu_work_complete(dev, fits):
    return (bool(dev.get('status')) and 'FAIL' not in dev['status'] and
            fits.get('status') == 'COMPLETE' and
            all(dev.get('producers', {}).get(seq, {}).get('status') == 'COMPLETE'
                for seq in ['BT-005', 'SN-013']))
