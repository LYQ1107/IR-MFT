#!/usr/bin/env python3
"""Sequential CPU reproduction of the eight frozen CGR original Dev sources."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.native_common_test_backfill import parity
from tools.run_pseudo_ir_comparison import dump, sha256


def read(path):
    return json.loads(Path(path).read_text())


def main():
    spec_path = ROOT / 'configs/perceptual_correlation_recovery_backfill_20261124.json'
    spec = read(spec_path)
    assert spec['fixed_seed'] == 0 and not spec['additional_seed_work']
    assert sha256(ROOT / 'tools/register_correlation_recovery_backfill.py') == spec['source_sha256']
    assert all(sha256(p) == h for p, h in spec['checkpoint_sources'].items())
    assert all(sha256(p) == h for p, h in spec['artifact_sources'].items())
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    protocol = read(ROOT / plan['protocol'])
    work = Path(plan['work_root'])
    state_path = work / 'correlation_recovery_original_dev_reproduction_state.json'
    lock = state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not state_path.exists(), 'Preserve prior run instead of silently restarting'
    report_path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_CORRELATION_RECOVERY_DEV_REPRODUCTION.json'
    assert not report_path.exists()
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in [
        Path(__file__), ROOT / 'tools/native_common_test_backfill.py',
        ROOT / 'tools/replay_recoverable_correlation.py', ROOT / 'tools/eval_mft26ir.py',
        ROOT / 'yolox/tracker/byte_tracker.py']}
    cases = [row for row in spec['cases'] if row.get('source') and row['intervention'] == 'normal']
    assert len(cases) == 8
    jobs = {row['id']: dict(status='PENDING') for row in cases}
    state = dict(status='REPRODUCING_ALL8_FROZEN_CGR_ORIGINAL_DEV', controller_pid=os.getpid(),
        specification_sha256=sha256(spec_path), maximum_CPU_workers=1, GPU_work=False,
        fixed_seed=0, additional_seed_work=False, Test_accuracy_read=False,
        jobs=jobs, source_hashes=sources)
    dump(state_path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
    try:
        for row in cases:
            assert shutil.disk_usage(work).free >= 40 * 2**30
            used = int(subprocess.check_output(['du', '-s', '-B1', str(work)], text=True).split()[0])
            assert used < 64 * 2**30
            original = Path(row['source']['source_Dev_metrics'])
            assert sha256(original) == row['source']['source_Dev_metrics_sha256']
            manifest = read(original)['manifest']
            assert sha256(ROOT / 'tools/replay_recoverable_correlation.py') == manifest['replay_source_sha256']
            configs = [p for p in (ROOT / 'configs').rglob(row['source']['source_Dev_case'] + '.json')
                       if sha256(p) == manifest['config_sha256']]
            assert len(configs) == 1
            output = work / 'correlation_recovery_original_dev_reproduction' / row['id']
            assert not output.exists()
            log = work / (row['id'] + '.original_CGR_Dev_reproduction.log')
            command = [sys.executable, '-u', 'tools/replay_recoverable_correlation.py',
                '--config', str(configs[0]), '--output', str(output), '--split', 'dev']
            with log.open('xb', buffering=0) as stream:
                worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                jobs[row['id']].update(status='RUNNING', pid=worker.pid, log=str(log),
                                      command=command, original_config_sha256=sha256(configs[0]))
                dump(state_path, state)
                code = worker.wait()
            jobs[row['id']]['exit_code'] = code
            assert code == 0, str(log)
            proof = parity(output, original.parent, protocol['dev'])
            jobs[row['id']].update(status='COMPLETE', source_proof=proof)
            dump(state_path, state)
            print(json.dumps(dict(case=row['id'], all39metrics_tracks_official_CSV_exact=True)), flush=True)
        assert all(sha256(ROOT / p) == h for p, h in sources.items())
        assert sha256(spec_path) == state['specification_sha256']
        state['status'] = 'COMPLETE_ALL8_FROZEN_CGR_ORIGINAL_DEV_EXACT'
        state['UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump(state_path, state)
        dump(report_path, dict(state, scope='All8frozen original normal/paired CGR Dev sources reproduced '
            'on2videos/4834frames each. All39summary fields, bothcomplete tracks and entire officialCSV '
            'are exact. Cached CPU-only replay; not a new model or full7Test evaluation. '
            'GPU4/5 inference and remaining engineering still pending.',
            goal_met=False, supported_innovations=0))
        print(state['status'], flush=True)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_CGR_DEV_REPRODUCTION', error=repr(error))
        dump(state_path, state)
        raise


if __name__ == '__main__':
    main()
