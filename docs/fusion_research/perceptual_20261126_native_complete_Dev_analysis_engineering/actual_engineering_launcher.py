#!/usr/bin/python3
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = Path('/data3/liuyeqiang/IR-MFT')
WORK = Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
revision = sys.argv[1]
assert re.fullmatch(r'r[0-9]+', revision)
base = WORK / ('native_complete_Dev_analysis_CPU_engineering_20261126_' + revision)
base.mkdir(exist_ok=False)
script = ROOT / 'tools/analyze_mft_native_complete_dev.py'
source_hash = hashlib.sha256(script.read_bytes()).hexdigest()
compile(script.read_bytes(), str(script), 'exec')
source = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json'
expected = '63426a8689f27ce662557b2df1d737db1d55d18af7e716bdd1374bdc290d688e'
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
executions = []


def run(label, stage):
    output = base / label
    log = base / (label + '.stdout.txt')
    command = ['/usr/bin/python3', '-u', str(script), '--dev-report', str(source),
               '--expect-report-sha256', expected, '--stage', str(stage),
               '--output-directory', str(output)]
    started = time.monotonic()
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
        identity = None
        wanted = b''.join(part.encode() + b'\0' for part in command)
        for _ in range(1000):
            try:
                folder = Path('/proc') / str(child.pid)
                raw = (folder / 'cmdline').read_bytes()
                fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
                if raw == wanted:
                    identity = dict(PID=child.pid, start_ticks=int(fields[19]),
                                    raw_argv_sha256=hashlib.sha256(raw).hexdigest())
                    break
            except FileNotFoundError:
                pass
            if child.poll() is not None:
                break
            time.sleep(.001)
        code = child.wait()
    assert identity is not None
    row = dict(label=label, command=command, identity=identity, exit_code=code,
               wall_seconds=time.monotonic() - started,
               log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
               output_directory_exists=output.exists(), source_sha256=source_hash,
               CUDA_VISIBLE_DEVICES='', GPU_work=False)
    (base / (label + '.execution.json')).write_text(json.dumps(row, indent=2) + '\n')
    executions.append(row)
    return output, log, code


output, log, code = run('reference6', 6)
assert code == 0, log.read_text()
actual = json.loads((output / 'complete.json').read_text())
old = json.loads((ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV_ANALYSIS_ZH.json').read_text())
checks = {}
for key in ['all_six_own_registered_Dev_best', 'descriptive_strongest_controls_across_modes',
            'full_normal_curves', 'original_frozen_budget_decision']:
    assert actual[key] == old[key], key
    checks[key] = True
for key in ['IR_own_best_comparisons', 'per_sequence_comparisons']:
    stripped = [{k: v for k, v in row.items() if k != 'delta47'} for row in actual[key]]
    assert stripped == old[key], key
    checks[key] = True
assert actual['same_epoch_architecture_cue_interactions'] == old['all_six_same_epoch_interactions']
checks['all6_same_epoch_interactions'] = True
assert all(set(row['delta47']) == set(actual['official_fields'])
           for key in ['IR_own_best_comparisons', 'per_sequence_comparisons']
           for row in actual[key])
negative, negative_log, negative_code = run('reject_reference6_as_stage9', 9)
assert negative_code != 0 and not negative.exists()
assert 'Only an actually completed matching full Dev stage is accepted' in negative_log.read_text()
assert hashlib.sha256(script.read_bytes()).hexdigest() == source_hash
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
result = dict(
    status='COMPLETE_CPU_REFERENCE6_DESCRIPTIVE_ANALYSIS_REGRESSION_AND_INCOMPLETE_STAGE9_REJECTION_NOT_NEW_ACCURACY',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_analysis_tool=str(script.relative_to(ROOT)), source_analysis_tool_sha256=source_hash,
    actual_source_report=str(source.relative_to(ROOT)), actual_source_report_sha256=expected,
    actual_source_stage=6, actual_source_cases=62,
    reference6_original_published_sections_exact=checks, all47_deltas_present=True,
    actual_combined_comparisons=len(actual['IR_own_best_comparisons']),
    actual_per_sequence_comparisons=len(actual['per_sequence_comparisons']),
    actual_complete_normal_curves=len(actual['full_normal_curves']),
    actual_curve_points=sum(len(row['points']) for row in actual['full_normal_curves']),
    actual_stage9_execution=False, incomplete_stage9_rejected_before_any_outputs=True,
    actual_executions=executions, original_NN_cases_or_source_modified=False,
    Test_results_read=False, GPU_work=False, new_HOTA=False,
    supported_innovations=0, goal_met=False, actual_parent_tool_exit_code=None)
(base / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items()
                  if k not in ['actual_executions', 'reference6_original_published_sections_exact']}))
