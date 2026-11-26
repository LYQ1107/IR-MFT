#!/usr/bin/env python3
"""Wait for an original extended Dev decision, then run one pinned CPU audit.

Does not signal/restart the original controller or launch any GPU work. A
waiting-window timeout is an observation result, not an experiment failure.
Only the returned child exit status is an actual subprocess completion; the
waiter's own tool exit must be confirmed separately by the caller.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TOOL = ROOT / 'tools/audit_mft_native_extended_dev.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(pid):
    p = Path('/proc') / str(pid)
    raw = (p / 'cmdline').read_bytes()
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(PID=pid, start_ticks=int(fields[19]), state=fields[0],
        raw_argv_sha256=hashlib.sha256(raw).hexdigest(),
        actual_command=raw.replace(b'\0', b' ').decode().strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=(9, 12), required=True)
    parser.add_argument('--auditor-sha256', required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--controller-start-ticks', type=int, required=True)
    parser.add_argument('--controller-argv-sha256', required=True)
    parser.add_argument('--execution-prefix', type=Path, required=True)
    parser.add_argument('--window-seconds', type=int, default=43200)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.window_seconds <= 0:
        raise ValueError('Require explicitly CPU-only visibility and a positive observation window')
    if sha(TOOL) != args.auditor_sha256:
        raise ValueError('Pinned extended Dev auditor differs')
    from tools.audit_mft_native_extended_dev import fixed_inputs, paths, readiness, read

    _, _, work, _ = fixed_inputs()
    prefix = args.execution_prefix
    if prefix.parent.resolve() != work.resolve():
        raise ValueError('Execution receipts must stay in the registered private work directory')
    receipt = Path(str(prefix) + '.execution.json')
    log = Path(str(prefix) + '.child.log')
    report, _ = paths(args.stage)
    if receipt.exists() or log.exists() or report.exists():
        raise ValueError('Preserve an earlier waiter or publication; do not duplicate a launch')
    prior = work / f'native_pyramid_Dev_stage_{args.stage - 3:02d}.json'
    prior_value = read(prior)
    if (prior_value['status'] != 'COMPLETE_ALL_REGISTERED_DEV_AT_STAGE' or
            prior_value['Test_read_for_selection'] is not False or
            not prior_value['decision']['extend_all_six'] or
            prior_value['decision']['next_stage'] != args.stage):
        raise ValueError('No original complete-Dev authorization for this extended stage')
    source = Path(__file__).resolve()
    expected = dict(PID=args.controller_pid, start_ticks=args.controller_start_ticks,
                    raw_argv_sha256=args.controller_argv_sha256)
    record = dict(status='WAITING_FOR_ORIGINAL_COMPLETE_EXTENDED_DEV', stage=args.stage,
        waiter=pin(os.getpid()), waiter_source=str(source), waiter_source_sha256=sha(source),
        CPU_auditor=str(TOOL), CPU_auditor_sha256=args.auditor_sha256,
        original_controller_expected=expected, prior_original_authorization_sha256=sha(prior),
        self_reported_exit_code=None, actual_tool_exit_code=None, actual_child_exit_code=None,
        child=None, CUDA_VISIBLE_DEVICES='', GPU_work=False, Test_results_read=False,
        no_training_or_GPU_actions=True, observation_timeout_is_not_original_job_failure=True,
        goal_met=False)

    def save():
        record['UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = Path(str(prefix) + '.temporary_' + str(os.getpid()))
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        os.replace(temporary, receipt)

    started, emitted = time.monotonic(), 0
    save()
    try:
        while True:
            if sha(TOOL) != args.auditor_sha256 or sha(source) != record['waiter_source_sha256']:
                raise ValueError('Pinned CPU audit/waiter source changed')
            if sha(prior) != record['prior_original_authorization_sha256']:
                raise ValueError('Original preceding Dev authorization changed')
            ready = readiness(args.stage)[0]
            record.update(original_stage_readiness=ready, elapsed_wait_seconds=time.monotonic() - started)
            if ready['status'].startswith('READY_'):
                break
            actual = pin(args.controller_pid)
            if any(actual[k] != v for k, v in expected.items()) or actual['state'] in ('T', 't', 'Z'):
                raise ValueError('Original controller identity/live state no longer matches')
            record['original_controller_actual'] = actual
            if time.monotonic() - started >= args.window_seconds:
                record.update(status='OBSERVATION_WINDOW_ENDED_NOT_ORIGINAL_JOB_FAILURE', self_reported_exit_code=0)
                save()
                print(json.dumps(dict(status=record['status'], stage=args.stage, original_job_restarted=False)), flush=True)
                return
            save()
            if time.monotonic() - emitted >= 60:
                print(json.dumps(dict(status=record['status'], stage=args.stage,
                    original_phase=ready['controller_phase'], original_controller_identity_verified=True,
                    elapsed_wait_seconds=record['elapsed_wait_seconds'], GPU_work=False, goal_met=False)), flush=True)
                emitted = time.monotonic()
            time.sleep(15)
        command = [sys.executable, '-u', str(TOOL), '--stage', str(args.stage), '--publish']
        record.update(status='RUNNING_PINNED_CPU_EXTENDED_DEV_AUDIT', requested_child_command=command,
            original_stage_decision_sha256=sha(work / f'native_pyramid_Dev_stage_{args.stage:02d}.json'),
            child_log=str(log))
        child_started, peak = time.monotonic(), 0
        with log.open('xb', buffering=0) as output:
            process = subprocess.Popen(command, cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                         OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            save()
            while process.poll() is None:
                p = Path('/proc') / str(process.pid)
                try:
                    raw = (p / 'cmdline').read_bytes()
                    if record['child'] is None and raw == b'\0'.join(s.encode() for s in command) + b'\0':
                        record['child'] = dict(pin(process.pid), actual_argv_matches_requested=True)
                        save()
                    for line in (p / 'status').read_text().splitlines():
                        if line.startswith('VmRSS:'):
                            peak = max(peak, int(line.split()[1]) * 1024)
                except OSError:
                    pass
                time.sleep(.1)
        code = process.wait()
        record.update(actual_child_exit_code=code, child_wall_seconds=time.monotonic() - child_started,
                      child_peak_sampled_RSS_GiB=peak / 2**30, child_log_sha256=sha(log))
        if code:
            raise RuntimeError('Preserve actual CPU audit failure: ' + str(log))
        if record['child'] is None:
            raise ValueError('Actual CPU child identity not observed/matched')
        value = read(report)
        if (value['status'] != 'COMPLETE_INDEPENDENT_EXTENDED_FULL_DEV_AUDIT_NOT_FINAL_STUDY' or
                value['stage'] != args.stage or value['source_audit_tool_sha256'] != args.auditor_sha256):
            raise ValueError('Unexpected extended Dev audit output')
        record.update(status='COMPLETE_PINNED_CPU_EXTENDED_DEV_EXPORT_NOT_FINAL_STUDY',
                      report=str(report), report_sha256=sha(report), self_reported_exit_code=0)
        save()
        print(json.dumps(dict(status=record['status'], stage=args.stage, actual_child_exit_code=code,
                              report_sha256=record['report_sha256'], goal_met=False)), flush=True)
    except Exception as error:
        record.update(status='FAILED_PRESERVE_CPU_EXTENDED_DEV_WAITER', error=repr(error), self_reported_exit_code=1)
        save()
        raise


if __name__ == '__main__':
    main()
