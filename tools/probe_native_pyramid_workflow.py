#!/usr/bin/env python3
"""CPU workflow suite with immutable source hashes, before any formal GPU fit."""
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.launch_native_pyramid_study import source_files
from tools.native_pyramid_experiment import CANONICAL_PROOF, PIPELINE_REVISION, STUDY, context, read
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint = context()
    output = work / 'native_pyramid_CPU_workflow_tests.json'
    assert not output.exists(), 'Preserve previous CPU workflow evidence'
    tests = [str(p.relative_to(ROOT)) for p in ROOT.glob('tests/test_native_pyramid_*.py')]
    assert len(tests) == 8
    tests.append('tests/test_native_content_reference.py')
    sources = {name: sha256(ROOT / name) for name in source_files()}
    tests_hashes = {name: sha256(ROOT / name) for name in tests}
    for name in sources:
        if name.endswith('.py'): py_compile.compile(str(ROOT / name), doraise=True)
    started = time.monotonic()
    log = work / 'native_pyramid_CPU_workflow_tests.log'
    command = [sys.executable, '-m', 'pytest', '-q', *sorted(tests)]
    with log.open('x') as stream:
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    assert process.returncode == 0, str(log)
    assert all(sha256(ROOT / name) == digest for name, digest in sources.items())
    assert all(sha256(ROOT / name) == digest for name, digest in tests_hashes.items())
    text = log.read_text()
    assert '52 passed' in text and 'failed' not in text
    canonical = CANONICAL_PROOF
    proof = read(canonical)
    assert proof['all39_summary_exact'] and proof['complete_official_CSV_exact'] and proof['Dev_frames'] == 4834
    assert all(sha256(ROOT / name) == digest for name, digest in proof['source_hashes'].items())
    report = dict(status='COMPLETE_CPU_NATIVE_WORKFLOW_TESTS', tests_passed=52, pytest_command=command, pipeline_revision=PIPELINE_REVISION,
        tested_source_hashes=sources, test_source_hashes=tests_hashes, log_sha256=sha256(log),
        canonical_Dev_proof_sha256=sha256(canonical), study_sha256=sha256(STUDY),
        seconds=time.monotonic()-started, GPU_work=False, formal_fit_started=False,
        scope='CPU source versions locked around unit/integration suite and complete canonical Dev replay. Hashes do not claim complete line/branch coverage or production GPU validation.',
        pending_actual_GPU_canaries=True, input_content_guard_revision=1, goal_met=False, supported_innovations=0)
    dump(output, report)
    print(json.dumps(dict(status=report['status'], tests_passed=52, sources=len(sources), seconds=report['seconds'])), flush=True)


if __name__ == '__main__': main()
