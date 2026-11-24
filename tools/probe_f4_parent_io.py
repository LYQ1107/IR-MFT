#!/usr/bin/env python3
"""CPU-only receipt of actual F4 parent files; pending is not ineligibility."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fusion_memory_parent_io import DevParentFiles, ParentEvidencePending, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run this receipt with CUDA_VISIBLE_DEVICES empty')
    started = time.monotonic()
    attempts = dict(forbidden_GT_or_Test_accuracy_reads=0)
    def audit(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if DevParentFiles.is_test_accuracy(path) or path.name == 'targets.npy':
            attempts['forbidden_GT_or_Test_accuracy_reads'] += 1
            raise ValueError('GT/Test accuracy cannot enter the F4 parent receipt')
    sys.addaudithook(audit)
    reader = DevParentFiles()
    output = args.output or reader.work/'fusion_memory_parent_IO_actual_state.json'
    require(not output.exists(), 'Preserve an existing receipt; choose a new output path')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    require(int(memory['MemAvailable'].split()[0])*1024 >= 42*2**30, 'Preserve 40GiB host RAM plus the2GiB reader budget')
    total = sum(p.stat().st_size for p in reader.work.rglob('*') if p.is_file() and not p.is_symlink())
    free = shutil.disk_usage(reader.work).free
    require(total <= 64*2**30 and free >= 40*2**30, 'Preserve registered storage limits')
    try:
        evidence = reader.load()
        state = dict(status='COMPLETE_PARENT_DEV_FILE_VERIFICATION_NO_EXECUTION', evidence=evidence,
            actual_parent_Dev_files_available=True,
            actual_parent_eligible_in_verified_Dev_files=evidence['selection']['chosen_family'] is not None)
    except ParentEvidencePending as pending:
        state = dict(status='PENDING_ORIGINAL_COMPLETE_PARENT_DEV_FILES', missing_markers=pending.paths,
            actual_parent_Dev_files_available=False, actual_parent_eligible_in_verified_Dev_files=None,
            eligibility_unknown_not_negative=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    require(rss <= 2, 'Parent file reader exceeded its2GiB budget')
    sources = ['tools/fusion_memory_parent_io.py', 'tools/probe_f4_parent_io.py',
        'tools/fusion_memory_design.py', 'configs/perceptual_f4_parent_io_20261125.json',
        'configs/perceptual_fusion_memory_study_20261125.json', 'tests/test_fusion_memory_parent_io.py']
    state.update(UTC=datetime.now(timezone.utc).isoformat(), source_hashes={name: reader.hash(ROOT/name) for name in sources},
        actual_verified_files={str(p): value for p, value in reader.digests.items()},
        forbidden_read_attempts=attempts, seconds=time.monotonic()-started, peak_RSS_GiB=rss,
        work_GiB=total/2**30, independent_free_GiB=free/2**30,
        new_GPU_work=False, new_NN_fits=0, new_image_FPN_cache=False, new_accuracy=False,
        checkpoint_tensor_loading_performed=False, learned_fusion_transfer=False,
        full_F4_pipeline_ready=False, formal_launch_ready=False, goal_met=False, supported_innovations=0,
        scope='Byte hashes and original Dev file lineage only; strict tensor loading, real GPU engineering, executor, release, evaluation and cost remain required.')
    with output.open('x') as stream:
        json.dump(state, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), status=state['status'], seconds=state['seconds'],
        peak_RSS_GiB=rss, new_accuracy=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
