#!/usr/bin/env python3
"""Reproduce existing RGB complete Dev official metrics, not a new F4 result."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.fusion_memory_evaluation import official_csv47, run_official_tracks
from tools.fusion_memory_model_loading import DevQualifiedModels
from tools.fusion_memory_parent_io import ParentEvidencePending
from tools.neck72_runtime import require_host_memory, require_storage

SPEC = ROOT/'configs/perceptual_f4_official_evaluation_20261125.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pytest-receipt', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    started = time.monotonic()
    spec = read(SPEC)
    assert all(sha(ROOT/name) == digest for name, digest in spec['frozen_dependencies'].items())
    tests = read(args.pytest_receipt)
    assert tests['exit_code'] == 0 and sha(tests['log']) == tests['log_sha256']
    assert all(sha(ROOT/name) == digest for name, digest in tests['source_hashes'].items())
    match = re.search(r'\b(\d+) passed\b', Path(tests['log']).read_text())
    assert match and int(match.group(1)) == 10
    plan = read(ROOT/'configs/perceptual_mft_20261122.json')
    protocol = read(ROOT/plan['protocol'])
    work = Path(plan['work_root'])
    output, evaluation = work/'fusion_memory_official_evaluation_CPU_proof.json', work/'fusion_memory_existing_RGB_Dev_evaluation'
    assert not output.exists() and not evaluation.exists()
    require_host_memory(4*2**30, maximum_workers=1)
    require_storage(work)
    source = Path(spec['actual_CPU_scope']['source_RGB_Dev_directory'])
    metadata = source/'metrics.json'
    original_csv = source/'trackeval_output/pedestrian_detailed.csv'
    assert sha(metadata) == spec['actual_CPU_scope']['source_RGB_Dev_metrics_sha256']
    assert sha(original_csv) == spec['actual_CPU_scope']['source_RGB_Dev_CSV_sha256']
    original = read(metadata)
    assert original['protocol']['split'] == 'dev' and original['protocol']['variant'] == 'rgb'
    assert original['protocol']['seed'] == 0 and original['protocol']['frames'] == 4834
    assert original['protocol']['sequences'] == protocol['dev'] == ['BT-005', 'SN-013']
    assert not original['protocol']['max_frames']
    allowed_arrays = {(source/'detections'/(seq+'.npz')).resolve() for seq in protocol['dev']}
    allowed_gt = {(Path(plan['evaluation']['GT_root'])/'train'/seq/'gt/gt.txt').resolve() for seq in protocol['dev']}
    phase = dict(evaluation=False)
    attempts = dict(inference_GT_reads=0, forbidden_Test_or_array_or_image_reads=0, tensor_loads=0)
    evaluation_gt_reads = []
    def audit(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        mode = values[1] or ''
        writing = any(c in mode for c in 'wax')
        if path.name == 'gt.txt':
            if not phase['evaluation']:
                attempts['inference_GT_reads'] += 1
                raise AssertionError('GT read before completed inference-source verification')
            assert path in allowed_gt
            evaluation_gt_reads.append(str(path))
        if ('test' in path.parts or (path.suffix in ['.npy', '.npz'] and path not in allowed_arrays) or
                (not writing and path.suffix.lower() in ['.jpg', '.jpeg', '.png'])):
            attempts['forbidden_Test_or_array_or_image_reads'] += 1
            raise AssertionError('Existing complete Dev outputs only; no Test/images/newcache')
    sys.addaudithook(audit)
    def forbid_tensor(*a, **kw):
        attempts['tensor_loads'] += 1
        raise AssertionError('No neural tensor loading is part of metric reproduction')
    with patch.object(torch, 'load', forbid_tensor):
        try:
            DevQualifiedModels()
        except ParentEvidencePending as pending:
            missing = pending.paths
        else:
            raise AssertionError('Current parent file proof expects incomplete original Dev parents')
        rows = []
        for seq, frames in [('BT-005', 1834), ('SN-013', 3000)]:
            packet, track = source/'detections'/(seq+'.npz'), source/'track_results'/(seq+'.txt')
            with np.load(packet, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['frame_ids'], np.arange(1, frames+1))
                assert data['offsets'].shape == (frames+1,) and data['offsets'][0] == 0
                assert data['offsets'][-1] == len(data['boxes']) == len(data['scores'])
            rows.append(dict(sequence=seq, frames=frames, packet=str(packet), packet_sha256=sha(packet),
                track=str(track), track_sha256=sha(track)))
        assert sum(r['frames'] for r in rows) == 4834 and len(missing) == 6
        evaluation.mkdir()
        manifest = dict(status='EXISTING_COMPLETE_RGB_DEV_INFERENCE_OUTPUTS_VERIFIED_BEFORE_GT_EVALUATION',
            frames=4834, split='dev', original_RGB_metadata_sha256=sha(metadata), rows=rows,
            no_new_inference=True, no_F4_parent_qualification=True)
        manifest_path = evaluation/'inference_manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        phase['evaluation'] = True
        gt_hashes = {str(path): sha(path) for path in allowed_gt}
        assert all(gt_hashes[str((Path(plan['evaluation']['GT_root'])/'train'/seq/'gt/gt.txt').resolve())] ==
                   protocol['derived_gt_sha256']['train/'+seq] for seq in protocol['dev'])
        values = run_official_tracks(source/'track_results', evaluation, plan, 'dev')
    assert values['metrics39'] == original['trackeval'] and sha(values['official_CSV']) == sha(original_csv)
    previous = official_csv47(original_csv, values['official_schema'], 'dev', original['trackeval'])
    assert all(values[k] == previous[k] for k in ['combined47', 'per_sequence47', 'selection_metrics'])
    assert all(sha(r['track']) == r['track_sha256'] and sha(r['packet']) == r['packet_sha256'] for r in rows)
    assert all(sha(p) == digest for p, digest in gt_hashes.items()) and not any(attempts.values())
    assert not torch.cuda.is_initialized()
    # Only redundant copies made by the existing evaluator are removed.
    copied = evaluation/'trackeval_format'
    assert copied.parent == evaluation
    shutil.rmtree(copied)
    sources = {name: sha(ROOT/name) for name in ['tools/fusion_memory_evaluation.py', 'tools/fusion_memory_dev_freeze.py',
        'tools/probe_f4_official_evaluation.py', 'tests/test_fusion_memory_evaluation.py', str(SPEC.relative_to(ROOT))]}
    sources.update(spec['frozen_dependencies'])
    result = dict(status='COMPLETE_OFFICIAL47_COMPONENT_EXISTING_RGB_DEV_REPRODUCTION_ONLY',
        source_hashes=sources, specification_sha256=sha(SPEC), synthetic_tests_passed=10,
        pytest_receipt=str(args.pytest_receipt), pytest_receipt_sha256=sha(args.pytest_receipt),
        pytest_log_sha256=tests['log_sha256'], original_RGB_metadata_sha256=sha(metadata),
        original_RGB_CSV_sha256=sha(original_csv), reproduced_CSV=values['official_CSV'],
        reproduced_CSV_sha256=sha(values['official_CSV']), all39_summary_and_original276CSV_exact=True,
        all47_per_sequence_and_combined_exact=True, existing_RGB_Dev47=values['combined47'],
        existing_RGB_Dev_per_sequence47=values['per_sequence47'], official_metric_schema=values['official_schema'],
        input_rows=rows, inference_manifest=str(manifest_path), inference_manifest_sha256=sha(manifest_path),
        actual_GT_evaluation_read=True, evaluation_GT_file_sha256=gt_hashes, evaluation_GT_open_count=len(evaluation_gt_reads),
        guarded_attempt_counters=attempts, actual_Test_accuracy_read=False, actual_parent_status='PENDING_SIX_ORIGINAL_DEV_MARKERS',
        actual_fusion_tensor_loads=0, actual_learned_fusion_transferred=False, new_NN_inference=False, new_NN_fits=0,
        actual_new_official_evaluation=True, new_candidate_accuracy=False, baseline_reproduction_only=True,
        new_GPU_work=False, new_image_FPN_disk_cache=False, redundant_MOT_copies_removed=True,
        formal_F4_launch_ready=False, fixed_seed=0, additional_seed_work=False,
        seconds=time.monotonic()-started, peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        scope='Existing complete canonical RGB Dev4834-frame outputs only, not F4 or Test. GT is read only by separate evaluation after input verification. Reproduction is not a new accuracy improvement.',
        goal_met=False, supported_innovations=0)
    assert result['peak_RSS_GiB'] <= 4
    require_storage(work)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), seconds=result['seconds'], peak_RSS_GiB=result['peak_RSS_GiB'],
        existing_Dev_HOTA=values['selection_metrics']['HOTA'], new_candidate_accuracy=False)), flush=True)


if __name__ == '__main__':
    main()
