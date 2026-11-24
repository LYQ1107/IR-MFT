#!/usr/bin/env python3
"""Joint-disable factorial cell on actual ordered Train pixels, CPU only.

Does not initialize CUDA, launch a formal fit or authorize a Test candidate.
"""
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch

from tools.fusion_factorial_rules import CELLS, module_kwargs
from tools.native_pyramid_content_guard import PinnedTrainImageReader
from tools.native_pyramid_experiment import STUDY, context, read
from tools.native_pyramid_fit import optimizer_step
from tools.native_pyramid_train_data import OrderedTrainPixels, epoch_clips
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.exp import get_exp
from yolox.models.frozen_head_repair import head_predictions
from yolox.models.native_pyramid_fusion import NativePyramidFusion

SPEC = ROOT / 'configs/perceptual_native_contrast_quality_factorial_20261124.json'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    plan, study, protocol, calibration, work, checkpoint = context()
    spec = read(SPEC)
    assert spec['parent_study_sha256'] == sha256(STUDY)
    assert spec['status'] == 'PREREGISTERED_FACTORIAL_ATTRIBUTION_EXTENSION_NOT_GPU_LAUNCH_READY'
    assert not (work / 'native_pyramid_normal_selection_frozen.json').exists()
    output = work / 'native_contrast_quality_factorial_CPU_proof.json'
    assert not output.exists(), 'Keep prior canary evidence'
    require_host_memory(8*2**30, maximum_workers=1); require_storage(work)
    parent_path = work / 'native_pyramid_pipeline_registered.json'
    parent = read(parent_path)
    sources = dict(parent['source_hashes'])
    sources.update({name: sha256(ROOT/name) for name in [str(SPEC.relative_to(ROOT)),
        'tools/fusion_factorial_rules.py', 'tools/probe_fusion_factorial.py',
        'tests/test_fusion_factorial_rules.py']})
    assert all(sha256(ROOT/name) == digest for name, digest in sources.items())
    parent_digest = sha256(parent_path)
    started = time.monotonic()
    test_log = work / 'native_contrast_quality_factorial_CPU_pytest.log'
    test_command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_fusion_factorial_rules.py']
    with test_log.open('x') as stream:
        subprocess.run(test_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    count = re.search(r'\b(\d+) passed\b', test_log.read_text())
    assert count and int(count.group(1)) == 16
    clips = list(epoch_clips(study['Train_delivery']['lengths'], 1))[:2]
    assert len(clips) == 2 and clips[0].sequence == clips[1].sequence
    assert clips[0].frames == (1, 2, 3, 4) and clips[1].frames == (5, 6, 7, 8)
    expected = {}
    for clip in clips:
        for frame in clip.frames:
            for modality, data_root, folder, suffix in [
                    ('rgb', plan['data_root'], 'visible', 'jpg'),
                    ('ir', plan['generated_ir_root'], 'infrared_diffv2ir', 'png')]:
                path = Path(data_root)/'train'/clip.sequence/folder/f'{frame:06d}.{suffix}'
                expected[clip.sequence, frame, modality] = dict(sha256=sha256(path), bytes=path.stat().st_size)
    reader = PinnedTrainImageReader(plan['data_root'], plan['generated_ir_root'], protocol['train'],
        expected_content=expected, max_bytes=256*2**20, reserve_bytes=40*2**30)
    teacher = get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'), None).get_model().float().eval().requires_grad_(False)
    assert sha256(checkpoint) == study['source_RGB_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    teacher.load_state_dict(saved['model'], strict=True); del saved
    frozen = state_sha256(teacher.state_dict())
    records = []
    for mode in study['modes']:
        initial_hashes, parameter_counts = {}, {}
        for cell in CELLS:
            torch.manual_seed(0)
            candidate = NativePyramidFusion(**module_kwargs(study, mode, cell)).float()
            initial_hashes[cell] = state_sha256(candidate.state_dict())
            parameter_counts[cell] = sum(p.numel() for p in candidate.parameters())
            del candidate
        assert len(set(initial_hashes.values())) == len(set(parameter_counts.values())) == 1
        assert parameter_counts['00'] == study['architecture'][mode+'_parameters']
        torch.manual_seed(0)
        module = NativePyramidFusion(**module_kwargs(study, mode, '00')).float()
        optimizer = torch.optim.AdamW(module.parameters(), lr=study['fit']['learning_rate'],
                                     weight_decay=study['fit']['weight_decay'])
        provider = OrderedTrainPixels(plan, protocol, study['Train_delivery']['lengths'], calibration,
            reader, study['training_input_hw'], 'cpu')
        provider.begin_epoch(1)
        state, gray, rows = None, None, []
        for position, clip in enumerate(clips):
            batch = provider.prepare(clip, 'ir')
            if mode == 'temporal' and position:
                assert state is not None and state.frame == 4
                assert all(not v.requires_grad for v in state.latent+state.covariance)
            row, state, gray = optimizer_step(module, teacher.backbone, teacher.head,
                batch, optimizer, study, position, state, gray)
            rows.append(row)
        assert rows[0]['initial_full_RGB_exact'] and rows[0]['native_stem_gradient_L1'] == 0
        assert rows[1]['native_stem_gradient_L1'] > 0 and rows[1]['decoded_box_max_change'] > 0
        assert all(row['optimizer_updated'] for row in rows)
        assert all(p.grad is None for p in module.quality.parameters()), 'Off retains capacity but bypasses gate MLPs'
        if mode == 'temporal': assert state.sequence == clips[0].sequence and state.frame == 8
        assert state_sha256(module.state_dict()) != initial_hashes['00']
        with torch.no_grad():
            features = teacher.backbone(batch.rgb.float())
            reference, metadata = head_predictions(teacher.head, features)
            missing, empty = [], None
            for index, frame in enumerate(batch.address.frames):
                value, empty, _ = module(tuple(v[index:index+1] for v in features), None,
                    tuple(v[index:index+1] for v in metadata['scores']), available=False,
                    state=empty, sequence=batch.address.sequence, frame=frame)
                assert empty is None
                missing.append(value)
            prediction, _ = head_predictions(teacher.head,
                tuple(torch.cat([v[level] for v in missing]) for level in range(3)))
            assert torch.equal(prediction, reference)
        assert state_sha256(teacher.state_dict()) == frozen and all(p.grad is None for p in teacher.parameters())
        records.append(dict(mode=mode, cell='00', actual_Train_sequence=clips[0].sequence,
            actual_Train_frames=list(range(1,9)), delivered_clips=provider.prepared,
            actual_delivered_prefix_sha256=provider.digest.hexdigest(),
            four_cells_identical_initial_tensor_state=initial_hashes,
            four_cells_equal_parameter_inventory=parameter_counts, optimizer_rows=rows,
            gate_MLP_parameters_retained_but_unused=True,
            trained_missing_full_FP32_head_exact_RGB=True, original_RGB_parameters_and_BN_unchanged=True,
            temporal_state_continued_with_detached_history=mode == 'temporal'))
        print(json.dumps(dict(mode=mode, cell='00', two_actual_Train_updates_passed=True,
                              missing_full_FP32_head_exact=True)), flush=True)
        del module, optimizer, provider, batch, features, reference, metadata, prediction, missing
    assert len(reader.content_hashes) == 16
    assert all(sha256(ROOT/name) == digest for name, digest in sources.items())
    assert sha256(parent_path) == parent_digest and not torch.cuda.is_initialized()
    require_storage(work)
    peak_RSS = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30
    assert peak_RSS <= 8, 'CPU canary exceeded its admitted per-process RAM bound'
    result = dict(status='COMPLETE_CPU_JOINT_CONTRAST_QUALITY_DISABLE_ENGINEERING',
        specification_sha256=sha256(SPEC), parent_native_registry_sha256=parent_digest,
        source_hashes=sources, tests_passed=int(count.group(1)), test_command=test_command,
        test_log_sha256=sha256(test_log), records=records,
        actual_Train_input_content=reader.content_manifest(), image_reader_statistics=reader.statistics(),
        unique_Train_RGB_frames=8, unique_current_November_IR_frames=8,
        total_Train_frame_conditions=16, total_actual_optimizer_updates=4,
        device='cpu', GPU_work=False, CUDA_initialized=False, weights_saved=0,
        formal_fits_started=0, Dev_rows=0, Test_rows=0, new_HOTA=False,
        parent_pipeline_unchanged=True, complete_GPU_Dev_Test_pipeline_implemented=False,
        formal_followup_launch_ready=False, fixed_seed=0, additional_seed_work=False,
        seconds=time.monotonic()-started,
        peak_process_RSS_GiB=peak_RSS,
        scope='Only two joint-disable CPU canaries, using8actual ordered Train frames in each mode. '
              'Four cells share initial tensors/parameter inventory; off gate parameters are unused. '
              'No complete Train fit, production FP16/B1/GPU identity, Dev/Test accuracy, '
              'new architecture or contribution is established. Original queued jobs are unchanged.',
        goal_met=False,supported_innovations=0)
    dump(output,result)
    print(result['status'],result['tests_passed'],'checks;',result['seconds'],'seconds',flush=True)


if __name__ == '__main__':
    main()
