#!/usr/bin/env python3
"""Nine matched spatial variants on six real Train frames, CPU checks only.

This is neither a formal full-Train fit nor a Dev/Test accuracy experiment.
The shared frozen RGB feature values live only in bounded process RAM.
"""
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from tools.content_verified_train_reader import ContentVerifiedTrainImageReader
from tools.native_pyramid_train_data import preprocess_observation
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.spatial_native_fusion import SpatialNativeFusion
from yolox.exp import get_exp
from yolox.models.frozen_head_repair import head_predictions, original_supervised_loss, preserve_reference


SPEC = ROOT/'configs/perceptual_native_spatial_prototype_20261124.json'


def gradient(module, fragment):
    return sum(float(p.grad.abs().sum()) for name, p in module.named_parameters()
               if fragment in name and p.grad is not None)


def registrations(work):
    paths = [work/'structural_full_train_pipeline_registered.json',
        work/'structural_content_guard_recovery_registered.json',
        work/'native_pyramid_pipeline_registered.json',
        ROOT/'configs/perceptual_correlation_recovery_workflow_20261124.json',
        work/'neck72_worker_interfaces_registered.json', work/'neck72_study_registered.json',
        work/'neck72_fresh_runtime_registered.json', work/'fusion_factorial_pipeline_registered.json']
    sources, rows = {}, []
    for path in paths:
        data = json.loads(path.read_text())
        for name, digest in data['source_hashes'].items():
            assert sha256(ROOT/name) == digest, (path, name)
            assert name not in sources or sources[name] == digest
            sources[name] = digest
        rows.append(dict(path=str(path), sha256=sha256(path), sources=len(data['source_hashes'])))
    return sources, rows


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    specification = json.loads(SPEC.read_text())
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    assert sha256(plan_path) == specification['main_plan_sha256']
    plan = json.loads(plan_path.read_text())
    protocol_path = ROOT/plan['protocol']
    assert sha256(protocol_path) == specification['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    assert sha256(ROOT/'yolox/models/native_pyramid_fusion.py') == specification['parent_source_sha256']
    native_path = ROOT/'configs/perceptual_native_pyramid_study_20261124.json'
    assert sha256(native_path) == specification['parent_native_study_sha256']
    native = json.loads(native_path.read_text())
    work = Path(plan['work_root'])
    require_host_memory(8*2**30, maximum_workers=1)
    require_storage(work)
    output = work/'native_spatial_CPU_proof.json'
    assert not output.exists(), 'Keep prior evidence immutable'
    test_log = work/'native_spatial_pytest.log'
    match = re.search(r'\b(\d+) passed\b', test_log.read_text())
    assert match and int(match.group(1)) == 10
    sources, frozen_registrations = registrations(work)
    names = ['tools/spatial_native_fusion.py', 'tools/probe_spatial_native_train.py',
        'tests/test_spatial_native_fusion.py', str(SPEC.relative_to(ROOT))]
    sources.update({name: sha256(ROOT/name) for name in names})
    cpu = specification['CPU']
    assert cpu['seed'] == 0 and len(cpu['addresses']) == 6
    assert {a['sequence'] for a in cpu['addresses']} == set(protocol['train'])
    calibration_path = work/'correlation_cache/calibration.json'
    assert sha256(calibration_path) == native['calibration_sha256']
    calibration = json.loads(calibration_path.read_text())
    assert calibration['Dev_rows'] == calibration['Test_rows'] == 0
    parent = json.loads((ROOT/'configs/perceptual_correlation_repair_20261123.json').read_text())
    checkpoint = Path(parent['source_RGB_checkpoint'])
    assert sha256(checkpoint) == native['source_RGB_sha256']
    started = time.monotonic()
    torch.manual_seed(0)
    teacher = get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'), None).get_model().float().eval().requires_grad_(False)
    original = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    teacher.load_state_dict(original, strict=True)
    reader = ContentVerifiedTrainImageReader(plan['data_root'], plan['generated_ir_root'],
        protocol['train'], max_bytes=64*2**20, reserve_bytes=40*2**30)
    cached, gt_hashes = [], {}
    for address in cpu['addresses']:
        sequence, frame = address['sequence'], address['frame']
        rgb, ir = reader.read(sequence, frame, 'rgb'), reader.read(sequence, frame, 'ir')
        gt_path = Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        assert sha256(gt_path) == protocol['derived_gt_sha256']['train/'+sequence]
        gt_hashes[sequence] = sha256(gt_path)
        gt = np.loadtxt(gt_path, delimiter=',', ndmin=2)
        gt = gt[(gt[:, 0] == frame) & (gt[:, 6] > 0) & (gt[:, 7] == 1)]
        pixels, available = {}, {}
        for cue in specification['matched_cues']:
            value, auxiliary, valid, _, ratio, present = preprocess_observation(
                rgb, ir if cue == 'ir' else None, cue, cpu['input_hw'], calibration)
            pixels[cue] = torch.from_numpy(auxiliary[None].copy())
            available[cue] = present
        assert all(available.values()), 'Use the fixed available engineering addresses; do not silently substitute'
        targets = torch.zeros(1, 500, 5)
        boxes = gt[:, 2:6].astype(np.float32).copy()
        boxes[:, :2] += boxes[:, 2:]/2
        boxes *= ratio
        assert 0 < len(boxes) <= 500
        targets[0, :len(boxes), 1:] = torch.from_numpy(boxes)
        with torch.no_grad():
            features = tuple(v.detach() for v in teacher.backbone(torch.from_numpy(value[None].astype(np.float32))))
            reference, metadata = head_predictions(teacher.head, features)
        rolled = np.roll(ir, (ir.shape[0]//4, ir.shape[1]//4), (0, 1))
        roll_pixels = preprocess_observation(rgb, rolled, 'ir', cpu['input_hw'], calibration)[1]
        cached.append(dict(address=address, features=features, reference=reference, metadata=metadata,
            pixels=pixels, valid=torch.from_numpy(valid[None].copy()), targets=targets,
            IR_roll=torch.from_numpy(roll_pixels[None].copy()), targets_count=len(boxes)))
        print(json.dumps(dict(prepared_Train=sequence, frame=frame, targets=len(boxes))), flush=True)
    rows, initial_states = [], set()
    for variant in specification['variants']:
        for cue in specification['matched_cues']:
            torch.manual_seed(0)
            kwargs = {k: variant[k] for k in ['use_quality', 'use_spatial', 'use_redundancy']}
            module = SpatialNativeFusion(**kwargs).float().train()
            assert sum(p.numel() for p in module.parameters()) == specification['architecture']['expected_parameters']
            initial = state_sha256(module.state_dict())
            initial_states.add(initial)
            optimizer = torch.optim.AdamW(module.parameters(), lr=cpu['learning_rate'], weight_decay=cpu['weight_decay'])
            steps = []
            for position, data in enumerate(cached):
                features, metadata = data['features'], data['metadata']
                repaired, state, _ = module(features, data['pixels'][cue], metadata['scores'], valid=data['valid'])
                assert state is None
                predictions, _ = head_predictions(teacher.head, repaired)
                if position == 0:
                    assert torch.equal(predictions, data['reference']), 'Initial full head must equal RGB'
                original_loss = original_supervised_loss(teacher.head, predictions, metadata,
                    data['targets'], cpu['input_hw'])
                preservation = preserve_reference(predictions, data['reference'], metadata['strides'], reliable_only=True)
                loss = original_loss[0]+cpu['lambda_reliable_RGB']*preservation
                assert bool(torch.isfinite(loss)) and loss.requires_grad
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = {fragment: gradient(module, fragment) for fragment in ['encoder.blocks.0.0',
                    'output.', 'spatial.4', 'redundancy.2', 'rgb_projection', 'aux_projection']}
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cpu['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                steps.append(dict(step=position+1, **data['address'], loss=float(loss.detach()),
                    full_YOLOX=float(original_loss[0].detach()), box=float(original_loss[1].detach()),
                    objectness=float(original_loss[2].detach()), classification=float(original_loss[3].detach()),
                    reliable_RGB=float(preservation.detach()), gradient_norm=float(norm), gradients_L1=gradients,
                    decoded_box_max_change=float((predictions[:, :, :4]-data['reference'][:, :, :4]).detach().abs().max())))
            assert steps[0]['gradients_L1']['output.'] > 0 and steps[0]['gradients_L1']['encoder.blocks.0.0'] == 0
            assert any(s['gradients_L1']['encoder.blocks.0.0'] > 0 for s in steps[1:])
            assert any(s['gradients_L1']['spatial.4'] > 0 for s in steps[1:])
            assert any(s['gradients_L1']['rgb_projection'] > 0 and s['gradients_L1']['aux_projection'] > 0 for s in steps[2:])
            if variant['use_redundancy']:
                assert any(s['gradients_L1']['redundancy.2'] > 0 for s in steps[1:])
            assert any(s['decoded_box_max_change'] > 0 for s in steps[1:])
            module.eval().requires_grad_(False)
            data = cached[0]
            with torch.no_grad():
                learned, _, diagnostics = module(data['features'], data['pixels'][cue], data['metadata']['scores'], valid=data['valid'])
                normal, _ = head_predictions(teacher.head, learned)
                missing, _, _ = module(data['features'], None, data['metadata']['scores'], available=False)
                unavailable, _ = head_predictions(teacher.head, missing)
                assert torch.equal(unavailable, data['reference'])
                joined = tuple(torch.cat([d['features'][level] for d in cached[:2]]) for level in range(3))
                batch_reference, batch_meta = head_predictions(teacher.head, joined)
                mixed, _, _ = module(joined, torch.cat([d['pixels'][cue] for d in cached[:2]]),
                    batch_meta['scores'], valid=torch.cat([d['valid'] for d in cached[:2]]), available=[True, False])
                mixed_head, _ = head_predictions(teacher.head, mixed)
                assert torch.equal(mixed_head[1], batch_reference[1])
                spatial_std = [float(row['gate'][row['valid']].std()) for row in diagnostics['levels']]
                assert all(value > 0 for value in spatial_std)
                roll_change = None
                if cue == 'ir':
                    wrong, _, _ = module(data['features'], data['IR_roll'], data['metadata']['scores'], valid=data['valid'])
                    wrong_head, _ = head_predictions(teacher.head, wrong)
                    roll_change = float((wrong_head-normal).abs().max())
                    assert roll_change > 0
            rows.append(dict(variant=variant['name'], cue=cue, parameters=sum(p.numel() for p in module.parameters()),
                initial_state_sha256=initial, optimizer_updates=len(steps), steps=steps,
                initial_full_head_RGB_exact=True, learned_unavailable_full_head_RGB_exact=True,
                partial_availability_missing_row_full_head_RGB_exact=True, learned_gate_spatial_std=spatial_std,
                actual_IR_image_roll_decoded_max_change=roll_change))
            print(json.dumps(dict(variant=variant['name'], cue=cue, updates=len(steps),
                full_head_missing_exact=True, gradients_verified=True)), flush=True)
            del module, optimizer
    assert len(initial_states) == 1 and len(rows) == 9
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert all(torch.equal(value, original[name].to(value.dtype)) for name, value in teacher.state_dict().items())
    assert sources == {name: sha256(ROOT/name) for name in sources}
    assert frozen_registrations == registrations(work)[1]
    assert not torch.cuda.is_initialized()
    require_storage(work)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30
    assert peak <= 8
    report = dict(status='COMPLETE_SIX_TRAIN_DOMAIN_NINE_SPATIAL_VARIANT_CUE_CPU_ENGINEERING_ONLY',
        specification_sha256=sha256(SPEC), source_hashes=sources, frozen_registrations=frozen_registrations,
        tests_passed=int(match.group(1)), test_log_sha256=sha256(test_log),
        initial_failed_test_collection=dict(reason='First shell omitted the supplementary dependency paths; loguru import failed',
            tests_executed=0, model_source_change_needed=False),
        rows=rows, source_image_content=reader.content_manifest(), Train_GT_sha256=gt_hashes,
        source_RGB_sha256=sha256(checkpoint), calibration_sha256=sha256(calibration_path),
        image_reader_statistics=reader.statistics(), unique_actual_Train_frames=6,
        actual_Train_domain_count=6, optimizer_updates_all_groups=54, fixed_seed=0,
        additional_seed_work=False, GPU_work=False, RGB_encoder_head_BN_state_and_gradients_unchanged=True,
        Dev_rows=0, Test_rows=0, new_accuracy_evaluation=False, formal_fit_started=False,
        saved_weights=False, image_or_FPN_disk_cache=False, peak_process_RSS_GiB=peak,
        seconds=time.monotonic()-started, scope='Six fixed Train frames and54engineering updates total, not a complete Train epoch, formal accuracy, new Test result, four-direction completion or independent innovation.',
        formal_pipeline_launch_ready=False, full_current_IR_four_directions_complete=False,
        goal_met=False, supported_innovations=0)
    dump(output, report)
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'], peak_RSS_GiB=peak)), flush=True)


if __name__ == '__main__':
    main()
