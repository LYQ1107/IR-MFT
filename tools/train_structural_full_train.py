#!/usr/bin/env python3
"""Same S/C/R adapter and fit function, now with three complete Train traversals."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from tools.run_pseudo_ir_comparison import dump, sha256
from tools.structural_full_train_inputs import FullTrainInputs, FullTrainLoopBridge
from tools.structural_full_train_schedule import table_digest
from tools.structural_stream_training import initialize_encoder, initialize_head_adapter
from tools.structural_train_image_reader import TrainImageReader, with_image_reader
from yolox.models.structural_factorial_repair import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cue', choices=['rgb', 'gray', 'ir'], required=True)
    parser.add_argument('--components', choices=['111', '011', '101', '110'], required=True)
    parser.add_argument('--seed', type=int, choices=[0], required=True)
    parser.add_argument('--smoke-steps', type=int, choices=[0, 20], default=0)
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical in ['4', '5']
    plan_path = ROOT / 'configs/perceptual_mft_20261122.json'
    plan = json.loads(plan_path.read_text())
    work = Path(plan['work_root'])
    dependency = json.loads((work / 'structural_stream_optimizer_GPU_schedule.json').read_text())
    assert dependency['status'] == 'COMPLETE_TWO_STEP_LIVE_GPU_PARITY_NO_FORMAL_FIT'
    gpu_proof_path = work / 'structural_stream_optimizer_GPU/state.json'
    assert dependency['proof_sha256'] == sha256(gpu_proof_path)
    assert json.loads(gpu_proof_path.read_text())['status'] == 'COMPLETE_TWO_STEP_LIVE_GPU_OPTIMIZER_PARITY'
    study_path = ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json'
    study = json.loads(study_path.read_text())
    assert any((r['cue'], r['components'], r['seed']) == (args.cue, args.components, 0) for r in study['fits'])
    config_path = ROOT / study['configuration']
    config = json.loads(config_path.read_text())
    assert sha256(config_path) == study['configuration_sha256']
    assert sha256(plan_path) == study['main_plan_sha256']
    policy_path = ROOT / 'configs/perceptual_single_seed_policy_20261123.json'
    assert sha256(policy_path) == study['single_seed_policy_sha256']
    policy = json.loads(policy_path.read_text())
    assert policy['fixed_seed'] == 0 and not policy['additional_seed_fits']
    assert policy['structural_base_config_sha256'] == config['parent_structural_config_sha256']
    protocol_path = ROOT / plan['protocol']
    assert sha256(protocol_path) == study['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    schedule_path = work / 'structural_full_train_schedule/complete.json'
    schedule = json.loads(schedule_path.read_text())
    assert sha256(schedule_path) == study['schedule_proof_sha256']
    assert sha256(schedule['address_file']['path']) == schedule['address_file']['sha256']
    with np.load(schedule['address_file']['path']) as z:
        table = z['addresses']
    sequences = schedule['sequence_order']
    assert set(sequences) == set(protocol['train'])
    assert table_digest(sequences, table) == study['semantic_address_sha256']
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    parent_path = ROOT / 'configs/perceptual_correlation_repair_20261123.json'
    parent = json.loads(parent_path.read_text())
    assert sha256(parent_path) == config['parent_plan_sha256']
    checkpoint = Path(parent['source_RGB_checkpoint'])
    assert sha256(checkpoint) == parent['source_RGB_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    head, adapter = initialize_head_adapter(saved, config['architecture'], args.components, 'cuda')
    encoder = initialize_encoder(saved, ROOT / 'exps/yolox_mft26ir_visible.py', 'cuda')
    del saved
    calibration_path = work / 'correlation_cache/calibration.json'
    calibration = json.loads(calibration_path.read_text())
    assert calibration['config_sha256'] == sha256(parent_path)
    image_reader = TrainImageReader(plan['data_root'], plan['generated_ir_root'], sequences)
    input_class = with_image_reader(FullTrainInputs)
    provider = input_class(plan, protocol, config, calibration, dict.fromkeys(sequences), encoder, 'cuda',
                           lengths=schedule['lengths'], image_reader=image_reader)
    fit = config['fit']
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=fit['learning_rate'], weight_decay=fit['weight_decay'])
    out = work / 'structural_full_train_fit' / f'{args.cue}_{args.components}_s0'
    if args.smoke_steps:
        out = out / 'smoke_20'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'complete.json').exists() and not (out / 'failure.json').exists()
    names = ['tools/train_structural_full_train.py', 'tools/structural_full_train_inputs.py',
        'tools/structural_full_train_schedule.py', 'tools/structural_stream_loop.py',
        'tools/structural_stream_training.py', 'tools/structural_train_image_reader.py',
        'tools/train_structural_relative.py', 'tools/structural_grid.py',
        'yolox/models/structural_factorial_repair.py', 'yolox/models/structural_relative_repair.py',
        'yolox/models/structural_relative_losses.py', 'yolox/models/frozen_head_repair.py',
        'yolox/models/recoverable_curriculum.py']
    state = dict(status='FITTING_FULL_TRAIN_SAME_STRUCTURAL_ARCHITECTURE',
        cue=args.cue, variant=args.components, component_flags=FLAGS[args.components], seed=0,
        physical_gpu=physical, config_sha256=sha256(config_path), parent_sha256=sha256(parent_path),
        full_train_study_sha256=sha256(study_path), full_train_address_sha256=study['semantic_address_sha256'],
        schedule_proof_sha256=sha256(schedule_path), source_RGB_sha256=sha256(checkpoint),
        single_seed_user_policy_sha256=sha256(policy_path), source_hashes={name: sha256(ROOT / name) for name in names},
        GPU_input_optimizer_proof_sha256=sha256(gpu_proof_path),
        calibration_sha256=sha256(calibration_path), Train_GT_hashes=provider.gt_hashes,
        trainable_parameters=sum(p.numel() for p in adapter.parameters()),
        frozen_original_head=True, frozen_original_encoder=True, Train_videos=sequences,
        Train_frames=27251, logical_Train_epochs_registered=3, optimizer_macro_passes_registered=72,
        progress_epoch_unit='Optimizer macro pass;24macro passes per complete Train traversal.',
        updates_registered=20736, Dev_fit_rows=0, Test_read=False,
        no_image_or_FPN_disk_cache=True, geometry='Exact frozen currentRGB decoded geometry; semantic loss only.')
    assert state['trainable_parameters'] == study['trainable_parameters'] == 145600
    bridge = FullTrainLoopBridge(provider, sequences, state, table)
    dump(out / 'fitting_manifest.json', state)
    loop_args = copy.copy(args)
    loop_args.variant = 'no_cycle' if adapter.centered else 'no_centering'
    try:
        bridge.run(loop_args, adapter, head, optimizer, config, config_path, out)
        assert sha256(study_path) == state['full_train_study_sha256']
        complete = json.loads((out / 'complete.json').read_text())
        if not args.smoke_steps:
            assert complete['epochs'] == 72 and complete['successful_steps_this_invocation'] == 20736
            assert len(complete['live_frame_passes']) == 72
            assert all(r['complete_input_pass'] for r in complete['live_frame_passes'])
        complete.update(status='FULL_TRAIN_SMOKE20_VERIFIED' if args.smoke_steps else 'COMPLETE_THREE_FULL_TRAIN_TRAVERSALS',
            logical_Train_epochs_completed=0 if args.smoke_steps else 3,
            full_frame_address_delivery_verified=not bool(args.smoke_steps), post_loop_source_verified=True,
            image_reader_statistics=image_reader.statistics(), full_encoder_frozen_no_grad=all(
                not p.requires_grad and p.grad is None for p in encoder.parameters()))
        dump(out / 'full_train_complete.json', complete)
        print(json.dumps(dict(status=complete['status'], updates=complete['successful_steps_this_invocation'],
                              logical_Train_epochs=complete['logical_Train_epochs_completed'])), flush=True)
    except Exception as error:
        dump(out / 'failure.json', dict(status='FAILED_PRESERVE_FULL_TRAIN', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
