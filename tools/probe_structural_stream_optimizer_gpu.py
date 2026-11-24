#!/usr/bin/env python3
"""Two live-GPU Train updates versus the unchanged cached training function."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from tools import train_structural_relative as legacy
from tools.probe_structural_stream_optimizer import CachedSlices, assert_nested_equal
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.structural_stream_loop import StreamingLoopBridge
from tools.structural_stream_training import StreamingTrainInputs, initialize_encoder, initialize_head_adapter
from tools.structural_train_image_reader import TrainImageReader, with_image_reader
from yolox.models.structural_nocycle_refinement import NoCycleRefinement
from yolox.models.yolo_head import YOLOXHead


def dependency_ready(work):
    """No GPU work until the already registered complete combination Test is produced."""
    names = ['structural_factorial_fit_state.json', 'structural_factorial_dev_state.json',
             'structural_factorial_runtime_schedule.json', 'structural_factorial_test_state.json']
    states = {}
    for name in names:
        path = work / name
        if not path.exists():
            return False, states
        data = json.loads(path.read_text())
        states[name] = data['status']
        if 'FAIL' in data['status']:
            raise RuntimeError('Preserve failed combination stage before future GPU work')
        if name.endswith('_test_state.json'):
            producers = data.get('producers', {})
            if len(producers) != 7 or not all(v['status'] == 'COMPLETE' for v in producers.values()):
                return False, states
        elif not data['status'].startswith('COMPLETE'):
            return False, states
    return True, states


def main():
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical in ['4', '5']
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    ready, dependency = dependency_ready(work)
    assert ready, 'Do not interfere with the registered GPU4/5 pipeline'
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    config_path = ROOT / 'configs/perceptual_structural_relative_20261123.json'
    config = json.loads(config_path.read_text())
    parent_path = ROOT / 'configs/perceptual_correlation_repair_20261123.json'
    parent = json.loads(parent_path.read_text())
    out = work / 'structural_stream_optimizer_GPU'
    out.mkdir(exist_ok=False)
    names = ['tools/probe_structural_stream_optimizer_gpu.py', 'tools/probe_structural_stream_optimizer.py',
             'tools/structural_stream_training.py', 'tools/structural_stream_loop.py',
             'tools/structural_train_image_reader.py', 'tools/train_structural_relative.py',
             'tools/train_correlation_repair.py', 'tools/structural_grid.py',
             'yolox/models/structural_factorial_repair.py', 'yolox/models/structural_nocycle_refinement.py',
             'yolox/models/structural_relative_repair.py', 'yolox/models/frozen_head_repair.py',
             'yolox/models/structural_relative_losses.py', 'yolox/models/recoverable_curriculum.py']
    state = dict(status='REGISTERED_TWO_STEP_LIVE_GPU_OPTIMIZER_PARITY', controller_pid=os.getpid(),
                 physical_gpu=physical, dependencies=dependency, fixed_seed=0, engineering_steps=2,
                 source_hashes={name: sha256(ROOT / name) for name in names},
                 config_sha256=sha256(config_path), no_formal_training=True,
                 diagnostic_deterministic_algorithms=True, cudnn_deterministic=True,
                 CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                 Dev_fit_rows=0, Test_accuracy_or_labels_read=False)
    dump(out / 'state.json', state)
    started = time.perf_counter()
    try:
        checkpoint = Path(parent['source_RGB_checkpoint'])
        assert sha256(checkpoint) == parent['source_RGB_sha256']
        cache = CachedSlices(work, protocol, sha256(parent_path))
        with np.load(work / 'structural_stream_schedule/frame_addresses.npz') as z:
            schedule = {name: z[name] for name in z.files}
        for seq in protocol['train']:
            np.testing.assert_array_equal(schedule[seq][0], cache.metadata[seq]['frame_ids'])
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        a = config['architecture']
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        reference_head = YOLOXHead(1, width=1.25)
        reference_head.load_state_dict({k.removeprefix('head.'): v for k, v in saved['model'].items()
                                       if k.startswith('head.')}, strict=True)
        reference_head = reference_head.cuda().float().eval().requires_grad_(False)
        reference = NoCycleRefinement(latent=a['latent_channels'], radius=a['local_radius'],
            temperature=a['structural_temperature'], bound=a['feature_residual_bound_RMS_fraction'],
            risk_threshold=a['teacher_risk_threshold'], refinement='no_cycle',
            tolerance=a['forward_backward_tolerance_cells']).cuda()
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        head, adapter = initialize_head_adapter(saved, a, '111', 'cuda')
        encoder = initialize_encoder(saved, ROOT / 'exps/yolox_mft26ir_visible.py', 'cuda')
        del saved
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert_nested_equal(dict(head.state_dict()), dict(reference_head.state_dict()))
        assert_nested_equal(dict(adapter.state_dict()), dict(reference.state_dict()))
        image_reader = TrainImageReader(plan['data_root'], plan['generated_ir_root'], protocol['train'])
        calibration = json.loads((work / 'correlation_cache/calibration.json').read_text())
        cls = with_image_reader(StreamingTrainInputs)
        provider = cls(plan, protocol, config, calibration, schedule, encoder, 'cuda', image_reader=image_reader)
        bridge_state = copy.deepcopy(state)
        bridge = StreamingLoopBridge(provider, protocol['train'], bridge_state)
        actual_frames = []
        original_prepare = provider.prepare

        def checked_prepare(draw, cue, epoch):
            batch = original_prepare(draw, cue, epoch)
            expected = cache.batch(draw, cue)
            for x, y in zip(batch.rgb + batch.auxiliary + batch.structure + (batch.target,),
                            expected.rgb + expected.auxiliary + expected.structure + (expected.target,)):
                assert torch.equal(x, y.cuda()), 'Live B1 Train inputs differ from cached reference'
            actual_frames.extend(batch.frames)
            return batch

        provider.prepare = checked_prepare
        reference_namespace = dict(legacy.run_fit.__globals__)
        reference_namespace['tensors'] = lambda banks, draw, cue: (
            tuple(v.cuda() for v in cache.batch(draw, cue).rgb),
            tuple(v.cuda() for v in cache.batch(draw, cue).auxiliary), cache.batch(draw, cue).target.cuda())
        reference_namespace['structural_tensors'] = lambda banks, draw, cue: tuple(
            v.cuda() for v in cache.batch(draw, cue).structure)
        reference_loop = types.FunctionType(legacy.run_fit.__code__, reference_namespace, legacy.run_fit.__name__)
        assert reference_loop.__code__ is bridge.bind().__code__ is legacy.run_fit.__code__
        fit = config['fit']
        make_optimizer = lambda model: torch.optim.AdamW(model.parameters(), lr=fit['learning_rate'],
                                                        weight_decay=fit['weight_decay'])
        optimizer, ref_optimizer = make_optimizer(adapter), make_optimizer(reference)
        args = SimpleNamespace(cue='ir', variant='no_cycle', seed=0, smoke_steps=2)
        ref_out, live_out = out / 'cached_reference', out / 'live_inputs'
        ref_out.mkdir()
        live_out.mkdir()
        reference_loop(args, reference, reference_head, ref_optimizer, bridge.banks, {}, protocol['train'],
                       config, config_path, ref_out, copy.deepcopy(state))
        bridge.run(args, adapter, head, optimizer, config, config_path, live_out)
        reference_payload = torch.load(ref_out / 'latest.pt', map_location='cpu', weights_only=False)
        payload = torch.load(live_out / 'latest.pt', map_location='cpu', weights_only=False)
        assert_nested_equal(payload['adapter'], reference_payload['adapter'])
        assert_nested_equal(payload['optimizer'], reference_payload['optimizer'])
        assert_nested_equal(payload['history'], reference_payload['history'])
        assert_nested_equal(dict(head.state_dict()), dict(reference_head.state_dict()))
        assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
        assert all(sha256(ROOT / name) == digest for name, digest in state['source_hashes'].items())
        mappings = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                           '--format=csv,noheader,nounits'], text=True).splitlines()
        own = [line for line in mappings if line.split(',')[1].strip() == str(os.getpid())]
        expected_uuid = {'4': 'GPU-daa9b388-4540-242c-83d7-3261bb232a7c',
                         '5': 'GPU-f07a0798-7ddc-ec7c-a27c-d47124bbfe81'}[physical]
        assert len(own) == 1 and own[0].split(',')[0].strip() == expected_uuid
        state.update(status='COMPLETE_TWO_STEP_LIVE_GPU_OPTIMIZER_PARITY', seconds=time.perf_counter()-started,
            actual_Train_frames=actual_frames, unique_Train_frames=len(set(actual_frames)),
            actual_live_B1_features_descriptors_targets_exact=True, initialization_and_CPU_RNG_exact=True,
            updated9adapter_tensors_exact=True, AdamW_moments_steps_groups_exact=True,
            full_two_step_loss_history_exact=True, unchanged_legacy_loop_bytecode=True,
            frozen_head_exact=True, frozen_encoder_without_gradients=True,
            input_pass_records=bridge_state['live_frame_passes'], image_reader=image_reader.statistics(),
            physical_GPU_UUID=expected_uuid, physical_mapping_verified=True,
            peak_own_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
            scope='Two diagnostic GPUupdates/path, seed0, actualRGBand NovemberIR Train images. '
                  'The live encoder is executed B1; no feature callback. No formal24pass fit, '
                  'expanded coverage result, Dev/Test accuracy or HOTA is claimed. '
                  'Deterministic CUDA algorithms are enabled for this engineering comparison; '
                  'this is not a bit-exact reproduction claim for earlier nondeterministic fits.',
            goal_met=False, supported_innovations=0)
        dump(out / 'state.json', state)
        destination = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_TRAIN_STREAM_OPTIMIZER_GPU_PARITY.json'
        assert not destination.exists()
        dump(destination, state)
        print(json.dumps(dict(status=state['status'], seconds=state['seconds'])), flush=True)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_LIVE_GPU_OPTIMIZER_PARITY', error=repr(error))
        dump(out / 'state.json', state)
        raise


if __name__ == '__main__':
    main()
