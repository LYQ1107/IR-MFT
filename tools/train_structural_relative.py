#!/usr/bin/env python3
"""Train structure-verified relative semantics on existing frozen Train FPNs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.run_pseudo_ir_comparison import dump, sha256
from tools.train_correlation_repair import load_banks, batches, tensors
from yolox.models.yolo_head import YOLOXHead
from yolox.models.frozen_head_repair import head_predictions, original_supervised_loss
from yolox.models.recoverable_curriculum import valid_counterfactual_locations
from yolox.models.structural_relative_repair import StructuralRepair, relative_semantics
from yolox.models.structural_relative_losses import preserve_semantics


def load_structure(work, sequences, config_sha):
    state = json.loads((work/'structural_cache/state.json').read_text())
    assert state['status'] == 'COMPLETE' and state['specification_sha256'] == config_sha
    banks = {}; manifests = {}
    for sequence in sequences:
        path = work/'structural_cache'/sequence/'complete.json'
        meta = json.loads(path.read_text())
        assert meta['status'] == 'COMPLETE_TRAIN_SCALAR_STRUCTURE' and meta['specification_sha256'] == config_sha
        assert not meta['GT_read'] and meta['Dev_rows'] == meta['Test_rows'] == 0
        parent = work/'correlation_cache'/sequence/'features/complete.json'
        assert meta['parent_cache_manifest_sha256'] == sha256(parent)
        assert meta['frame_ids'] == json.loads(parent.read_text())['frame_ids']
        banks[sequence] = {}
        for name, row in meta['outputs'].items():
            assert sha256(row['path']) == row['sha256']
            banks[sequence][name] = np.load(row['path'], mmap_mode='r')
        manifests[sequence] = dict(path=str(path), sha256=sha256(path), frames=192)
    return banks, manifests


def structural_tensors(banks, draw, cue):
    values = {}
    for mode in set(['rgb', 'ir' if cue == 'ir' else 'rgb']):
        for name in ['descriptor', 'valid']:
            values[mode+'_'+name] = torch.from_numpy(np.stack(
                [banks[seq][mode+'_'+name][index] for seq, index in draw])).cuda()
    source = 'ir' if cue == 'ir' else 'rgb'
    return values['rgb_descriptor'], values[source+'_descriptor'], values['rgb_valid'], values[source+'_valid']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cue', choices=['rgb', 'gray', 'ir'], required=True)
    parser.add_argument('--variant', choices=['full', 'no_structure', 'no_cycle', 'no_centering'], default='full')
    parser.add_argument('--seed', type=int, choices=[0], required=True)
    parser.add_argument('--smoke-steps', type=int, default=0)
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical in ['2', '3'] and args.smoke_steps >= 0
    torch.set_num_threads(2); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    config_path = ROOT/'configs/perceptual_structural_relative_20261123.json'
    config = json.loads(config_path.read_text()); parent_path = ROOT/'configs/perceptual_correlation_repair_20261123.json'
    policy_path = ROOT/'configs/perceptual_single_seed_policy_20261123.json'
    policy = json.loads(policy_path.read_text())
    assert policy['fixed_seed'] == args.seed == 0 and not policy['additional_seed_fits']
    assert policy['structural_base_config_sha256'] == sha256(config_path)
    parent = json.loads(parent_path.read_text()); work = Path(plan['work_root'])
    assert sha256(parent_path) == config['parent_plan_sha256']
    assert sha256(ROOT/'configs/perceptual_mft_20261122.json') == config['main_plan_sha256']
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    out = work/'structural_relative_fit'/f'{args.cue}_{args.variant}_s{args.seed}'
    if args.smoke_steps:
        out = out/f'smoke_{args.smoke_steps}'
    out.mkdir(parents=True, exist_ok=True); assert not (out/'complete.json').exists()
    sequences, banks, fpn_sources = load_banks(work, protocol, parent_path, args.cue, False)
    structure, structure_sources = load_structure(work, sequences, sha256(config_path))
    checkpoint = Path(parent['source_RGB_checkpoint']); assert sha256(checkpoint) == parent['source_RGB_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    head = YOLOXHead(1, width=1.25)
    head.load_state_dict({key.removeprefix('head.'): value for key, value in saved['model'].items()
                         if key.startswith('head.')}, strict=True); del saved
    head = head.cuda().float().eval().requires_grad_(False)
    architecture = config['architecture']
    adapter = StructuralRepair(latent=architecture['latent_channels'], radius=architecture['local_radius'],
        temperature=architecture['structural_temperature'], bound=architecture['feature_residual_bound_RMS_fraction'],
        risk_threshold=architecture['teacher_risk_threshold'], variant=args.variant,
        tolerance=architecture['forward_backward_tolerance_cells']).cuda()
    fit = config['fit']; optimizer = torch.optim.AdamW(adapter.parameters(), lr=fit['learning_rate'], weight_decay=fit['weight_decay'])
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in [Path(__file__),
        ROOT/'yolox/models/structural_relative_repair.py', ROOT/'yolox/models/structural_relative_losses.py',
        ROOT/'yolox/models/frozen_head_repair.py', ROOT/'yolox/models/recoverable_curriculum.py',
        ROOT/'tools/train_correlation_repair.py']}
    state = dict(status='FITTING_STRUCTURAL_RELATIVE', cue=args.cue, variant=args.variant, seed=args.seed,
        physical_gpu=physical, config_sha256=sha256(config_path), parent_sha256=sha256(parent_path),
        source_RGB_sha256=sha256(checkpoint), FPN_sources=fpn_sources, structure_sources=structure_sources,
        source_hashes=sources, single_seed_user_policy_sha256=sha256(policy_path),
        frozen_original_head=True, frozen_original_encoder=True,
        trainable_parameters=sum(value.numel() for value in adapter.parameters()),
        Train_videos=sequences, Train_frames=1152, Dev_fit_rows=0, Test_read=False,
        geometry='Exact detached RGB decoded geometry; only original obj/class supervision is optimized.')
    dump(out/'fitting_manifest.json', state)
    # Training loop is kept below the complete provenance setup.
    run_fit(args, adapter, head, optimizer, banks, structure, sequences, config, config_path, out, state)


def run_fit(args, adapter, head, optimizer, banks, structure, sequences, config, config_path, out, state):
    fit = config['fit']; architecture = config['architecture']
    steps_per_epoch = sum(len(banks[seq]['targets']) for seq in sequences)//4
    assert steps_per_epoch == 288
    epochs = 1 if args.smoke_steps else config['epochs']
    history = []; successful = 0; started = time.perf_counter()
    permutation = torch.tensor([1, 0, 3, 2], device='cuda')
    initial_exact = False
    for epoch in range(1, epochs+1):
        values = []; draw_digest = hashlib.sha256(); adapter.train()
        for step, draw in enumerate(batches(sequences, banks, args.seed, epoch)):
            if args.smoke_steps and step >= args.smoke_steps:
                break
            draw_digest.update(json.dumps(draw, separators=(',', ':')).encode())
            position = (epoch-1)*steps_per_epoch+step
            factor = ((position+1)/steps_per_epoch if position < steps_per_epoch else
                .03+.97*.5*(1+math.cos(math.pi*(position-steps_per_epoch)/((config['epochs']-1)*steps_per_epoch))))
            for group in optimizer.param_groups:
                group['lr'] = fit['learning_rate']*factor
            rgb, auxiliary, target = tensors(banks, draw, args.cue)
            query, key, query_valid, key_valid = structural_tensors(structure, draw, args.cue)
            with torch.no_grad():
                reference, meta = head_predictions(head, rgb)

            def predict(view, descriptor, validity, null=False):
                repaired = adapter(rgb, view, meta['scores'], query, descriptor, query_valid, validity, null=null)
                return head_predictions(head, repaired)[0]

            current = predict(auxiliary, key, key_valid)
            null = predict(auxiliary, key, key_valid, null=True)
            prediction = relative_semantics(current, null, reference,
                centered=args.variant != 'no_centering', bound=architecture['semantic_logit_bound'])
            assert torch.equal(prediction[:, :, :4], reference[:, :, :4])
            if epoch == 1 and step == 0:
                assert torch.equal(current, reference) and torch.equal(null, reference)
                assert torch.equal(prediction, reference)
                initial_exact = True
            supervised_parts = original_supervised_loss(head, prediction, meta, target, config['training_input_hw'])
            supervised = supervised_parts[2]+supervised_parts[3]
            reliable = preserve_semantics(prediction, reference, reliable_only=True)
            wrong_view = [value[permutation] for value in auxiliary]
            wrong_current = predict(wrong_view, key[permutation], key_valid[permutation])
            wrong_null = predict(wrong_view, key[permutation], key_valid[permutation], null=True)
            wrong_prediction = relative_semantics(wrong_current, wrong_null, reference,
                centered=args.variant != 'no_centering', bound=architecture['semantic_logit_bound'])
            validity = valid_counterfactual_locations(target, permutation, meta)
            wrong = preserve_semantics(wrong_prediction, reference, validity)
            loss = supervised+fit['lambda_reliable_RGB']*reliable+fit['lambda_wrong_auxiliary']*wrong
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite structural loss; retain failure, do not claim completion')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(adapter.parameters(), fit['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step(); successful += 1
            values.append([float(v.detach()) for v in [loss, supervised, reliable, wrong, gradient]])
            if (step+1) % 32 == 0 or args.smoke_steps:
                print(json.dumps(dict(epoch=epoch, step=step+1, steps=steps_per_epoch,
                    loss=np.mean(values, axis=0).tolist(), physical_gpu=state['physical_gpu'],
                    seconds=time.perf_counter()-started,
                    peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)), flush=True)
        row = dict(epoch=epoch, steps=len(values), loss=np.mean(values, axis=0).tolist(),
                   sample_digest=draw_digest.hexdigest())
        history.append(row)
        exported = {key: value.detach().cpu() for key, value in adapter.state_dict().items()}
        payload = dict(adapter=exported, completed_epochs=epoch, history=history,
            config_sha256=sha256(config_path), metadata=state, optimizer=optimizer.state_dict())
        temporary = out/'latest.tmp.pt'; torch.save(payload, temporary); os.replace(temporary, out/'latest.pt')
        if epoch in config['snapshots'] or args.smoke_steps:
            compact = {key: value for key, value in payload.items() if key != 'optimizer'}
            temporary = out/f'epoch{epoch:02d}.tmp.pt'; torch.save(compact, temporary)
            os.replace(temporary, out/f'epoch{epoch:02d}.pt')
        dump(out/'history.json', history); print(json.dumps(row), flush=True)
    assert initial_exact
    assert all(sha256(ROOT/name) == value for name, value in state['source_hashes'].items())
    assert state['config_sha256'] == sha256(config_path)
    state.update(status='SMOKE_COMPLETE' if args.smoke_steps else 'COMPLETE_TRAIN_STRUCTURAL_RELATIVE',
        epochs=epochs, successful_steps_this_invocation=successful, history=history,
        seconds=time.perf_counter()-started, initial_raw_RGB_head_bit_exact=True,
        peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
    dump(out/'complete.json', state)


if __name__ == '__main__':
    main()
