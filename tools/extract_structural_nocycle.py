#!/usr/bin/env python3
"""Load matched no-cycle controls; reuse the audited structural inference math."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import torch

from tools.extract_structural_relative import produce
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.exp import get_exp
from yolox.models.structural_nocycle_refinement import NoCycleRefinement, VARIANTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sequence', required=True)
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    p.add_argument('--specification')
    p.add_argument('--source-check', type=int, default=0)
    p.add_argument('--limit-frames', type=int, default=0)
    args = p.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical in ['2', '3']
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    phase_path = ROOT/'configs/perceptual_structural_nocycle_20261123.json'
    phase = json.loads(phase_path.read_text())
    config_path = ROOT/phase['base_configuration']
    config = json.loads(config_path.read_text())
    policy_path = ROOT/'configs/perceptual_single_seed_policy_20261123.json'
    policy = json.loads(policy_path.read_text())
    assert sha256(config_path) == phase['base_config_sha256'] == policy['structural_base_config_sha256']
    assert sha256(policy_path) == phase['single_seed_policy_sha256']
    assert phase['fixed_seed'] == policy['fixed_seed'] == 0 and not policy['additional_seed_inference']
    parent_path = ROOT/'configs/perceptual_correlation_repair_20261123.json'
    parent = json.loads(parent_path.read_text())
    assert sha256(parent_path) == config['parent_plan_sha256']
    work = Path(plan['work_root'])
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    assert args.sequence in protocol['dev' if args.split == 'dev' else 'historical_test']
    assert args.source_check >= 0 and args.limit_frames >= 0
    assert not args.limit_frames or args.split == 'dev'
    if args.split == 'test':
        freeze = json.loads((work/'structural_nocycle_selection_frozen.json').read_text())
        assert freeze['refinement_plan_sha256'] == sha256(phase_path)
        assert freeze['fixed_seed'] == 0 and freeze['complete_Dev_before_Test']
    calibration_path = work/'correlation_cache/calibration.json'
    calibration = json.loads(calibration_path.read_text())
    assert calibration['config_sha256'] == sha256(parent_path)
    if args.source_check:
        spec = dict(models=[dict(id='CG3_source_RGB', cue='rgb', baseline=True)])
        for cue, variant in [('rgb', 'no_cycle'), ('gray', 'no_cycle'), ('ir', 'no_cycle'),
                             ('ir', 'no_cycle_no_centering'), ('ir', 'no_cycle_no_risk')]:
            spec['models'].append(dict(id=f'NC_zero_{cue}_{variant}', cue=cue,
                fit_variant=variant, variant='no_centering' if variant.endswith('no_centering') else 'no_cycle',
                zero_adapter=True, seed=0))
        spec_sha256 = None
    else:
        spec = json.loads(Path(args.specification).read_text())
        spec_sha256 = sha256(args.specification)
        assert spec['config_sha256'] == sha256(config_path)
        assert spec['refinement_plan_sha256'] == sha256(phase_path)
        assert spec['single_seed_user_policy_sha256'] == sha256(policy_path)
    assert all(row.get('seed', 0) == 0 for row in spec['models'])
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    checkpoint = Path(parent['source_RGB_checkpoint'])
    assert sha256(checkpoint) == parent['source_RGB_sha256']
    model = get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'), None).get_model()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model = model.cuda().half().eval().requires_grad_(False)
    a = config['architecture']
    loaded, model_info, needed = {}, {}, {'rgb'}
    for row in spec['models']:
        name, cue = row['id'], row['cue']
        assert name not in loaded
        needed.add('ir_image_roll' if row.get('intervention') == 'IR_image_spatial_roll' else cue)
        if row.get('baseline'):
            loaded[name], model_info[name] = None, dict(row)
            continue
        variant = row['fit_variant']
        expected_head = 'no_cycle' if VARIANTS[variant]['centered'] else 'no_centering'
        assert row['variant'] == expected_head
        adapter = NoCycleRefinement(latent=a['latent_channels'], radius=a['local_radius'],
            temperature=a['structural_temperature'], bound=a['feature_residual_bound_RMS_fraction'],
            risk_threshold=a['teacher_risk_threshold'], refinement=variant,
            tolerance=a['forward_backward_tolerance_cells'])
        info = dict(row, refinement_flags=VARIANTS[variant])
        if not row.get('zero_adapter'):
            path = Path(row['checkpoint'])
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['config_sha256'] == sha256(config_path) and saved['completed_epochs'] == row['epoch']
            meta = saved['metadata']
            assert (meta['cue'], meta['variant'], meta['seed']) == (cue, variant, 0)
            assert meta['source_RGB_sha256'] == parent['source_RGB_sha256']
            assert meta['single_seed_user_policy_sha256'] == sha256(policy_path)
            if row.get('checkpoint_origin') == 'initial_structural_relative':
                assert cue == 'ir' and variant == 'no_cycle' and row['epoch'] == 6
                assert path == Path(phase['primary_IR']['checkpoint'])
            else:
                assert meta['refinement_plan_sha256'] == sha256(phase_path)
                assert meta['refinement_flags'] == VARIANTS[variant]
            adapter.load_state_dict(saved['adapter'], strict=True)
            info.update(checkpoint_sha256=sha256(path), metadata=meta)
        loaded[name] = adapter.cuda().float().eval().requires_grad_(False)
        model_info[name] = info
    nested_work = work/'nocycle_refinement'
    produce(args, physical, plan, config, config_path, policy_path, spec, spec_sha256,
            calibration, calibration_path, nested_work, model, loaded, model_info, needed)
    out = nested_work/'structural_relative_predictions'/args.split/args.sequence
    if args.source_check:
        out = out/f'source_check_{args.source_check}'
    elif args.limit_frames:
        out = out/f'prefix_check_{args.limit_frames}'
    meta = json.loads((out/'complete.json').read_text())
    for name in ['tools/extract_structural_nocycle.py', 'yolox/models/structural_nocycle_refinement.py']:
        meta['source_hashes'][name] = sha256(ROOT/name)
    meta.update(refinement_plan_sha256=sha256(phase_path), loader_source_sha256=sha256(__file__))
    dump(out/'complete.json', meta)


if __name__ == '__main__':
    main()
