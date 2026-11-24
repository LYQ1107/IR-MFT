#!/usr/bin/env python3
"""Unchanged structural inference core, using only registered full-Train weights."""
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
from tools.structural_gate_observer import StructuralGateObserver, SCOPE
from yolox.exp import get_exp
from yolox.models.structural_factorial_repair import StructuralFactorial, FLAGS


def context():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    study_path = ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json'
    study = json.loads(study_path.read_text())
    config_path = ROOT / study['configuration']
    assert sha256(config_path) == study['configuration_sha256']
    config = json.loads(config_path.read_text())
    policy_path = ROOT / 'configs/perceptual_single_seed_policy_20261123.json'
    assert sha256(policy_path) == study['single_seed_policy_sha256']
    policy = json.loads(policy_path.read_text())
    assert policy['fixed_seed'] == 0 and not policy['additional_seed_inference']
    assert policy['structural_base_config_sha256'] == study['parent_structural_config_sha256']
    fit = json.loads((Path(plan['work_root']) / 'structural_full_train_fit_state.json').read_text())
    assert fit['status'] == 'COMPLETE' and all(j['status'] == 'COMPLETE' for j in fit['jobs'].values())
    assert fit['full_train_study_sha256'] == sha256(study_path)
    return plan, study, study_path, config, config_path, policy_path


def load_bundle(spec, plan, study, study_path, config, config_path):
    parent_path = ROOT / 'configs/perceptual_correlation_repair_20261123.json'
    parent = json.loads(parent_path.read_text())
    assert sha256(parent_path) == config['parent_plan_sha256']
    checkpoint = Path(parent['source_RGB_checkpoint'])
    assert sha256(checkpoint) == parent['source_RGB_sha256']
    model = get_exp(str(ROOT / 'exps/yolox_mft26ir_visible.py'), None).get_model()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model = model.cuda().half().eval().requires_grad_(False)
    architecture = config['architecture']
    loaded, information, needed = {}, {}, {'rgb'}
    for row in spec['models']:
        name, cue = row['id'], row['cue']
        assert name not in loaded and row.get('seed', 0) == 0
        needed.add('ir_image_roll' if row.get('intervention') == 'IR_image_spatial_roll' else cue)
        if row.get('baseline'):
            loaded[name], information[name] = None, dict(row)
            continue
        bits = row['components']
        assert row['variant'] == ('no_cycle' if bits[1] == '1' else 'no_centering')
        adapter = StructuralFactorial(latent=architecture['latent_channels'], radius=architecture['local_radius'],
            temperature=architecture['structural_temperature'], bound=architecture['feature_residual_bound_RMS_fraction'],
            risk_threshold=architecture['teacher_risk_threshold'], components=bits,
            tolerance=architecture['forward_backward_tolerance_cells'])
        info = dict(row, component_flags=FLAGS[bits])
        if not row.get('zero_adapter'):
            path = Path(row['checkpoint'])
            fit_proof_path = path.parent / 'full_train_complete.json'
            fit_proof = json.loads(fit_proof_path.read_text())
            assert fit_proof['status'] == 'COMPLETE_THREE_FULL_TRAIN_TRAVERSALS'
            assert fit_proof['post_loop_source_verified'] and fit_proof['full_frame_address_delivery_verified']
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['config_sha256'] == sha256(config_path) and saved['completed_epochs'] == row['epoch']
            meta = saved['metadata']
            assert (meta['cue'], meta['variant'], meta['seed']) == (cue, bits, 0)
            assert meta['full_train_study_sha256'] == sha256(study_path)
            assert meta['component_flags'] == FLAGS[bits]
            assert meta['full_train_address_sha256'] == study['semantic_address_sha256']
            assert meta['Train_frames'] == 27251 and meta['source_RGB_sha256'] == parent['source_RGB_sha256']
            adapter.load_state_dict(saved['adapter'], strict=True)
            info.update(checkpoint_sha256=sha256(path), metadata=meta,
                        full_fit_proof_sha256=sha256(fit_proof_path))
        loaded[name], information[name] = adapter.cuda().float().eval().requires_grad_(False), info
    return model, loaded, information, needed


def zero_specification():
    models = [dict(id='CG3_source_RGB', cue='rgb', baseline=True, seed=0)]
    for cue, bits in [('rgb', '111'), ('gray', '111'), ('ir', '111'), ('ir', '011'), ('ir', '101'), ('ir', '110')]:
        models.append(dict(id=f'FT_zero_{cue}_{bits}', cue=cue, components=bits, seed=0,
            variant='no_cycle' if bits[1] == '1' else 'no_centering', zero_adapter=True))
    for intervention in ['IR_image_spatial_roll', 'unavailable']:
        models.append(dict(id='FT_zero_ir_' + intervention, cue='ir', components='111',
            seed=0, variant='no_cycle', zero_adapter=True, intervention=intervention))
    return dict(models=models)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--specification')
    parser.add_argument('--source-check', type=int, default=0)
    parser.add_argument('--limit-frames', type=int, default=0)
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical in ['4', '5'] and args.source_check >= 0 and args.limit_frames >= 0
    assert not args.limit_frames or args.split == 'dev'
    plan, study, study_path, config, config_path, policy_path = context()
    work = Path(plan['work_root'])
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    assert args.sequence in protocol['dev' if args.split == 'dev' else 'historical_test']
    if args.split == 'test':
        freeze = json.loads((work / 'structural_full_train_Test_selection_frozen.json').read_text())
        assert freeze['study_sha256'] == sha256(study_path) and freeze['complete50Dev_before_Test']
    specification = zero_specification() if args.source_check else json.loads(Path(args.specification).read_text())
    specification_sha = None if args.source_check else sha256(args.specification)
    if not args.source_check:
        assert specification['config_sha256'] == sha256(config_path)
        assert specification['full_train_study_sha256'] == sha256(study_path)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    model, loaded, information, needed = load_bundle(specification, plan, study, study_path, config, config_path)
    calibration_path = work / 'correlation_cache/calibration.json'
    calibration = json.loads(calibration_path.read_text())
    assert calibration['config_sha256'] == config['parent_plan_sha256']
    observer = StructuralGateObserver()
    for name, adapter in loaded.items():
        if adapter is not None:
            observer.attach(name, adapter)
    try:
        produce(args, physical, plan, config, config_path, policy_path, specification, specification_sha,
                calibration, calibration_path, work / 'full_train_observations', model, loaded, information, needed)
    finally:
        observer.close()
    out = work / 'full_train_observations/structural_relative_predictions' / args.split / args.sequence
    if args.source_check:
        out = out / f'source_check_{args.source_check}'
    elif args.limit_frames:
        out = out / f'prefix_check_{args.limit_frames}'
    meta = json.loads((out / 'complete.json').read_text())
    for name in ['tools/extract_structural_full_train.py', 'yolox/models/structural_factorial_repair.py',
                 'tools/structural_gate_observer.py']:
        meta['source_hashes'][name] = sha256(ROOT / name)
    for name in loaded:
        statistics = observer.summary(name)
        if statistics:
            meta['models'][name]['gate_statistics'] = statistics
    meta['gate_statistics_scope'] = SCOPE + ' Disabled-R variants feed zero teacher scores; their component_flags disclose that intervention.'
    meta['full_train_study_sha256'] = sha256(study_path)
    dump(out / 'complete.json', meta)


if __name__ == '__main__':
    main()
