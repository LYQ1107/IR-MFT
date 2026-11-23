#!/usr/bin/env python3
"""S/C/R combinations using the unchanged structural-relative prediction code."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import torch
from tools.extract_structural_relative import produce
from tools.perceptual_gpu_handoff import future_gpus
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.structural_gate_observer import StructuralGateObserver,SCOPE as GATE_SCOPE
from yolox.exp import get_exp
from yolox.models.structural_factorial_repair import StructuralFactorial,FLAGS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sequence',required=True)
    parser.add_argument('--split',choices=['dev','test'],default='dev')
    parser.add_argument('--specification')
    parser.add_argument('--source-check',type=int,default=0)
    parser.add_argument('--limit-frames',type=int,default=0)
    args=parser.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES');assert physical in future_gpus()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work=Path(plan['work_root'])
    assert json.loads((work/'structural_factorial_fit_state.json').read_text())['status']=='COMPLETE'
    phase_path=ROOT/'configs/perceptual_structural_factorial_20261123.json'
    phase=json.loads(phase_path.read_text())
    parent_phase=json.loads((ROOT/'configs/perceptual_structural_nocycle_20261123.json').read_text())
    config_path=ROOT/phase['base_configuration'];config=json.loads(config_path.read_text())
    assert sha256(ROOT/'configs/perceptual_mft_20261122.json')==config['main_plan_sha256']
    policy_path=ROOT/'configs/perceptual_single_seed_policy_20261123.json'
    policy=json.loads(policy_path.read_text())
    assert phase['base_config_sha256']==sha256(config_path)==policy['structural_base_config_sha256']
    assert phase['single_seed_policy_sha256']==sha256(policy_path)
    assert phase['fixed_seed']==policy['fixed_seed']==0 and not policy['additional_seed_inference']
    parent_path=ROOT/'configs/perceptual_correlation_repair_20261123.json'
    parent=json.loads(parent_path.read_text());assert sha256(parent_path)==config['parent_plan_sha256']
    protocol=json.loads((ROOT/plan['protocol']).read_text())
    assert args.sequence in protocol['dev'if args.split=='dev'else'historical_test']
    assert args.source_check>=0 and args.limit_frames>=0
    assert not args.limit_frames or args.split=='dev'
    if args.split=='test':
        freeze=json.loads((work/'structural_factorial_test_selection_frozen.json').read_text())
        assert freeze['factorial_plan_sha256']==sha256(phase_path) and freeze['complete_Dev_before_Test']
    calibration_path=work/'correlation_cache/calibration.json'
    calibration=json.loads(calibration_path.read_text());assert calibration['config_sha256']==sha256(parent_path)
    if args.source_check:
        models=[dict(id='CG3_source_RGB',cue='rgb',baseline=True)]
        for bits,flags in FLAGS.items():
            models.append(dict(id='FCT_zero_'+bits,cue='ir',components=bits,zero_adapter=True,seed=0,
                variant='no_cycle'if flags['centered']else'no_centering'))
        spec=dict(models=models);spec_sha256=None
    else:
        spec=json.loads(Path(args.specification).read_text());spec_sha256=sha256(args.specification)
        assert spec['config_sha256']==sha256(config_path)
        assert spec['factorial_plan_sha256']==sha256(phase_path)
        assert spec['single_seed_user_policy_sha256']==sha256(policy_path)
    assert all(r.get('seed',0)==0 for r in spec['models'])
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.45,0)
    checkpoint=Path(parent['source_RGB_checkpoint']);assert sha256(checkpoint)==parent['source_RGB_sha256']
    model=get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'),None).get_model()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True)
    del saved;model=model.cuda().half().eval().requires_grad_(False)
    a=config['architecture'];loaded={};model_info={};needed={'rgb'}
    for row in spec['models']:
        name=row['id'];assert name not in loaded;needed.add(row['cue'])
        if row.get('baseline'):
            loaded[name]=None;model_info[name]=dict(row);continue
        bits=row['components'];assert bits in FLAGS
        assert row['variant']==('no_cycle'if FLAGS[bits]['centered']else'no_centering')
        adapter=StructuralFactorial(latent=a['latent_channels'],radius=a['local_radius'],
            temperature=a['structural_temperature'],bound=a['feature_residual_bound_RMS_fraction'],
            risk_threshold=a['teacher_risk_threshold'],components=bits,tolerance=a['forward_backward_tolerance_cells'])
        assert sum(p.numel()for p in adapter.parameters())==phase['trainable_parameters']==145600
        info=dict(row,component_flags=FLAGS[bits])
        if not row.get('zero_adapter'):
            path=Path(row['checkpoint']);saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['config_sha256']==sha256(config_path) and saved['completed_epochs']==row['epoch']
            meta=saved['metadata'];assert meta['cue']=='ir' and meta['seed']==0
            assert meta['source_RGB_sha256']==parent['source_RGB_sha256']
            assert meta['single_seed_user_policy_sha256']==sha256(policy_path)
            if row.get('checkpoint_origin')=='initial_structural_relative':
                assert bits=='111' and meta['variant']=='no_cycle' and row['epoch']==6
                assert path==Path(parent_phase['primary_IR']['checkpoint'])
            else:
                assert meta['variant']==bits and meta['factorial_plan_sha256']==sha256(phase_path)
                assert meta['component_flags']==FLAGS[bits]
            adapter.load_state_dict(saved['adapter'],strict=True)
            info.update(checkpoint_sha256=sha256(path),metadata=meta);del saved
        loaded[name]=adapter.cuda().float().eval().requires_grad_(False);model_info[name]=info
    observer=StructuralGateObserver()
    if args.split=='test':
        for name,adapter in loaded.items():
            if adapter is not None:observer.attach(name,adapter)
    nested=work/'factorial_observations'
    try:
        produce(args,physical,plan,config,config_path,policy_path,spec,spec_sha256,
                calibration,calibration_path,nested,model,loaded,model_info,needed)
    finally:observer.close()
    out=nested/'structural_relative_predictions'/args.split/args.sequence
    if args.source_check:out=out/f'source_check_{args.source_check}'
    elif args.limit_frames:out=out/f'prefix_check_{args.limit_frames}'
    meta=json.loads((out/'complete.json').read_text())
    for name in ['tools/extract_structural_factorial.py','yolox/models/structural_factorial_repair.py',
                 'tools/structural_gate_observer.py','tools/perceptual_gpu_handoff.py',
                 'configs/perceptual_gpu_handoff_20261123.json']:
        meta['source_hashes'][name]=sha256(ROOT/name)
    if args.split=='test':
        for name in loaded:
            stats=observer.summary(name)
            if stats:meta['models'][name]['gate_statistics']=stats
        meta['gate_statistics_scope']=GATE_SCOPE
    meta.update(factorial_plan_sha256=sha256(phase_path),loader_source_sha256=sha256(__file__))
    dump(out/'complete.json',meta)


if __name__=='__main__':main()
