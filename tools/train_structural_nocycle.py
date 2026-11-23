#!/usr/bin/env python3
"""Same cached training loop; isolate centering and teacher-risk after cycle removal."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from tools.run_pseudo_ir_comparison import dump,sha256
from tools.train_correlation_repair import load_banks
from tools.train_structural_relative import load_structure,run_fit
from yolox.models.yolo_head import YOLOXHead
from yolox.models.structural_nocycle_refinement import NoCycleRefinement,VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--cue',choices=['rgb','gray','ir'],required=True)
    p.add_argument('--variant',choices=list(VARIANTS),required=True)
    p.add_argument('--seed',type=int,choices=[0],required=True)
    p.add_argument('--smoke-steps',type=int,default=0);args=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES');assert physical in ['2','3'] and args.smoke_steps>=0
    main_path=ROOT/'configs/perceptual_mft_20261122.json';plan=json.loads(main_path.read_text());work=Path(plan['work_root'])
    phase_path=ROOT/'configs/perceptual_structural_nocycle_20261123.json';phase=json.loads(phase_path.read_text())
    assert any((r['cue'],r['variant'],r['seed'])==(args.cue,args.variant,args.seed)for r in phase['fits'])
    config_path=ROOT/phase['base_configuration'];config=json.loads(config_path.read_text())
    policy_path=ROOT/'configs/perceptual_single_seed_policy_20261123.json';policy=json.loads(policy_path.read_text())
    assert phase['base_config_sha256']==sha256(config_path)==policy['structural_base_config_sha256']
    assert phase['single_seed_policy_sha256']==sha256(policy_path)
    assert args.seed==phase['fixed_seed']==policy['fixed_seed']==0 and not policy['additional_seed_fits']
    assert phase['epochs']==config['epochs']==24
    parent_path=ROOT/'configs/perceptual_correlation_repair_20261123.json';parent=json.loads(parent_path.read_text())
    assert sha256(parent_path)==config['parent_plan_sha256'] and sha256(main_path)==config['main_plan_sha256']
    protocol=json.loads((ROOT/plan['protocol']).read_text())
    torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.45,0)
    out=work/'structural_nocycle_fit'/f'{args.cue}_{args.variant}_s0'
    if args.smoke_steps:out=out/f'smoke_{args.smoke_steps}'
    out.mkdir(parents=True,exist_ok=True);assert not (out/'complete.json').exists()
    sequences,banks,fpn_sources=load_banks(work,protocol,parent_path,args.cue,False)
    structure,structure_sources=load_structure(work,sequences,sha256(config_path))
    checkpoint=Path(parent['source_RGB_checkpoint']);assert sha256(checkpoint)==parent['source_RGB_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);head=YOLOXHead(1,width=1.25)
    head.load_state_dict({k.removeprefix('head.'):v for k,v in saved['model'].items()if k.startswith('head.')},strict=True)
    del saved;head=head.cuda().float().eval().requires_grad_(False)
    a=config['architecture'];adapter=NoCycleRefinement(latent=a['latent_channels'],radius=a['local_radius'],
        temperature=a['structural_temperature'],bound=a['feature_residual_bound_RMS_fraction'],
        risk_threshold=a['teacher_risk_threshold'],refinement=args.variant,
        tolerance=a['forward_backward_tolerance_cells']).cuda()
    fit=config['fit'];optimizer=torch.optim.AdamW(adapter.parameters(),lr=fit['learning_rate'],weight_decay=fit['weight_decay'])
    names=['tools/train_structural_nocycle.py','tools/train_structural_relative.py',
        'yolox/models/structural_nocycle_refinement.py','yolox/models/structural_relative_repair.py',
        'yolox/models/structural_relative_losses.py','yolox/models/frozen_head_repair.py',
        'yolox/models/recoverable_curriculum.py','tools/train_correlation_repair.py']
    state=dict(status='FITTING_STRUCTURAL_RELATIVE',cue=args.cue,variant=args.variant,seed=0,
        physical_gpu=physical,config_sha256=sha256(config_path),parent_sha256=sha256(parent_path),
        refinement_plan_sha256=sha256(phase_path),refinement_flags=VARIANTS[args.variant],
        source_RGB_sha256=sha256(checkpoint),FPN_sources=fpn_sources,structure_sources=structure_sources,
        source_hashes={n:sha256(ROOT/n)for n in names},single_seed_user_policy_sha256=sha256(policy_path),
        frozen_original_head=True,frozen_original_encoder=True,
        trainable_parameters=sum(v.numel()for v in adapter.parameters()),Train_videos=sequences,
        Train_frames=1152,Dev_fit_rows=0,Test_read=False,
        geometry='Exact detached RGB geometry; original obj/class losses only. Same base training loop.')
    assert state['trainable_parameters']==phase['trainable_parameters']==145600
    dump(out/'fitting_manifest.json',state)
    loop_args=copy.copy(args)
    loop_args.variant='no_cycle'if adapter.centered else'no_centering'
    run_fit(loop_args,adapter,head,optimizer,banks,structure,sequences,config,config_path,out,state)
    assert sha256(phase_path)==state['refinement_plan_sha256']


if __name__=='__main__':
    main()
