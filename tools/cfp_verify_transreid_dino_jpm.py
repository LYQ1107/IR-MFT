#!/usr/bin/env python3
"""Actual pinned DINO/JPM source parity, fit-crop gradients and delta restore."""
import argparse,ast,copy,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cfp.dinov2_identity import load_pinned_small
from cfp.dinov2_fish_data import load_crop_banks,pixel_batch
from cfp.transreid_dino_jpm import DinoJointPatchReID
from cfp.masked_identity_checkpoint import delta_state,load_delta_state
from cfp.provenance import sha256,snapshot_sources,write_json
from third_party.transreid_jpm.shuffle import shuffle_unit


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU-only actual fitcrop proof required')
    torch.set_num_threads(2);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    ref=Path('/data3/liuyeqiang/CFP-FishMOT_references/phase7/TransReID');commit=subprocess.check_output(['git','-C',str(ref),'rev-parse','HEAD'],text=True).strip();assert commit=='dec55046fcdfadee14e2c28e2df89305d8f7557a'
    source=(ref/'model/make_model.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='shuffle_unit');namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(ref/'model/make_model.py'),'exec'),namespace)
    for n in [162,163]:
        x=torch.arange(n*384).reshape(1,n,384).float();assert torch.equal(namespace['shuffle_unit'](x,5,2),shuffle_unit(x,5,2))
    r=a.root;weights=r/'external_models/dinov2_vits14_v1/dinov2_vits14_pretrain.pth';proof=json.loads((r/'dinov2_small_inference_source_smoke_v1.json').read_text());backbone=load_pinned_small('/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2',weights,proof['imported_source_proofs'])
    model=DinoJointPatchReID(copy.deepcopy(backbone),2,'jpm');capacity=DinoJointPatchReID(copy.deepcopy(backbone),2,'cls_capacity');assert sum(p.numel() for p in model.parameters() if p.requires_grad)==sum(p.numel() for p in capacity.parameters() if p.requires_grad)
    banks,dimensions,_=load_crop_banks(r/'identity_pilot_crop_cache_v1',r/'dinov2_two_crop_evidence_v1',r/'identity_two_crop_index_v1');video='BT-003';ids=[i for i,x in enumerate(banks[video]['records']) if x['view']=='gt' and x['frame']==1][:2];assert len(ids)==2
    images,content=pixel_batch([(video,i) for i in ids],banks,dimensions,'cpu');base=model.immutable_base();model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5);out=model(images,content);loss=.5*F.cross_entropy(out['logits'][0],torch.tensor([0,1]))+.125*sum(F.cross_entropy(x,torch.tensor([0,1])) for x in out['logits'][1:]);loss.backward()
    gradients={n:float(p.grad.double().norm()) for n,p in model.named_parameters() if p.grad is not None};assert all(np.isfinite(v) for v in gradients.values())
    for prefix in ['backbone.blocks.10.','global_branch.','local_branch.','classifiers.']:
        assert any(v>0 for n,v in gradients.items() if n.startswith(prefix))
    assert all(p.grad is None for n,p in model.backbone.named_parameters() if not p.requires_grad);opt.step();saved=delta_state(model,base);ck=a.output.with_suffix('.pt');torch.save(dict(model_delta=saved,pretrained_sha256=sha256(weights)),ck)
    fresh=DinoJointPatchReID(copy.deepcopy(backbone),2,'jpm');load_delta_state(fresh,torch.load(ck,map_location='cpu',weights_only=False)['model_delta'],fresh.immutable_base());assert all(torch.equal(v,fresh.state_dict()[n]) for n,v in model.state_dict().items());model.eval();fresh.eval();bn={n:v.clone() for n,v in model.named_buffers() if n.startswith('necks.')}
    with torch.inference_mode():
        e=model(images,content)['embedding'];assert e.shape==(2,1920) and torch.equal(e,fresh(images,content)['embedding']);assert torch.count_nonzero(model(images,content*0)['embedding'])==0
    assert all(torch.equal(v,bn[n]) for n,v in model.named_buffers() if n.startswith('necks.'))
    result=dict(status='VERIFIED_ACTUAL_PINNED_DINO_JPM_SOURCE_GRADIENT_AND_DELTA',source_commit=commit,pretrained_sha256=sha256(weights),source_shuffle_exact_equal=True,video=video,actual_fit_crops=2,outer_inner_excluded=['SN-015','BT-001'],output_dimension=1920,equal_jpm_capacity_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),all_model_keys_exact_restore=len(model.state_dict()),delta_keys=len(saved),immutable_base_keys=len(base),checkpoint_bytes=ck.stat().st_size,checkpoint_sha256=sha256(ck),heldout_style_eval_BN_state_unchanged=True,frozen_prefix_unchanged=True,zero_content_has_no_embedding=True,gradient_norms=gradients,scope='One actualCPUoptimizer step only; not full8epoch fitting, MOTgain, officialwholeTransReID or novelmethod. SourceJPM copiedlastblock/foursharedpartbranch/BNneck/headweights/local1/4scale preserved; DINOpatch14/noSIE andsame-video fishloss adaptation explicit. Prefix CLS/patches alreadygloballymixed; parts notpurelocal/anatomical.',sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,'cfp/transreid_dino_jpm.py','third_party/transreid_jpm/shuffle.py','cfp/masked_identity_checkpoint.py',str(ref/'model/make_model.py')]))
    write_json(a.output,result);print(json.dumps({k:result[k] for k in ['status','output_dimension','equal_jpm_capacity_trainable_parameters','all_model_keys_exact_restore','delta_keys','checkpoint_bytes']}))

if __name__=='__main__':main()
