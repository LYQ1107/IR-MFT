#!/usr/bin/env python3
"""Actual licensed DINO/current-fitcrop mask exclusion and gradient/EMA smoke."""
import argparse,json,os
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from cfp.dinov2_identity import load_pinned_small,resized_observations
from cfp.dinov2_fish_data import load_crop_banks,pixel_batch
from cfp.fish_masked_identity import MaskedPixelFishIdentity,masked_patch_identity_loss
from cfp.masked_identity_checkpoint import frozen_base,delta_state,load_delta_state
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('NewCPU-only fittedcrop smoke required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(4);r=a.root;proof=json.loads((r/'dinov2_small_inference_source_smoke_v1.json').read_text());weights=r/'external_models/dinov2_vits14_v1/dinov2_vits14_pretrain.pth';backbone=load_pinned_small('/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2',weights,proof['imported_source_proofs']);model=MaskedPixelFishIdentity(backbone,2).eval();base=frozen_base(model)
    banks,dimensions,_=load_crop_banks(r/'identity_pilot_crop_cache_v1',r/'dinov2_two_crop_evidence_v1',r/'identity_two_crop_index_v1');video='BT-003';ids=[i for i,x in enumerate(banks[video]['records']) if x['view']=='gt' and x['frame']==1][:2];assert len(ids)==2
    images,content=pixel_batch([(video,i) for i in ids],banks,dimensions,'cpu');prepared,fraction=resized_observations(images,content);mask=(fraction.flatten(1)>0)&(torch.arange(162)[None]%2==0);changed=prepared.clone()
    for j in range(2):
        for index in mask[j].nonzero(as_tuple=True)[0].tolist():
            y,x=divmod(index,18);changed[j,:,y*14:(y+1)*14,x*14:(x+1)*14]+=7
    with torch.inference_mode():
        original=model.masked_prepared(prepared,mask);perturbed=model.masked_prepared(changed,mask);target=model.teacher.forward_features(prepared)['x_norm_patchtokens'];altered=model.teacher.forward_features(changed)['x_norm_patchtokens'];assert torch.equal(original,perturbed);assert not torch.equal(target,altered)
    frozen={name:p.clone() for name,p in model.student.backbone.named_parameters() if not p.requires_grad};teacher_before={name:p.clone() for name,p in model.teacher.named_parameters()};model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5);observed=model.training_observations(images,content,mask);masked,count=masked_patch_identity_loss(observed);identity=F.cross_entropy(observed['identity']['logits'],torch.tensor([0,1]));loss=identity+.25*masked;loss.backward()
    gradients={name:float(p.grad.double().norm()) for name,p in model.named_parameters() if p.grad is not None};assert all(np.isfinite(v) for v in gradients.values());assert any(v>0 for k,v in gradients.items() if k.startswith('predictor.'));assert any(v>0 for k,v in gradients.items() if k.startswith('student.backbone.blocks.11.'));assert all(p.grad is None and not p.requires_grad for p in model.teacher.parameters());optimizer.step();model.update_target(True);assert int(model.teacher_updates)==1;assert all(torch.equal(p,frozen[name]) for name,p in model.student.backbone.named_parameters() if name in frozen);assert any(not torch.equal(p,teacher_before[name]) for name,p in model.teacher.named_parameters())
    checkpoint=a.output.with_suffix('.pt');saved_delta=delta_state(model,base);torch.save(dict(model_delta=saved_delta,pretrained_sha256=sha256(weights)),checkpoint)
    fresh=MaskedPixelFishIdentity(load_pinned_small('/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2',weights,proof['imported_source_proofs']),2).eval();fresh_base=frozen_base(fresh);loaded=torch.load(checkpoint,map_location='cpu',weights_only=False);load_delta_state(fresh,loaded['model_delta'],fresh_base)
    assert all(torch.equal(value,fresh.state_dict()[key]) for key,value in model.state_dict().items())
    checkpoint_proof=dict(all_model_keys_equal=len(model.state_dict()),delta_keys=len(saved_delta),omitted_base_keys=len(base),checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=sha256(checkpoint),teacher_updates=int(fresh.teacher_updates))
    model.teacher.forward_features=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('Teacheronline'));model.predictor.forward=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('Predictiononline'))
    with torch.inference_mode():assert model.online(images,content)['embedding'].shape==(2,384)
    write_json(a.output,dict(status='VERIFIED_ACTUAL_PIXEL_MASK_AND_LOSSLESS_BASE_DELTA_CHECKPOINT',checkpoint_proof=checkpoint_proof,video=video,crops=2,identity_vocabulary=2,masked_real_content_patches=count,pretrained_sha256=sha256(weights),input_pixel_sha256=__import__('hashlib').sha256(images.numpy().tobytes()).hexdigest(),masked_target_pixel_change_cannot_change_prediction=True,unmasked_teacher_changes_with_target_pixels=True,teacher_stopped_independent=True,teacher_updates=1,frozen_prefix_unchanged=True,gradient_norms=gradients,identity_CE=float(identity.detach()),masked_L1=float(masked.detach()),teacher_decoder_excluded_online=True,sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,'cfp/fish_masked_identity.py','cfp/masked_identity_checkpoint.py']),scope='ActualstrictApacheDINO22M/currentGTfitcrops fromBT003, excludesfirstouterSN015/innerBT001. Genuine patch-embedding masks before all attention, not deleting cached globallymixed features. OneCPUoptimizer/EMAupdate only; not8epochtraining, identitygain/nativeMOT/novelty/finalgoal claim. Teacher target offline-only. Fullcontrolledmaskedfishtraining remains required.'))
    print(json.dumps(dict(masked_patches=count,masked_pixel_exclusion=True,gradients=len(gradients),teacher_updates=1)))

if __name__=='__main__':main()
