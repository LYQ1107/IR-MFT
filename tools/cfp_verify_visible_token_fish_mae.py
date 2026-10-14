#!/usr/bin/env python3
"""Actual licensed all-block gradients, preparedpixel exclusion and exact restore."""
import argparse,json,os
from pathlib import Path
import numpy as np,torch
from cfp.dinov2_identity import load_pinned_small,resized_observations
from cfp.visible_token_fish_mae import VisibleTokenFishMAE
from cfp.unlabeled_crop_pool import load_unlabeled_pool,unlabeled_pixel_batch
from cfp.masked_identity_checkpoint import sample_patch_queries
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU-only actual unlabeledcrop proof required')
    torch.set_num_threads(2);torch.manual_seed(0);torch.use_deterministic_algorithms(True);r=a.root;weights=r/'external_models/dinov2_vits14_v1/dinov2_vits14_pretrain.pth';proof=json.loads((r/'dinov2_small_inference_source_smoke_v1.json').read_text());backbone=load_pinned_small('/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2',weights,proof['imported_source_proofs']);model=VisibleTokenFishMAE(backbone).eval();banks,pool=load_unlabeled_pool(r/'unlabeled_detector_crop_fit_pool_v2');images,content=unlabeled_pixel_batch([('BT-003',0),('BT-003',1)],banks);prepared,fraction=resized_observations(images,content);q=sample_patch_queries(fraction.flatten(1),np.random.default_rng(0),.6);changed=prepared.clone()
    for row in range(2):
        for k in q[row].nonzero(as_tuple=True)[0].tolist():
            y,x=divmod(k,18);changed[row,:,y*14:(y+1)*14,x*14:(x+1)*14]+=7
    with torch.inference_mode():assert torch.equal(model.reconstruct_prepared(prepared,q),model.reconstruct_prepared(changed,q))
    visible=VisibleTokenFishMAE(load_pinned_small('/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2',weights,proof['imported_source_proofs']),'visible_target').train()
    visible.training_observations(images,content,q)['loss'].backward()
    assert visible.mask_token.grad is not None and visible.mask_token.grad.abs().sum()>0
    assert sum(p.numel() for p in visible.parameters() if p.requires_grad)==sum(p.numel() for p in model.parameters() if p.requires_grad)
    del visible
    frozen={n:p.clone() for n,p in model.backbone.named_parameters() if not p.requires_grad};model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5);observed=model.training_observations(images,content,q);loss=observed['loss'];loss.backward();gradients={n:float(p.grad.double().norm()) for n,p in model.named_parameters() if p.grad is not None};assert all(np.isfinite(v) for v in gradients.values())
    for block in range(12):assert any(v>0 for n,v in gradients.items() if n.startswith(f'backbone.blocks.{block}.'))
    for prefix in ['backbone.patch_embed.','backbone.cls_token','decoder.','pixels.']:assert any(v>0 for n,v in gradients.items() if n.startswith(prefix))
    optimizer.step();model.successful_updates+=1;assert all(torch.equal(p,frozen[n]) for n,p in model.backbone.named_parameters() if n in frozen)
    saved=a.output.with_suffix('.pt');torch.save(dict(model=model.state_dict(),pretrained_sha256=sha256(weights)),saved);fresh=VisibleTokenFishMAE(load_pinned_small('/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2',weights,proof['imported_source_proofs']));fresh.load_state_dict(torch.load(saved,map_location='cpu',weights_only=False)['model'],strict=True);assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in model.state_dict().items());model.eval();fresh.eval()
    with torch.inference_mode():assert torch.equal(model.encoder_current_only(images,content),fresh.encoder_current_only(images,content))
    model.decoder.forward=lambda *args,**kw:(_ for _ in ()).throw(AssertionError('Decoderonline'))
    with torch.inference_mode():assert model.encoder_current_only(images,content).shape==(2,384)
    result=dict(status='VERIFIED_ACTUAL_ALL_BLOCK_VISIBLE_TOKEN_MAE_GRADIENT_PIXEL_EXCLUSION_RESTORE',actual_unlabeled_detector_fit_crops=2,fit_video='BT-003',pool_manifest_sha256=sha256(r/'unlabeled_detector_crop_fit_pool_v2/manifest.json'),pretrained_sha256=sha256(weights),masked_current_prepared_patches=int(q.sum()),visible_encoder_patches=observed['visible_encoder_patch_counts'].tolist(),all12encoderblock_gradients=True,patch_embed_and_CLS_gradients=True,pretrained_position_and_unused_masktoken_frozen=True,hidden_prepared_pixel_change_cannot_change_prediction=True,equal_active_visible_query_marker_verified=True,fullmodel_keys_exact_restore=len(model.state_dict()),checkpoint_bytes=saved.stat().st_size,checkpoint_sha256=sha256(saved),decoder_excluded_online=True,loss=float(loss.detach()),gradient_norms=gradients,sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,'cfp/visible_token_fish_mae.py','cfp/unlabeled_crop_pool.py','cfp/fish_pixel_reconstruction.py','cfp/dinov2_identity.py','cfp/masked_identity_checkpoint.py']),scope='OneactualCPUoptimizer step on detectorview4fit-pool, noidentitylabelvalues/GTcrop rows. Pixels hiddenafterregisteredresize aredeleted beforeanyattention, not cachedgloballymixedtokens. All12blocks/patch_embed/CLS/norm train; fixedposition/unusedoriginalmasktoken so nearfull notallparameterfinetuning. OwnedDINOwarmstartedvisible-token MAE, noEMA/teacher/future/FishMAEunlicensedcode/officialreproduction. Notfull8epochpretraining, identitysupervision, nativeMOTgain/novelty/finalgoal.')
    write_json(a.output,result);print(json.dumps({k:result[k] for k in ['status','fullmodel_keys_exact_restore','checkpoint_bytes','masked_current_prepared_patches','visible_encoder_patches']}))

if __name__=='__main__':main()
