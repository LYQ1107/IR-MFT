#!/usr/bin/env python3
"""Actual idle-GPU1 FP16/deterministic all-block MAE training step before campaign."""
import argparse,gc,json,os,subprocess,time
from pathlib import Path
import numpy as np,torch
from cfp.dinov2_identity import load_pinned_small,resized_observations
from cfp.visible_token_fish_mae import VisibleTokenFishMAE
from cfp.unlabeled_crop_pool import load_unlabeled_pool,unlabeled_pixel_batch
from cfp.masked_identity_checkpoint import sample_patch_queries
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('New actual GPU1 deterministic FP16 proof required')
    mem,util=map(int,subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','))
    if mem>64 or util:raise ValueError('GPU1 isnotidle; donotinterfere')
    native=json.loads((a.root/'native_jpm_controls_firstfold_worker_v1.json').read_text())
    if native['status']!='RUNNING_NATIVE_JPM_CONTROLS':raise ValueError('Shortprobe onlywhile nativeGPU3ownerconfirmedlive andpretrainqueued')
    os.kill(native['pid'],0);pretrain=json.loads((a.root/'visible_token_fish_pretraining_all6_driver_v2.json').read_text())
    if pretrain['status']!='WAITING_BOTH_ACTUAL_NATIVE_OWNERS' or pretrain['launches']:raise ValueError('No concurrent GPU1pretraining')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.4)
    r=a.root;weights=r/'external_models/dinov2_vits14_v1/dinov2_vits14_pretrain.pth';proof=json.loads((r/'dinov2_small_inference_source_smoke_v1.json').read_text());banks,pool=load_unlabeled_pool(r/'unlabeled_detector_crop_fit_pool_v2');observations=[('BT-003',i) for i in range(16)];images,content=unlabeled_pixel_batch(observations,banks,'cuda');_,fraction=resized_observations(images,content);queries=sample_patch_queries(fraction.flatten(1),np.random.default_rng(0),.6);results=[]
    for mode in ['masked_visible_tokens','visible_target']:
        torch.manual_seed(0);model=VisibleTokenFishMAE(load_pinned_small('/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2',weights,proof['imported_source_proofs']),mode).cuda().train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5);scaler=torch.cuda.amp.GradScaler();started=time.monotonic();attempts=0
        for attempt in range(12):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():out=model.training_observations(images,content,queries)
            loss=out['loss']
            if not torch.isfinite(loss):raise ValueError('ActualCUDA reconstructionloss nonfinite')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());before=scaler.get_scale()
            if finite:torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            scaler.step(optimizer);scaler.update();attempts+=1
            if finite:break
        if not finite:raise ValueError('No finite all-block GPUgradient after realAMPbackoff')
        norms={n:float(p.grad.double().norm()) for n,p in model.named_parameters() if p.grad is not None}
        for block in range(12):assert any(v>0 for n,v in norms.items() if n.startswith(f'backbone.blocks.{block}.'))
        assert model.mask_token.grad is not None and model.mask_token.grad.abs().sum()>0
        results.append(dict(mode=mode,actual_optimizer_updates=1,actual_attempts=attempts,amp_overflow_skips=attempts-1,final_scale=scaler.get_scale(),loss=float(loss.detach()),active_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),visible_encoder_patch_counts=out['visible_encoder_patch_counts'].cpu().tolist(),all12block_gradients_verified=True,gradient_norms=norms,elapsed_seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
        del out,loss,model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()
    assert results[0]['active_parameters']==results[1]['active_parameters'];write_json(a.output,dict(status='VERIFIED_ACTUAL_GPU1_FP16_ALL_BLOCK_VISIBLE_TOKEN_MAE_STEPS',torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),physical_gpu=1,deterministic=True,cublas_workspace=':4096:8',fit_crops=16,identity_values_used=False,GT_crop_values_used=False,results=results,sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,'cfp/visible_token_fish_mae.py','cfp/unlabeled_crop_pool.py','cfp/dinov2_identity.py','cfp/masked_identity_checkpoint.py']),scope='Two actualAMP/deterministicGPUoptimizer steps on16BT003detectorfitcrops withpotentialAMPbackoff; verifiesrealCUDA raggedtoken/scatter/fullblock/activequery-marker behavior, not8epochpretraining/nativeMOT/rawFPS ornovelty. OriginalnativeGPU1ownerterminal; shortprobe onlywhileGPU3owner/pretrainwait confirmedlive.'))
    print(json.dumps([{k:e[k] for k in ['mode','actual_attempts','actual_optimizer_updates','active_parameters']} for e in results]))

if __name__=='__main__':main()
