#!/usr/bin/env python3
"""Fit-only unlabeled visible-token pretraining; atomic actual optimizer/RNG resume."""
import argparse,hashlib,json,os,random,shutil,subprocess,time
from pathlib import Path
import numpy as np,torch
from cfp.dinov2_identity import load_pinned_small,resized_observations
from cfp.visible_token_fish_mae import VisibleTokenFishMAE
from cfp.unlabeled_crop_pool import load_unlabeled_pool,unlabeled_pixel_batch
from cfp.dinov2_fish_data import augment_observations
from cfp.masked_identity_checkpoint import sample_patch_queries
from cfp.provenance import sha256,snapshot_sources,write_json


def atomic_checkpoint(path,payload):
    temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)


def main():
    ap=argparse.ArgumentParser()
    for name in ['root','config','pool','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--storage-root',type=Path,
                    help='Filesystem used for new checkpoints/scratch. Input assets under --root remain immutable; both mounts must retain the 40 GiB reserve.')
    ap.add_argument('--mode',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--stop-after-epoch',type=int);a=ap.parse_args();c=json.loads(a.config.read_text());gpu=os.environ.get('CUDA_VISIBLE_DEVICES');storage_root=a.storage_root or a.root
    if c['state']!='FROZEN' or a.mode not in c['modes'] or a.seed not in c['seed_plan'] or a.output.exists() and not a.resume or gpu not in ('0','1','2','3') or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8' or a.output.resolve().is_relative_to(Path.cwd().resolve()):raise ValueError('New registered outsideGit fit oridenticalresume on actualidle approvedGPU0-3 anddeterministicworkspace required')
    if a.stop_after_epoch is not None and not 1<=a.stop_after_epoch<c['epochs']:raise ValueError('Actualdeliberate termination beforefinalepoch required')
    values=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    memory,total,util=map(int,(x.strip() for x in values.split(',')))
    overlap=os.environ.get('CFP_ALLOW_GPU_OVERLAP')=='1'
    if overlap:
        # Explicitly authorized shared-GPU recovery: the allocator is capped
        # below, while this launch-time check preserves headroom for the other
        # process and rejects unexpectedly small/nonstandard devices.
        reserve=int(os.environ.get('CFP_GPU_OVERLAP_RESERVE_MB','4096'))
        budget=int(os.environ.get('CFP_GPU_OVERLAP_BUDGET_MB','16384'))
        if total<40000 or memory+budget+reserve>total:raise ValueError('Insufficientshared-GPUheadroom')
    elif memory>64 or util:raise ValueError('Assigned GPU notactuallyidle; preserveotherjobs')
    overlap_guard=dict(enabled=overlap,launch_memory_used_mb=memory,device_total_mb=total,launch_utilization_percent=util,allocator_budget_mb=(budget if overlap else None),reserve_mb=(reserve if overlap else None))
    # The input/legacy asset mount and the new-output mount are both guarded.
    # A separate output mount avoids consuming the protected 40 GiB reserve while
    # retaining atomic resumable checkpoints for this run.
    if shutil.disk_usage(a.root).free<40*2**30 or shutil.disk_usage(storage_root).free<40*2**30:raise OSError('Protected40GiBreserve')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    banks,pool=load_unlabeled_pool(a.pool)
    if set(pool['fit_videos'])!=set(c['fit_videos']) or set(pool['fit_videos'])&set(c['excluded_videos']):raise ValueError('Noheldout pixels or identitylabels inpretraining')
    weights=a.root/'external_models/dinov2_vits14_v1/dinov2_vits14_pretrain.pth';proof=json.loads((a.root/'dinov2_small_inference_source_smoke_v1.json').read_text());backbone=load_pinned_small('/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2',weights,proof['imported_source_proofs']);model=VisibleTokenFishMAE(backbone,a.mode);fixed={n:p.detach().clone() for n,p in model.backbone.named_parameters() if not p.requires_grad}
    torch.cuda.set_per_process_memory_fraction(.4);model.cuda();optimizer=torch.optim.AdamW([dict(params=[p for p in model.backbone.parameters() if p.requires_grad],lr=c['backbone_lr']),dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')],lr=c['decoder_lr'])],weight_decay=c['weight_decay']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,c['epochs']);scaler=torch.cuda.amp.GradScaler();sampler=np.random.default_rng(a.seed);augmentation=np.random.default_rng(a.seed+2000);mask_rng=np.random.default_rng(a.seed+3000)
    sources=[__file__,'cfp/visible_token_fish_mae.py','cfp/unlabeled_crop_pool.py','cfp/dinov2_fish_data.py','cfp/dinov2_identity.py','cfp/local_identity_encoder.py','cfp/masked_identity_checkpoint.py','cfp/fish_pixel_reconstruction.py',str(a.config)];guard=dict(mode=a.mode,seed=a.seed,config_sha256=sha256(a.config),pool_manifest_sha256=sha256(a.pool/'manifest.json'),pretrained_sha256=sha256(weights),source_sha256={p:sha256(p) for p in sources},imported_source_proofs=proof['imported_source_proofs'],physical_gpu=gpu,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],storage_root=str(storage_root),input_asset_root=str(a.root),gpu_overlap=overlap_guard);report=dict(status='RUNNING',pid=os.getpid(),guard=guard,config=c,epochs=[],parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),scope=c['scope'],physical_ID_values_used=False,GT_crop_values_used=False,storage_policy='Input asset mount and new-output mount each retain >=40GiB; output scratch is outside the protected asset mount when storage_root is supplied.')
    start=0;last=a.output/'.scratch-last.pt'
    if a.resume:
        if json.loads((a.output/'manifest.json').read_text())['status']!='CHECKPOINTED_EARLY_STOP':raise ValueError('Onlyregistered incomplete deliberatecheckpoint mayresume; completedassets immutable')
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['guard']!=guard:raise ValueError('Exactsame data/source/runtime/physicalGPU required')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler']);sampler.bit_generator.state=saved['rng_sampler'];augmentation.bit_generator.state=saved['rng_augmentation'];mask_rng.bit_generator.state=saved['rng_mask'];random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state_all(saved['rng_cuda']);start=saved['epoch'];report=saved['report']
    else:a.output.mkdir();report['sources']=snapshot_sources(a.output/'sources',sources)
    write_json(a.output/'manifest.json',report)
    for epoch in range(start+1,c['epochs']+1):
        model.train();losses=[];skips=0;digest=hashlib.sha256();counts=[];began=time.monotonic()
        for step in range(c['steps_per_epoch']):
            video=c['fit_videos'][int(sampler.integers(len(c['fit_videos'])))];selected=sampler.integers(len(banks[video]['crop_indices']),size=c['batch_size']);observations=[(video,int(i)) for i in selected];digest.update(json.dumps(dict(observations=observations,augmentation=augmentation.bit_generator.state,mask=mask_rng.bit_generator.state),sort_keys=True).encode());images,content=unlabeled_pixel_batch(observations,banks,'cuda');images,content=augment_observations(images,content,augmentation,c['augmentation']);_,fraction=resized_observations(images,content);queries=sample_patch_queries(fraction.flatten(1),mask_rng,c['mask_ratio']);digest.update(queries.cpu().numpy().tobytes());optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():out=model.training_observations(images,content,queries)
            loss=out['loss']
            if not torch.isfinite(loss):raise ValueError('ActualRGBreconstructionloss nonfinite')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if finite and 'first_finite_gradient_norms' not in report:
                norms={n:float(p.grad.double().norm()) for n,p in model.named_parameters() if p.grad is not None};report['first_finite_gradient_norms']=norms;report['first_finite_update']=dict(epoch=epoch,step=step)
                for block in range(12):
                    if not any(value>0 for name,value in norms.items() if name.startswith(f'backbone.blocks.{block}.')):raise ValueError('Noactualgradient inoneofthetwelveblocks')
                if not any(v>0 for n,v in norms.items() if n.startswith('backbone.patch_embed.')):raise ValueError('Patchencoder notgradientfitted')
            if finite:torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            before=scaler.get_scale();scaler.step(optimizer);scaler.update();success=scaler.get_scale()>=before;skips+=int(not success);model.successful_updates+=int(success);losses.append(float(loss.detach()));counts.extend(out['visible_encoder_patch_counts'].cpu().tolist())
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Actualtrainedparameter nonfinite')
        scheduler.step()
        if not all(torch.equal(p.detach().cpu(),fixed[n]) for n,p in model.backbone.named_parameters() if n in fixed):raise ValueError('Fixedposition/unusedtoken changed')
        e=dict(epoch=epoch,loss_mean=float(np.mean(losses)),sampled_input_and_queries_sha256=digest.hexdigest(),attempted_steps=c['steps_per_epoch'],amp_overflow_skips=skips,optimizer_updates=c['steps_per_epoch']-skips,visible_encoder_patch_min_median_max=[min(counts),float(np.median(counts)),max(counts)],seconds=time.monotonic()-began);report['epochs'].append(e)
        atomic_checkpoint(last,dict(epoch=epoch,guard=guard,report=report,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng_sampler=sampler.bit_generator.state,rng_augmentation=augmentation.bit_generator.state,rng_mask=mask_rng.bit_generator.state,rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()));write_json(a.output/'manifest.json',report);print(json.dumps(e),flush=True)
        if shutil.disk_usage(a.root).free<40*2**30 or shutil.disk_usage(storage_root).free<40*2**30:raise OSError('Protected40GiBreserve')
        if a.stop_after_epoch==epoch:report.update(status='CHECKPOINTED_EARLY_STOP',stopped_after_epoch=epoch);write_json(a.output/'manifest.json',report);print(json.dumps(dict(status='CHECKPOINTED_EARLY_STOP',epoch=epoch)),flush=True);return
    final=a.output/'encoder_final.pt';atomic_checkpoint(final,dict(backbone=model.backbone.state_dict(),guard=guard,config=c,epoch=c['epochs']));report.update(status='COMPLETE_UNLABELED_VISIBLE_TOKEN_PRETRAINING',encoder_checkpoint_sha256=sha256(final),full_training_scratch_sha256=sha256(last),successful_updates=int(model.successful_updates),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),fixed_position_and_unused_token_unchanged=True);write_json(a.output/'manifest.json',report)

if __name__=='__main__':main()
