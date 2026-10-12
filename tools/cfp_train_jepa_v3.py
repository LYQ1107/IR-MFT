#!/usr/bin/env python3
"""Round3: actual latent candidate discrimination with matched no-EMA controls."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from cfp.fish_jepa import FishJEPAStudent, update_ema_target
from cfp.recoverable_identity import RecoverableIdentityState
from cfp.recovery_training_data import load_recovery_data, sampling_groups, sample_indices
from cfp.jepa_training_data import teacher_index_maps, make_jepa_batch
from cfp.provenance import sha256, snapshot_sources, write_json


def build_model(spec, seed, rgb_encoder):
    reliability = None
    if spec.get('ris'):
        path = Path(spec['ris_checkpoints'][str(seed)])
        metadata = json.loads(path.with_name('manifest.json').read_text())
        if metadata['status'] != 'COMPLETE' or sha256(path) != metadata['checkpoint_sha256'] or metadata['rgb_encoder']['checkpoint_sha256'] != rgb_encoder['checkpoint_sha256']:
            raise ValueError('Frozen RIS latent coordinates differ')
        reliability = RecoverableIdentityState(metadata['config']['hidden'], metadata['config'].get('semantic', False))
        reliability.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True)
        reliability.requires_grad_(False).eval()
    return FishJEPAStudent(spec.get('architecture', 'gru'), spec.get('latent', 128), reliability, spec.get('freeze_encoder', False))


def parameter_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def diagnostic(model, teacher, rows, features, labels, maps, device, spec, seed):
    result = {}
    model.eval()
    teacher.eval()
    for view, corrupt in [('clean', False), ('actual', False), ('actual_corrupted', True)]:
        rng = np.random.default_rng(9000 + seed)
        counts = dict(queries=0, correct=0, mean_correct=0, N01=0, N10=0)
        cosine, raw_cosine, margins, latent_points, raw_points = [], [], [], [], []
        with torch.inference_mode():
            for start in range(0, len(rows), 128):
                batch = make_jepa_batch(rows[start:start+128], features, labels, maps, device, rng, 'actual' if corrupt else view,
                                        masking=corrupt, corruption=corrupt, future_gap=spec.get('future_gap', 0))
                past, times, valid, query, candidates, candidate_valid, positive, _, _, target, _, _, mask = batch
                prototype, available, prediction, _ = model.training_details(past, times, valid, query, mask if corrupt else None)
                target_latent = (teacher if spec.get('ema', False) else model.token_encoder)(target)
                score = torch.einsum('bd,bkd->bk', prototype, F.normalize(candidates, dim=-1)).masked_fill(~candidate_valid, -2.)
                mean = F.normalize((F.normalize(past, dim=-1) * valid[..., None]).sum(1), dim=-1)
                reference = torch.einsum('bd,bkd->bk', mean, F.normalize(candidates, dim=-1)).masked_fill(~candidate_valid, -2.)
                good, old = score.argmax(1) == positive, reference.argmax(1) == positive
                counts['queries'] += len(past)
                counts['correct'] += int(good.sum())
                counts['mean_correct'] += int(old.sum())
                counts['N01'] += int((good & ~old).sum())
                counts['N10'] += int((~good & old).sum())
                cosine.extend(F.cosine_similarity(prediction.float(), target_latent.float()).cpu().tolist())
                raw_cosine.extend(F.cosine_similarity(prototype.float(), target.float()).cpu().tolist())
                negatives = score.clone()
                negatives[torch.arange(len(score), device=device), positive] = -2.
                margins.extend((score[torch.arange(len(score), device=device), positive] - negatives.max(1).values).cpu().tolist())
                latent_points.append(prediction.cpu().numpy())
                raw_points.append(prototype.cpu().numpy())
        latent, raw = np.concatenate(latent_points), np.concatenate(raw_points)
        counts.update(rank1=counts['correct']/counts['queries'], mean_rank1=counts['mean_correct']/counts['queries'],
                      net_corrections=counts['N01']-counts['N10'], teacher_prediction_cosine=float(np.mean(cosine)),
                      raw_target_cosine=float(np.mean(raw_cosine)), positive_negative_margin=float(np.mean(margins)),
                      predicted_latent_variance_sum=float(latent.var(0).sum()), raw_prediction_variance_sum=float(raw.var(0).sum()),
                      mean_latent_norm=float(np.linalg.norm(latent.mean(0))),
                      scope='Retrospective heldout-train SN015 query rank/teacher alignment, not online identity correction counts; synthetic differs from actual.')
        result[view] = counts
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN' or sha256(args.queries / 'manifest.json') != config['query_manifest_sha256']:
        raise ValueError('Frozen train data/recipe required')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Immutable model output')
    spec = config['variants'][args.variant]
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rows, features, labels, data = load_recovery_data(args.queries, args.seed)
    maps = teacher_index_maps(args.queries, labels)
    fit = [row for row in rows if row['sequence'] in config['fit_videos']]
    calibration = [row for row in rows if row['sequence'] in config['calibration_videos']]
    groups = sampling_groups(fit, config['fit_videos'])
    steps = math.ceil(len(fit) / config['batch_size'])
    device = torch.device('cuda')
    model = build_model(spec, args.seed, data['rgb_encoder']).to(device)
    teacher = copy.deepcopy(model.token_encoder).requires_grad_(False).to(device).eval()
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    sources = [__file__, 'cfp/fish_jepa.py', 'cfp/jepa_training_data.py', 'cfp/recovery_training_data.py', 'cfp/predictive.py']
    guard = dict(config_sha256=sha256(args.config), query_manifest_sha256=sha256(args.queries/'manifest.json'),
                 feature_manifest_sha256=data['feature_manifest_sha256'], variant=args.variant, seed=args.seed,
                 torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), sources={str(path):sha256(path) for path in sources})
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(status='RUNNING', guard=guard, config=config, spec=spec, rgb_encoder=data['rgb_encoder'], feature_root=data['feature_root'],
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), source_snapshots=snapshot_sources(args.output,sources+[args.config]),
                  student_parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  target_parameters=sum(p.numel() for p in teacher.parameters()) if spec.get('ema', False) else 0,
                  allocated_target_parameters=sum(p.numel() for p in teacher.parameters()), fit_queries=len(fit), calibration_queries=len(calibration),
                  steps_per_epoch=steps, epochs=[], ema_updates=0, teacher_gradient_count=0,
                  scope='Supervised identity-aligned lightweight track-level JEPA; cached frozen RGB features, EMA target only training. Not unsupervised video pretraining. Decoded512dim prototype preserves candidate coordinates; teacher has no inference module.')
    first_epoch = 1
    if args.resume:
        old = json.loads((args.output/'manifest.json').read_text())
        saved = torch.load(args.resume,map_location=device,weights_only=False)
        if old['status']=='COMPLETE' or saved['guard']!=guard or saved['epoch']>=config['epochs']:
            raise ValueError('Resume input/source/runtime guard mismatch')
        model.load_state_dict(saved['model']);teacher.load_state_dict(saved['teacher']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']]);random.setstate(saved['python_rng'])
        record['epochs']=saved['epochs'];record['ema_updates']=saved['ema_updates'];first_epoch=saved['epoch']+1
    write_json(args.output/'manifest.json',record)
    for epoch in range(first_epoch,config['epochs']+1):
        model.train()
        if model.reliability is not None:model.reliability.eval()
        sums=dict(loss=0.,latent=0.,raw_alignment=0.,masked=0.,identity=0.,latent_identity=0.,variance=0.)
        started=time.perf_counter()
        query_order=hashlib.sha256()
        for step in range(steps):
            positions=sample_indices(groups,rng,config['batch_size'])
            query_order.update(np.asarray(positions,np.int64).tobytes())
            step_seed=args.seed*10000000+(epoch-1)*steps+step
            batch=make_jepa_batch([fit[i] for i in positions],features,labels,maps,device,np.random.default_rng(step_seed+200000),
                                  masking=spec.get('mask',False),corruption=spec.get('corruption',False),future_gap=spec.get('future_gap',0),
                                  view_rng=np.random.default_rng(step_seed+100000),mask_rng=np.random.default_rng(step_seed+300000))
            past,times,valid,query,candidates,candidate_valid,positive,_,_,target,mask_targets,mask_valid,context_mask=batch
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=config['amp']):
                prototype,available,prediction,masked_prediction=model.training_details(past,times,valid,query,context_mask)
                with torch.no_grad():
                    target_latent=teacher(target) if spec.get('ema',False) else model.token_encoder(target)
                    mask_latent=(teacher if spec.get('ema',False) else model.token_encoder)(mask_targets)
                latent=(1-F.cosine_similarity(prediction.float(),target_latent.float())).mean()
                raw=(1-F.cosine_similarity(prototype.float(),target.float())).mean()
                prior_visible=(valid & ~context_mask).long().cumsum(1) - (valid & ~context_mask).long()
                selected=context_mask & mask_valid & (prior_visible>0)
                masked=(1-F.cosine_similarity(masked_prediction[selected].float(),mask_latent[selected].float())).mean() if selected.any() else prediction.sum()*0.
                logits=(torch.einsum('bd,bkd->bk',prototype,F.normalize(candidates,dim=-1))/.1).masked_fill(~candidate_valid,-1e4)
                identity=F.cross_entropy(logits.float(),positive)
                with torch.no_grad():
                    candidate_latents=(teacher if spec.get('ema',False) else model.token_encoder)(candidates)
                latent_logits=(torch.einsum('bd,bkd->bk',prediction,candidate_latents)/.1).masked_fill(~candidate_valid,-1e4)
                latent_identity=F.cross_entropy(latent_logits.float(),positive)
                variance=F.relu(.02-torch.sqrt(prediction.float().var(0,unbiased=False)+1e-6)).mean()
                loss=identity if spec.get('ordinary',False) else .5*latent+.5*raw+.05*variance
                loss=loss+spec.get('latent_identity_weight',0.)*latent_identity
                if not spec.get('ordinary',False):
                    if spec.get('mask',False):loss=loss+.25*masked
                    if spec.get('identity',False):loss=loss+identity
            if not torch.isfinite(loss):raise ValueError('Nonfinite JEPA loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if spec.get('ema',False) and scaler.get_scale()>=old_scale:
                progress=((epoch-1)*steps+step)/max(config['epochs']*steps-1,1)
                momentum=config['ema_start']+(config['ema_end']-config['ema_start'])*progress
                update_ema_target(teacher,model.token_encoder,momentum);record['ema_updates']+=1
            if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Target teacher received gradients')
            for key,value in [('loss',loss),('latent',latent),('raw_alignment',raw),('masked',masked),('identity',identity),('latent_identity',latent_identity),('variance',variance)]:sums[key]+=float(value.detach())
        record['epochs'].append(dict(epoch=epoch,**{k:v/steps for k,v in sums.items()},seconds=time.perf_counter()-started,ema_updates=record['ema_updates'],query_order_sha256=query_order.hexdigest()))
        saved=dict(epoch=epoch,guard=guard,model=model.state_dict(),teacher=teacher.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                   numpy_rng=rng.bit_generator.state,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),epochs=record['epochs'],ema_updates=record['ema_updates'])
        temp=args.output/'last.pt.tmp';torch.save(saved,temp);temp.replace(args.output/'last.pt');write_json(args.output/'manifest.json',record)
        print('Epoch',epoch,record['epochs'][-1],flush=True)
        if shutil.disk_usage(args.queries).free<40*2**30:raise OSError('40GiB reserve after completed checkpoint')
    shutil.copyfile(args.output/'last.pt',args.output/'epoch8.pt')
    # Separate student-only export: no EMA target or optimizer tensors can enter runtime.
    inference=args.output/'inference_epoch8.pt'
    torch.save(dict(epoch=8,model=model.state_dict(),spec=spec,guard=guard,teacher_removed=True),inference)
    record.update(status='COMPLETE',checkpoint_sha256=sha256(args.output/'epoch8.pt'),inference_checkpoint_sha256=sha256(inference),
                  teacher_parameter_sha256=parameter_fingerprint(teacher),diagnostics=diagnostic(model,teacher,calibration,features,labels,maps,device,spec,args.seed),
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),inference_teacher_removed=True)
    write_json(args.output/'manifest.json',record)
    print('Complete',args.variant,args.output,flush=True)


if __name__=='__main__':main()
