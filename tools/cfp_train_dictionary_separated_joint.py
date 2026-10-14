#!/usr/bin/env python3
"""Fit-only actual DINO set targets; no heldout model/threshold selection."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from cfp.dictionary_separated_joint import DictionarySeparatedJoint
from cfp.joint_onpolicy_data import load_query_dataset
from cfp.joint_set_augmentation import augmented_query_batch
from cfp.joint_set_supervision import balanced_set_identity_loss
from cfp.provenance import sha256, snapshot_sources, write_json


def checkpoint(path, payload):
    temporary = path.with_suffix('.tmp'); torch.save(payload, temporary); os.replace(temporary, path)


def main():
    ap = argparse.ArgumentParser()
    for name in ['queries', 'config', 'output']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--mode', required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--stop-after-epoch', type=int)
    a = ap.parse_args(); c = json.loads(a.config.read_text())
    if (c['state'] != 'FROZEN' or a.mode not in c['model_specs'] or a.seed not in c['seed_plan']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or a.output.exists() and not a.resume
            or a.output.resolve().is_relative_to(Path.cwd().resolve())):
        raise ValueError('Registered CPU mode/seed and new outsideGit output or exact resume required')
    if a.stop_after_epoch is not None and not 1 <= a.stop_after_epoch < c['epochs']:
        raise ValueError('Actual early exit must precede final epoch')
    if shutil.disk_usage(a.output.parent).free < 42*2**30:
        raise OSError('Protected reserve plus parallel margin')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    sampler = np.random.default_rng(a.seed); codes = np.random.default_rng(a.seed+1000); subset = np.random.default_rng(a.seed+2000)
    records, banks, qm = load_query_dataset(a.queries)
    if (qm['config_sha256'] != c['query_config_sha256'] or set(qm['videos']) != set(c['fit_videos'])
            or set(qm['videos']) & set(c['outer_videos']+c['inner_videos'])):
        raise ValueError('Exactly four fit-only teacher videos required')
    groups = {v: [r for r in records if r['video']==v] for v in c['fit_videos']}
    if any(not rows for rows in groups.values()):
        raise ValueError('No empty supervision or fabricated replacement data')
    spec = c['model_specs'][a.mode]
    model = DictionarySeparatedJoint(c['dimension'], c['hidden'], c['heads'], c['capacity'], **spec)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c['epochs'])
    paths = [__file__, str(a.config), 'cfp/dictionary_separated_joint.py', 'cfp/joint_identity_decoder.py',
             'cfp/centered_joint_identity.py', 'cfp/joint_onpolicy_data.py', 'cfp/joint_set_augmentation.py',
             'cfp/joint_set_supervision.py', 'cfp/association.py', 'cfp/native_local_identity.py', 'cfp/predictive.py']
    guard = dict(config_sha256=sha256(a.config), query_manifest_sha256=sha256(a.queries/'manifest.json'), mode=a.mode,
                 seed=a.seed, source_sha256={p: sha256(p) for p in paths}, torch=torch.__version__,
                 deterministic_algorithms=True, physical_devices='', fit=c['fit_videos'], inner=c['inner_videos'], outer=c['outer_videos'])
    report = dict(status='RUNNING', pid=os.getpid(), guard=guard, config=c, epochs=[], elapsed_training_seconds=0.,
        allocated_parameters=sum(p.numel() for p in model.parameters()), active_parameters=sum(p.numel() for p in trainable.values()),
        conditioning_parameters=sum(p.numel() for p in model.dictionary.parameters()),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        queries_scope=qm['scope'], scope=c['scope'], evaluations={}, heldout_data_loaded=False,
        no_physical_label_in_model_inputs=True, teacher_current_pixels_or_future_in_model_inputs=False)
    start = 0
    if a.resume:
        saved = torch.load(a.output/'last.pt', map_location='cpu', weights_only=False)
        if saved['guard'] != guard or json.loads((a.output/'manifest.json').read_text())['status']=='COMPLETE':
            raise ValueError('Only identical incomplete training may resume')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        sampler.bit_generator.state=saved['rng_sampler']; codes.bit_generator.state=saved['rng_codes']; subset.bit_generator.state=saved['rng_subset']
        random.setstate(saved['rng_python']); np.random.set_state(saved['rng_numpy']); torch.set_rng_state(saved['rng_torch'])
        start, report = saved['epoch'], saved['report']
    else:
        a.output.mkdir(); report['source_snapshots'] = snapshot_sources(a.output/'sources', paths)
    write_json(a.output/'manifest.json', report)
    for epoch in range(start+1, c['epochs']+1):
        if any(sha256(p) != h for p, h in guard['source_sha256'].items()):
            raise ValueError('Active source drift')
        model.train(); digest=hashlib.sha256(); losses=[]; parts_sum=Counter(); augmentations=Counter(); began=time.monotonic()
        for step in range(c['steps_per_epoch']):
            video = c['fit_videos'][int(sampler.integers(len(groups)))]; row=groups[video][int(sampler.integers(len(groups[video])))]
            inputs, targets, metadata = augmented_query_batch(row, banks, codes, subset, c)
            digest.update(json.dumps(dict(row=row, metadata=metadata), sort_keys=True).encode())
            for key, value in inputs.items():
                digest.update(key.encode()); digest.update(value.numpy().tobytes())
            augmentations[metadata['kind']] += 1
            output = model(**inputs)
            loss, parts = balanced_set_identity_loss(output['logits'], targets, output['legal'], c['consistency_weight'])
            if not torch.isfinite(loss) or not parts['supervised']:
                raise ValueError('Finite genuinely supervised sampled frame required')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable.values()):
                raise ValueError('Nonfinite actual visual/set gradients')
            if not report.get('first_gradient_norms'):
                report['first_gradient_norms']={n: float(p.grad.double().norm()) for n,p in trainable.items() if p.grad is not None}
                report['first_gradient_parameters']=sum(p.numel() for p in trainable.values() if p.grad is not None)
                if not any(v > 0 for n,v in report['first_gradient_norms'].items() if n.startswith('dictionary.')):
                    raise ValueError('Matched conditioning capacity is inactive')
            torch.nn.utils.clip_grad_norm_(trainable.values(), 5.); optimizer.step()
            losses.append(float(loss.detach())); parts_sum.update(parts)
        scheduler.step(); seconds=time.monotonic()-began; report['elapsed_training_seconds']+=seconds
        entry=dict(epoch=epoch, attempted_steps=len(losses), optimizer_updates=len(losses), mean_loss=float(np.mean(losses)),
                   objective_means={k: v/len(losses) for k,v in parts_sum.items()}, augmentation_counts=dict(augmentations),
                   sampled_input_sha256=digest.hexdigest(), seconds=seconds)
        report['epochs'].append(entry)
        checkpoint(a.output/'last.pt', dict(epoch=epoch, guard=guard, report=report, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng_sampler=sampler.bit_generator.state,
            rng_codes=codes.bit_generator.state, rng_subset=subset.bit_generator.state,
            rng_numpy=np.random.get_state(), rng_python=random.getstate(), rng_torch=torch.get_rng_state()))
        write_json(a.output/'manifest.json', report); print(json.dumps(entry), flush=True)
        if shutil.disk_usage(a.output).free < 42*2**30:
            raise OSError('Protected reserve plus parallel margin')
        if a.stop_after_epoch==epoch:
            report['status']='CHECKPOINTED_EARLY_STOP'; write_json(a.output/'manifest.json', report)
            print(json.dumps(dict(status='CHECKPOINTED_EARLY_STOP', epoch=epoch)), flush=True); return
    checkpoint(a.output/'final.pt', dict(model=model.state_dict(), guard=guard, epoch=c['epochs']))
    report.update(status='COMPLETE', checkpoint_sha256=sha256(a.output/'final.pt'), cuda_initialized=torch.cuda.is_initialized())
    if report['cuda_initialized']:
        raise ValueError('CPU fitting must not initialize any GPU')
    write_json(a.output/'manifest.json', report)


if __name__=='__main__':
    main()
