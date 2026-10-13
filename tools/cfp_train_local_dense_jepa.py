#!/usr/bin/env python3
"""Train actual object-local JEPA and matched controls, with guarded restart."""
import argparse
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

from cfp.local_dense_data import LocalSceneBank, LocalSceneSampler
from cfp.local_dense_evaluation import evaluate_local_scenes
from cfp.local_dense_jepa import LocalDenseStudent, DenseLocalTarget
from cfp.local_dense_training import local_dense_objective
from cfp.provenance import sha256, snapshot_sources, write_json


def save_state(path, value):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def parameter_digest(parameters):
    result = hashlib.sha256()
    for name, parameter in parameters:
        result.update(name.encode())
        result.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for name in ('features', 'index', 'legacy-features', 'config', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--outer', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--local-count', type=int, required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--stop-after-epoch', type=int)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    if (config['state']!='FROZEN' or config['device']!='cpu' or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or args.mode not in config['modes'] or args.seed not in config['seed_plan']
            or args.local_count not in config['local_counts'] or args.outer not in config['videos']
            or args.output.exists() and not args.resume
            or args.output.resolve().is_relative_to(Path.cwd().resolve())):
        raise ValueError('Fixed declared recipe, registered fold/mode/seed/K, CPU-only and outsideGit immutable output required')
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch < config['epochs']:
        raise ValueError('Registered checkpoint exit before final epoch required')
    if shutil.disk_usage(args.output.parent).free < 40*2**30:
        raise OSError('40GiBreserve')
    cycle = config['videos']
    inner = cycle[(cycle.index(args.outer)+1) % len(cycle)]
    fit = [v for v in cycle if v not in (inner, args.outer)]
    torch.set_num_threads(config['threads_per_process'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_initialized():
        raise RuntimeError('This registered CPU job must never allocate CUDA resources')
    bank = LocalSceneBank(args.features, args.local_count, args.index, args.legacy_features)
    sampler = LocalSceneSampler(bank, fit, args.seed, config['identities_per_training_scene'],
        config['history_length'], config['maximum_history_age_frames'], config['minimum_past_gaps'])
    assert not any(v in (inner, args.outer) for v, identity in sampler.labels)
    student = LocalDenseStudent(len(sampler.labels), args.mode, args.local_count,
        config['dimension'], config['attention_heads'])
    teacher = DenseLocalTarget(student, 0. if args.mode in ('no_ema', 'local_contrastive') else config['ema_decay'])
    trainable = {name: p for name, p in student.named_parameters() if p.requires_grad}
    initial_student = parameter_digest(trainable.items())
    initial_teacher = parameter_digest(teacher.named_parameters())
    initial_local = {name: p.detach().clone() for name, p in student.local_encoder.named_parameters()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    mask_rng = np.random.default_rng(args.seed+2000)
    sources = [__file__, 'cfp/local_dense_jepa.py', 'cfp/local_dense_data.py',
        'cfp/local_dense_training.py', 'cfp/local_dense_evaluation.py', 'cfp/identity_evidence.py', 'cfp/provenance.py']
    guard = dict(mode=args.mode, seed=args.seed, local_count=args.local_count, outer=args.outer,
        inner=inner, fit=fit, config_sha256=sha256(args.config), feature_manifest_sha256=sha256(args.features/'manifest.json'),
        index_manifest_sha256=sha256(args.index/'manifest.json'), legacy_manifest_sha256=sha256(args.legacy_features/'manifest.json'),
        backbone_weight_sha256=bank.manifest['weight_sha256'], source_sha256={p: sha256(p) for p in sources},
        torch=torch.__version__, threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], teacher_decay=teacher.decay,
        initial_student_sha256=initial_student, initial_teacher_sha256=initial_teacher)
    report = dict(status='RUNNING', guard=guard, config=config, epochs=[], evaluations={}, training_seconds=0.,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        classifier_keys=[[v, identity] for v, identity in sampler.labels],
        allocated_student_parameters=sum(p.numel() for p in student.parameters()),
        active_student_parameters=sum(p.numel() for p in trainable.values()),
        active_parameters_by_branch={prefix: sum(p.numel() for n, p in trainable.items() if n.startswith(prefix))
            for prefix in ('local_encoder.', 'temporal.', 'time_encoder.', 'predictor.', 'prediction_',
                           'output_norm.', 'queries', 'horizon_encoder.', 'global_projection.', 'gru.', 'relation.', 'classifier.', 'logit_scale')},
        teacher_parameters=sum(p.numel() for p in teacher.parameters()),
        scope=config['scope'], fitting_invariant='Only4fit current/past crop observations and physical offline supervision. No inner/outer gradient/state/classifier update; immutable basebackbone not fitted. No teacher/GT/physicalIDs/currenttarget in historical prediction inputs.',
        registered_sampling_grid='Exact original20frame residues1/2 and native frame counts, including empty sampled scenes. This is sparse diagnostic, not all-nativeframes.')
    epoch_start = 0
    if args.resume:
        saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
        if saved['guard'] != guard or json.loads((args.output/'manifest.json').read_text())['status']=='COMPLETE':
            raise ValueError('Exact source/data/config/runtime and incomplete real checkpoint required')
        student.load_state_dict(saved['student'], strict=True)
        teacher.load_state_dict(saved['teacher'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        sampler.rng.bit_generator.state = saved['rng_sampler']
        mask_rng.bit_generator.state = saved['rng_target_mask']
        torch.set_rng_state(saved['rng_torch'])
        np.random.set_state(saved['rng_numpy'])
        random.setstate(saved['rng_python'])
        epoch_start, report = saved['epoch'], saved['report']
    else:
        args.output.mkdir(parents=True)
        report['sources'] = snapshot_sources(args.output/'sources', sources+[args.config])
    write_json(args.output/'manifest.json', report)
    try:
        for epoch in range(epoch_start+1, config['epochs']+1):
            started = time.monotonic()
            student.train()
            teacher.train(False)
            entries, sample_hasher, updates = [], hashlib.sha256(), 0
            for step in range(config['steps_per_epoch']):
                scene = sampler.sample()
                selected = mask_rng.random((len(scene['track_ids']), args.local_count)) < config['target_mask_probability']
                observed = scene['target_fraction'].numpy()>0
                for j in range(len(selected)):
                    if observed[j].any() and not (selected[j]&observed[j]).any():
                        selected[j, np.flatnonzero(observed[j])[0]] = True
                target_mask = torch.from_numpy(selected)
                sampling = dict(video=scene['video'], frame=scene['frame'], gap=scene['minimum_gap'],
                    tracks=scene['track_ids'].tolist(), history_indices=scene['history_indices'],
                    current_indices=scene['current_indices'], target_indices=scene['target_indices'], target_mask=selected.tolist())
                sample_hasher.update(json.dumps(sampling, sort_keys=True, separators=(',', ':')).encode())
                optimizer.zero_grad(set_to_none=True)
                loss, parts = local_dense_objective(student, teacher, scene, sampler.labels, target_mask,
                    config['prediction_weight'], config['relation_weight_inside_identity_objective'])
                if not torch.isfinite(loss):
                    raise ValueError('Actual local identity/prediction loss nonfinite')
                loss.backward()
                if any(p.grad is not None for p in teacher.parameters()):
                    raise ValueError('EMA teacher received a gradient')
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable.values()):
                    raise ValueError('CPU local prediction/identity gradient nonfinite')
                if epoch==1 and step==0:
                    report['first_gradient_norms'] = {n: float(p.grad.double().norm()) for n, p in trainable.items() if p.grad is not None}
                    required = []
                    if args.mode not in ('global', 'global_gru'):
                        required += ['local_encoder.']
                    if args.mode in ('full', 'no_ema', 'no_prediction', 'no_id', 'local_contrastive'):
                        required += ['predictor.']
                    if args.mode not in ('global', 'global_gru', 'local_attention'):
                        required += ['temporal.']
                    if args.mode=='global_gru':
                        required += ['gru.']
                    if any(not any(n.startswith(prefix) and value>0 for n, value in report['first_gradient_norms'].items()) for prefix in required):
                        raise ValueError('Declared actual local/temporal/predictor/GRU branch has no gradient')
                torch.nn.utils.clip_grad_norm_(trainable.values(), 5.)
                optimizer.step()
                updates += 1
                if args.mode in ('full', 'no_ema', 'no_id', 'local_contrastive'):
                    teacher.update(student)
                if not all(torch.isfinite(p).all() for p in trainable.values()):
                    raise ValueError('Updated local/prediction parameter nonfinite')
                parts['minimum_gap'] = scene['minimum_gap']
                entries.append(parts)
            scheduler.step()
            seconds = time.monotonic()-started
            report['training_seconds'] += seconds
            entry = dict(epoch=epoch, attempted_steps=len(entries), optimizer_updates=updates,
                teacher_updates=int(teacher.updates), seconds=seconds, sampled_input_sha256=sample_hasher.hexdigest(),
                metrics_mean={name: float(np.mean([row[name] for row in entries])) for name in entries[0]})
            report['epochs'].append(entry)
            save_state(args.output/'last.pt', dict(epoch=epoch, guard=guard, report=report,
                student=student.state_dict(), teacher=teacher.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), rng_sampler=sampler.rng.bit_generator.state,
                rng_target_mask=mask_rng.bit_generator.state, rng_torch=torch.get_rng_state(),
                rng_numpy=np.random.get_state(), rng_python=random.getstate()))
            write_json(args.output/'manifest.json', report)
            print(json.dumps(entry), flush=True)
            if shutil.disk_usage(args.output).free < 40*2**30:
                raise OSError('40GiBreserve')
            if args.stop_after_epoch==epoch:
                report.update(status='CHECKPOINTED_EARLY_STOP', stopped_after_epoch=epoch)
                write_json(args.output/'manifest.json', report)
                print(json.dumps(dict(status=report['status'], epoch=epoch)), flush=True)
                return
        report['student_parameters_changed'] = parameter_digest(trainable.items()) != initial_student
        report['local_encoder_l2_change'] = float(sum((p-initial_local[n]).detach().square().sum()
            for n, p in student.local_encoder.named_parameters()).sqrt())
        report['teacher_parameters_changed'] = parameter_digest(teacher.named_parameters()) != initial_teacher
        report['teacher_has_gradient'] = any(p.grad is not None for p in teacher.parameters())
        if not report['student_parameters_changed'] or report['teacher_has_gradient'] or torch.cuda.is_initialized():
            raise ValueError('No real optimization, teacher gradient or unintended GPU allocation')
        report['status'] = 'EVALUATING'
        write_json(args.output/'manifest.json', report)
        for video in (inner, args.outer):
            report['evaluations'][video] = evaluate_local_scenes(student, bank, video, config, args.output/'query_support')
            write_json(args.output/'manifest.json', report)
            print(video, 'GT-history actual-detector gap25', report['evaluations'][video]['protocol']['gt']['detector']['25']['retrieval'], flush=True)
        save_state(args.output/'final.pt', dict(student=student.state_dict(), teacher=teacher.state_dict(),
            guard=guard, config=config, classifier_keys=report['classifier_keys'], epoch=config['epochs']))
        report.update(status='COMPLETE', checkpoint_sha256=sha256(args.output/'final.pt'),
            cuda_initialized=torch.cuda.is_initialized(), actual_teacher_updates=int(teacher.updates))
        write_json(args.output/'manifest.json', report)
    except Exception as error:
        report.update(status='FAILED_ACTUAL_EXECUTION', error=str(error))
        write_json(args.output/'manifest.json', report)
        raise


if __name__ == '__main__':
    main()
