"""Resumable full-Train loop for a future source-pinned native study controller.

No command-line entry point or GPU launch: the release/inference/evaluation
controller must be completed separately before this becomes a formal fit.
"""
import hashlib
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from tools.native_pyramid_fit import optimizer_step
from tools.native_pyramid_train_data import address_bytes, detach_state, epoch_clips
from tools.run_pseudo_ir_comparison import dump, sha256


def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def guard_storage(work, cap_bytes=64 * 2**30, reserve_bytes=40 * 2**30):
    """Count work files without following symlinks or making another cache."""
    total = 0
    for folder, _, names in os.walk(work, followlinks=False):
        for name in names:
            try:
                total += (Path(folder) / name).lstat().st_size
            except FileNotFoundError:
                # Another healthy pipeline may atomically replace its latest.
                continue
    if total > cap_bytes or shutil.disk_usage(work).free < reserve_bytes:
        raise RuntimeError('Native study storage cap or independent free-space reserve failed')
    return dict(work_file_bytes=total, work_cap_bytes=cap_bytes, reserve_bytes=reserve_bytes,
                free_bytes=shutil.disk_usage(work).free)


def prefix_checkpoint(provider):
    if provider.epoch is None:
        return None
    return dict(epoch=provider.epoch, delivered_clips=provider.prepared,
                actual_address_sha256=provider.digest.hexdigest())


def restore_delivered_prefix(provider, prefix):
    """Restore a hash-verified historical delivery; do not pretend to read it again."""
    provider.begin_epoch(prefix['epoch'])
    count = prefix['delivered_clips']
    if not isinstance(count, int) or count < 0:
        raise ValueError('Invalid resumed Train clip count')
    for _ in range(count):
        clip = next(provider.expected_next, None)
        if clip is None:
            raise ValueError('Resumed prefix exceeds the complete Train epoch')
        provider.digest.update(address_bytes(clip))
        provider.used[clip.sequence].extend(clip.frames)
        provider.prepared += 1
    if provider.digest.hexdigest() != prefix['actual_address_sha256']:
        raise ValueError('Resumed Train address digest disagrees with the fixed schedule')
    return dict(resumed_previously_delivered_clips=count,
                resumed_prefix_sha256=provider.digest.hexdigest(), reread_images=0)


def validate_memory_at_prefix(state, previous_gray, lengths, epoch, delivered_clips):
    if state is None:
        return  # Quality mode, memory removal, or a genuinely unavailable last cue.
    clips = list(epoch_clips(lengths, epoch))
    if not 1 <= delivered_clips <= len(clips):
        raise ValueError('Past temporal memory has no delivered prefix')
    last = clips[delivered_clips - 1]
    if (state.sequence != last.sequence or state.frame != last.frames[-1] or
            previous_gray is None or previous_gray.dtype != np.uint8):
        raise ValueError('Saved temporal state does not belong to the last delivered RGB frame')
    if any(v.requires_grad for v in state.latent + state.covariance):
        raise ValueError('Past state must be detached before checkpointing')


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if temporary.exists():
        # Preserve an interrupted write for inspection; never load it as latest.
        temporary.rename(temporary.with_suffix(temporary.suffix + f'.interrupted_{time.time_ns()}'))
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_training_to_epoch(module, encoder, head, optimizer, provider, config, metadata,
                          config_path, output, through_epoch, work_root):
    """Six-arm orchestration must call all fits at the same Dev-authorized stage."""
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    if through_epoch not in config['budget_and_Dev']['stages']:
        raise ValueError('Unregistered full-Train stage')
    if metadata['study_sha256'] != sha256(config_path) or metadata['seed'] != 0:
        raise ValueError('Fit is not tied to the fixed seed0 study')
    expected_params = config['architecture'][module.mode + '_parameters']
    if sum(p.numel() for p in module.parameters()) != expected_params:
        raise ValueError('Unexpected auxiliary parameter budget')
    if encoder.training or head.training or any(p.requires_grad for p in (*encoder.parameters(), *head.parameters())):
        raise ValueError('Only auxiliary parameters may be trainable')
    if next(module.parameters()).is_cuda and (not torch.are_deterministic_algorithms_enabled() or
            not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark):
        raise ValueError('New native fits require the preregistered deterministic GPU settings')
    def verify_sources():
        if sha256(config_path) != metadata['study_sha256']:
            raise RuntimeError('Study changed during fitting')
        if any(sha256(root / name) != digest for name, digest in metadata['source_hashes'].items()):
            raise RuntimeError('Native fit implementation changed during fitting')
    verify_sources()
    output.mkdir(parents=True, exist_ok=True)
    storage = config['storage']
    def guard():
        return guard_storage(work_root, storage['work_cap_GiB'] * 2**30,
                             storage['minimum_independent_filesystem_free_GiB'] * 2**30)
    guard()
    frozen_start = dict(encoder=state_sha256(encoder.state_dict()), head=state_sha256(head.state_dict()))
    device = next(module.parameters()).device
    latest = output / 'latest.pt'
    histories, completed, position, actual_updates = [], 0, 0, 0
    state, previous_gray, prefix, accum, count = None, None, None, {}, 0
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['metadata'] != metadata or saved['frozen_RGB_start'] != frozen_start:
            raise ValueError('Resume lineage or frozen RGB state disagrees')
        module.load_state_dict(saved['module'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        histories, completed = saved['history'], saved['completed_epochs']
        position, actual_updates = saved['delivered_clips_total'], saved['actual_optimizer_updates']
        prefix, accum, count = saved['prefix'], saved['epoch_sums'], saved['epoch_clips']
        state, previous_gray = detach_state(saved['temporal_state'], device), saved['previous_RGB_gray']
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda': torch.cuda.set_rng_state(saved['CUDA_rng'], device)
        if prefix is not None:
            if prefix['epoch'] != completed + 1 or count != prefix['delivered_clips']:
                raise ValueError('Resume epoch counters disagree')
            restore_delivered_prefix(provider, prefix)
            validate_memory_at_prefix(state, previous_gray, provider.lengths, prefix['epoch'], count)
    if completed > through_epoch or (prefix is not None and completed >= through_epoch):
        raise ValueError('Never reinterpret a later fit as an earlier budget')
    clips_per_epoch = config['Train_delivery']['clips_per_epoch']
    if position != completed * clips_per_epoch + count or not 0 <= actual_updates <= position:
        raise ValueError('Resumed delivery and optimizer counters disagree')
    def save_latest():
        guard()
        verify_sources()
        atomic_torch_save(latest, dict(metadata=metadata, module=module.state_dict(),
            optimizer=optimizer.state_dict(), history=histories, completed_epochs=completed,
            delivered_clips_total=position, actual_optimizer_updates=actual_updates,
            prefix=prefix_checkpoint(provider), epoch_sums=accum, epoch_clips=count,
            temporal_state=detach_state(state, 'cpu'), previous_RGB_gray=previous_gray,
            frozen_RGB_start=frozen_start, torch_rng=torch.get_rng_state(),
            CUDA_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))
    started = time.monotonic()
    try:
        for epoch in range(completed + 1, through_epoch + 1):
            if provider.epoch is None:
                provider.begin_epoch(epoch)
                state, previous_gray, accum, count = None, None, {}, 0
            if provider.epoch != epoch:
                raise ValueError('Provider and fit epochs diverged')
            clips = list(epoch_clips(provider.lengths, epoch))
            for clip in clips[provider.prepared:]:
                batch = provider.prepare(clip, metadata['cue'])
                row, state, previous_gray = optimizer_step(module, encoder, head, batch, optimizer,
                    config, position, state, previous_gray)
                if position == 0 and not row['initial_full_RGB_exact']:
                    raise RuntimeError('Zero-initialized full detector does not equal fresh RGB')
                count += 1; position += 1; actual_updates += int(row['optimizer_updated'])
                for key in ['loss', 'full_YOLOX', 'box', 'objectness', 'classification', 'reliable_RGB']:
                    accum[key] = accum.get(key, 0.) + row[key]
                if count % 32 == 0:
                    print(dict(fit=metadata['fit_id'], epoch=epoch, clip=count, clips=len(clips),
                        loss=accum['loss']/count, actual_updates=actual_updates,
                        seconds=time.monotonic()-started), flush=True)
                if count % 128 == 0:
                    validate_memory_at_prefix(state, previous_gray, provider.lengths, epoch, count)
                    save_latest()
            proof = provider.finish_epoch()
            frozen_now = dict(encoder=state_sha256(encoder.state_dict()), head=state_sha256(head.state_dict()))
            if frozen_now != frozen_start:
                raise RuntimeError('Original RGB parameters or BatchNorm changed')
            history = dict(epoch=epoch, delivery=proof, mean_clip_losses={k: v/count for k, v in accum.items()},
                           delivered_clips_total=position, actual_optimizer_updates=actual_updates,
                           frozen_RGB_exact_unchanged=True)
            histories.append(history)
            snapshot = output / f'epoch_{epoch:02d}.pt'
            if snapshot.exists():
                old = torch.load(snapshot, map_location='cpu', weights_only=False)
                if (old['metadata'] != metadata or state_sha256(old['module']) != state_sha256(module.state_dict()) or
                        old['history'] != histories):
                    raise RuntimeError('A recovered epoch differs from its preserved snapshot')
            else:
                guard()
                atomic_torch_save(snapshot, dict(metadata=metadata, module=module.state_dict(),
                    completed_epochs=epoch, history=list(histories), source_RGB_only_by_reference=True))
            completed, state, previous_gray, accum, count = epoch, None, None, {}, 0
            save_latest()
            dump(output / 'progress.json', dict(status='COMPLETE_REGISTERED_TRAIN_STAGE' if epoch == through_epoch else 'FITTING',
                completed_full_Train_epochs=epoch, registered_through_epoch=through_epoch,
                delivered_clips_total=position, actual_optimizer_updates=actual_updates,
                history=histories, metadata=metadata, frozen_RGB_exact_unchanged=True,
                checkpoint_sha256=sha256(snapshot), storage=guard(),
                no_new_image_or_FPN_disk_cache=True, goal_met=False, supported_innovations=0))
        verify_sources()
        return histories
    except Exception as error:
        dump(output / f'failure_{time.time_ns()}.json', dict(error=repr(error),
            delivered_clips_total=position, actual_optimizer_updates=actual_updates,
            complete_epochs=completed, latest_preserved=latest.exists(), metadata=metadata))
        raise
