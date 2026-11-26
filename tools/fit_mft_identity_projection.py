#!/usr/bin/env python3
"""Reuse immutable Train FPNs for a small, supervised classical projection."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools.fusion_memory_features import pool_p3
from tools.mft_identity_projection import fit_fisher, paired_features, temporal_retrieval, transform
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.train_roi_identity import identities_and_boxes

SPEC = ROOT / 'configs/perceptual_fisher_memory_20261127.json'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only Train fit'
    torch.set_num_threads(2)
    started = time.monotonic()
    plan_path = ROOT / 'configs/perceptual_mft_20261122.json'
    plan = json.loads(plan_path.read_text())
    protocol_path = ROOT / plan['protocol']
    protocol = json.loads(protocol_path.read_text())
    spec = json.loads(SPEC.read_text())
    work = Path(plan['work_root'])
    base = work / 'fisher_memory_20261127'
    base.mkdir(exist_ok=True)
    assert not (base / 'fit_complete.json').exists(), 'Never overwrite completed fit'
    assert not (base / 'fit_registered.json').exists(), 'Preserve prior attempt before retry'
    cache_config = ROOT / 'configs/perceptual_correlation_repair_20261123.json'
    calibration = work / 'correlation_cache/calibration.json'
    source_files = [SPEC, Path(__file__), ROOT / 'tools/mft_identity_projection.py',
        ROOT / 'tools/fusion_memory_features.py', ROOT / 'tools/train_roi_identity.py',
        ROOT / 'tools/build_correlation_cache.py', plan_path, protocol_path, cache_config, calibration]
    registration = dict(status='REGISTERED_BEFORE_CPU_FISHER_FIT',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes={str(p): sha256(p) for p in source_files},
        fit_split='train', Dev_rows=0, Test_rows=0, GPU_work=False,
        fixed_seed=0, sampled_Train_frames_per_video=192)
    dump(base / 'fit_registered.json', registration)
    arrays = {k: [] for k in ['rgb', 'ir', 'video', 'identity', 'frame']}
    sources, anchors = {}, {}
    for video, sequence in enumerate(protocol['train']):
        folder = work / 'correlation_cache' / sequence / 'features'
        meta_path = folder / 'complete.json'
        meta = json.loads(meta_path.read_text())
        assert meta['status'] == 'COMPLETE_TRAIN_FROZEN_FPN'
        assert meta['frames'] == spec['fit']['Train_frames_per_video']
        assert meta['config_sha256'] == sha256(cache_config)
        assert meta['calibration_sha256'] == sha256(calibration)
        assert meta['source_RGB_sha256'] == spec['source_RGB_sha256']
        assert meta['source_sha256'] == sha256(ROOT / 'tools/build_correlation_cache.py')
        assert meta['Dev_rows'] == meta['Test_rows'] == 0
        gt_path = Path(plan['evaluation']['GT_root']) / 'train' / sequence / 'gt/gt.txt'
        assert sha256(gt_path) == meta['GT_sha256'] == protocol['derived_gt_sha256']['train/'+sequence]
        gt = np.loadtxt(gt_path, delimiter=',', ndmin=2)
        targets_path = folder / 'targets.npy'
        assert sha256(targets_path) == meta['targets']['sha256']
        targets = np.load(targets_path, mmap_mode='r')
        maps, map_hashes = {}, {}
        for cue in ['rgb', 'ir']:
            path = folder / (cue+'_P3.npy')
            actual_hash = sha256(path)
            assert actual_hash == meta['outputs'][cue+'_P3']['sha256']
            maps[cue] = np.load(path, mmap_mode='r')
            assert maps[cue].shape == (192, 320, 60, 108) and maps[cue].dtype == np.float16
            map_hashes[cue] = actual_hash
        anchors[sequence] = dict(frame=int(meta['frame_ids'][0]),
            P3_sha256={cue: hashlib.sha256(np.asarray(maps[cue][0]).tobytes()).hexdigest() for cue in maps},
            cache_metadata_sha256=sha256(meta_path), shape=[1, 320, 60, 108])
        count = 0
        for i, frame in enumerate(meta['frame_ids']):
            identity, boxes = identities_and_boxes(gt, frame, meta['ratios'][i], targets[i])
            if not len(identity):
                continue
            for cue in maps:
                # A single current map, discarded after ROI pooling; no FPN copy bank.
                feature = torch.from_numpy(np.asarray(maps[cue][i], dtype=np.float32)[None])
                values, valid = pool_p3(feature, boxes, tuple(meta['image_shapes'][i]),
                    (480, 864), meta['ratios'][i], normalize=True)
                assert bool(valid.all()), 'Every sampled Train GT ROI must be accounted for'
                arrays[cue].append(values.numpy().copy())
            arrays['video'].append(np.full(len(identity), video, np.int64))
            arrays['identity'].append(identity)
            arrays['frame'].append(np.full(len(identity), frame, np.int64))
            count += len(identity)
        sources[sequence] = dict(metadata_sha256=sha256(meta_path), GT_sha256=sha256(gt_path),
            targets_sha256=sha256(targets_path), P3_sha256=map_hashes, frames=192,
            observations=count, canonical_video_local_identities=len(np.unique(gt[(gt[:, 6]>0)&(gt[:, 7]==1), 1])))
        print(json.dumps(dict(sequence=sequence, Train_frames=192, ROI_observations=count,
                              elapsed_seconds=time.monotonic()-started)), flush=True)
        del maps
    arrays = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    paired = paired_features(arrays['rgb'], arrays['ir'])
    early = np.zeros(len(paired), bool)
    for video in np.unique(arrays['video']):
        for label in np.unique(arrays['identity'][arrays['video'] == video]):
            indices = np.flatnonzero((arrays['video']==video)&(arrays['identity']==label))
            indices = indices[np.argsort(arrays['frame'][indices], kind='stable')]
            cut = max(2, min(len(indices)-1, int(len(indices)*.7)))
            assert len(indices) >= 3
            early[indices[:cut]] = True
    diagnostic, diagnostic_fit = fit_fisher(paired[early], arrays['video'][early],
        arrays['identity'][early], spec['fit']['dimensions'], spec['fit']['ridge_fraction'])
    retrieval = {}
    for name, values in [('raw_RGB', arrays['rgb']), ('raw_IR', arrays['ir']),
                         ('raw_paired', paired), ('early_Train_fitted_Fisher', transform(paired, diagnostic))]:
        retrieval[name] = temporal_retrieval(values, arrays['video'], arrays['identity'], arrays['frame'])
    fitted, fit_details = fit_fisher(paired, arrays['video'], arrays['identity'],
        spec['fit']['dimensions'], spec['fit']['ridge_fraction'])
    model_path = base / 'Train_fisher_projection.npz'
    assert not model_path.exists()
    np.savez_compressed(model_path, **fitted)
    for path, expected in registration['source_hashes'].items():
        assert sha256(path) == expected, path
    dump(base / 'fit_complete.json', dict(status='COMPLETE_SAMPLED_TRAIN_ONLY_FISHER_PROJECTION',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        registration_sha256=sha256(base / 'fit_registered.json'), specification_sha256=sha256(SPEC),
        source_hashes=registration['source_hashes'], fit_details=fit_details,
        early_Train_diagnostic_fit=diagnostic_fit, temporal_Train_retrieval=retrieval,
        model_path=str(model_path), model_sha256=sha256(model_path), model_bytes=model_path.stat().st_size,
        Train_sources=sources, cache_first_frame_anchors=anchors, frames=1152,
        sampled_ROI_observations=len(paired), Dev_fit_rows=0, Test_fit_rows=0,
        GPU_work=False, new_NN_training=False, feature_or_image_bank_created=False,
        tracking_HOTA_result=False, supported_innovations=0, goal_met=False,
        elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(status='COMPLETE_SAMPLED_TRAIN_ONLY_FISHER_PROJECTION',
        observations=len(paired), dimensions=fit_details['dimensions'],
        Train_diagnostic_top1={k:v['equal_video_top1_percent']for k,v in retrieval.items()},
        new_tracking_HOTA=False, elapsed_seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
