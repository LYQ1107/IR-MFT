#!/usr/bin/env python3
"""Matched current-crop global/local features on the declared sparse pilot."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models import resnet50

from cfp.identity_data import crop_to_tensor
from cfp.local_identity_encoder import frozen_fish_stages, letterbox_visibility, resnet_spatial, strip_features
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.reid import FishReID


class CurrentCropDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, frame_paths, arrays, selected):
        self.root = Path(data_root) / 'train'
        self.paths = {r['frame_id']: r['rgb_file'] for r in frame_paths}
        self.arrays, self.selected = arrays, selected
        self.previous, self.image = None, None

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, position):
        frame, view, index, _ = self.selected[position]
        if frame != self.previous:
            self.image = cv2.imread(str(self.root / self.paths[frame]))
            if self.image is None:
                raise ValueError('Missing current RGB frame')
            self.previous = frame
        box = self.arrays['gt_boxes' if view == 0 else 'detection_boxes'][index]
        x1, y1, x2, y2 = box
        xywh = [x1, y1, x2 - x1, y2 - y1]
        try:
            crop = crop_to_tensor(self.image, xywh)
            visibility = letterbox_visibility(self.image.shape, xywh)
            valid = True
        except ValueError:
            crop, visibility, valid = torch.zeros(3, 128, 256), torch.zeros(1, 128, 256), False
        return crop, visibility, valid, position


def worker_init(_):
    cv2.setNumThreads(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('New immutable features and explicit idle GPU required')
    config = json.loads(args.config.read_text())
    source = json.loads((args.index / 'manifest.json').read_text())
    if config['state'] != 'FROZEN' or source['status'] != 'COMPLETE_INDEX_ONLY':
        raise ValueError('Frozen complete source index required')
    if shutil.disk_usage(args.output.parent).free < config['reserve_gib'] * 2**30:
        raise OSError('40GiB reserve')
    assets = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    fish_checkpoint = assets / 'phase1/reid_rgb_seed0/epoch12.pt'
    fm = json.loads(fish_checkpoint.with_name('manifest.json').read_text())
    if sha256(fish_checkpoint) != fm['checkpoint_sha256']:
        raise ValueError('Inherited exposed fish checkpoint changed')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    fish_state = torch.load(fish_checkpoint, map_location='cpu', weights_only=False)['model']
    fish = FishReID(fish_state['classifier.weight'].shape[0]).eval()
    fish.load_state_dict(fish_state, strict=True)
    fish.requires_grad_(False).cuda()
    pretrained = Path('/data3/liuyeqiang/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')
    imagenet = resnet50(weights=None)
    imagenet.load_state_dict(torch.load(pretrained, map_location='cpu', weights_only=True), strict=True)
    imagenet.fc = torch.nn.Identity()
    imagenet.eval().requires_grad_(False).cuda()
    args.output.mkdir(parents=True)
    report = dict(status='RUNNING', config=config, config_sha256=sha256(args.config),
        index_manifest_sha256=sha256(args.index / 'manifest.json'), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        fish_checkpoint_sha256=sha256(fish_checkpoint), imagenet_checkpoint_sha256=sha256(pretrained),
        physical_devices=os.environ['CUDA_VISIBLE_DEVICES'], torch=torch.__version__, videos={},
        scope='Offline current GT/detector crop pilot on registered frame pairs, not all-frame online bank or speed. Fish encoder all6exposed; ImageNet backbone no fish fitting, external pretraining overlap not audited away. Local strips are untrained controls, not anatomy or DenseJEPA.',
        source_snapshots=snapshot_sources(args.output / 'sources', [__file__, 'cfp/local_identity_encoder.py', 'cfp/identity_data.py', 'cfp/reid.py', args.config]))
    write_json(args.output / 'manifest.json', report)
    bytes_written = 0
    for video, entry in source['videos'].items():
        ip, paths = args.index / (video + '.npz'), args.index / (video + '.frames.json')
        if sha256(ip) != entry['index_sha256'] or sha256(paths) != entry['frame_paths_sha256']:
            raise ValueError('Two-crop index drift')
        with np.load(ip, allow_pickle=False) as a:
            arrays = {k: a[k] for k in a.files}
        selected = []
        for view, prefix in ((0, 'gt'), (1, 'detection')):
            frame_values, identities = arrays[prefix + '_frame'], arrays[prefix + '_identity']
            for i in np.flatnonzero(np.isin(frame_values % config['frame_period'], config['frame_residues'])):
                selected.append((int(frame_values[i]), view, int(i), int(identities[i])))
        selected.sort(key=lambda r: (r[0], r[1], r[2]))
        dataset = CurrentCropDataset(source['data_root'], json.loads(paths.read_text()), arrays, selected)
        loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['workers'],
            shuffle=False, pin_memory=True, worker_init_fn=worker_init)
        n = len(selected)
        specifications = {'fish_raw': (n, 2048), 'fish_pre': (n, 512), 'fish_post': (n, 512),
            'fish_local4': (n, 4, 512), 'fish_local8': (n, 8, 512), 'imagenet_global': (n, 2048),
            'imagenet_local4': (n, 4, 2048), 'imagenet_local8': (n, 8, 2048),
            'visibility4': (n, 4), 'visibility8': (n, 8), 'valid': (n,)}
        directory = args.output / video
        directory.mkdir()
        outputs = {}
        for name, shape in specifications.items():
            dtype = np.bool_ if name == 'valid' else np.float32 if name in ('fish_raw', 'fish_pre', 'fish_post') else np.float16
            outputs[name] = np.lib.format.open_memmap(directory / (name + '.npy'), mode='w+', dtype=dtype, shape=shape)
            bytes_written += outputs[name].nbytes
        if bytes_written > config['max_feature_bytes']:
            raise OSError('Declared feature storage budget exceeded')
        fp = assets / 'phase2/causal_train_detection_features_postneck_seed0' / (video + '.npz')
        provenance = json.loads(fp.with_suffix('.provenance.json').read_text())
        if sha256(fp) != provenance['features_sha256']:
            raise ValueError('Old detector-feature parity bank changed')
        with np.load(fp, allow_pickle=False) as a:
            original, original_valid = a['rgb'].copy(), a['valid_rgb'].copy()
        parity = dict(rows=0, maximum_component_error=0., maximum_cosine_error=0., validity_equal=True)
        start, next_position = time.monotonic(), 0
        with torch.inference_mode():
            for crops, visibility, valid, positions in loader:
                if not torch.equal(positions, torch.arange(next_position, next_position + len(positions))):
                    raise ValueError('Crop order lost')
                next_position += len(positions)
                crops = crops.cuda(non_blocking=True)
                with torch.cuda.amp.autocast():
                    spatial = resnet_spatial(fish.backbone, crops)
                    stages = frozen_fish_stages(fish, spatial)
                    im_spatial = resnet_spatial(imagenet, crops)
                    im_global = F.adaptive_avg_pool2d(im_spatial, 1).flatten(1)
                    im_local = {k: strip_features(im_spatial, k) for k in (4, 8)}
                values = {'fish_raw': stages['raw'], 'fish_pre': stages['pre_neck'], 'fish_post': stages['post_neck'],
                    'fish_local4': stages['local4'], 'fish_local8': stages['local8'], 'imagenet_global': im_global,
                    'imagenet_local4': im_local[4], 'imagenet_local8': im_local[8],
                    'visibility4': F.adaptive_avg_pool2d(visibility, (1, 4)).flatten(1),
                    'visibility8': F.adaptive_avg_pool2d(visibility, (1, 8)).flatten(1), 'valid': valid}
                pos = positions.numpy()
                for name, value in values.items():
                    value = value.float().cpu().numpy() if name != 'valid' else value.numpy()
                    if name != 'valid':
                        value[~valid.numpy()] = 0
                    if not np.isfinite(value).all():
                        raise ValueError('Nonfinite feature extraction')
                    outputs[name][pos] = value
                det_positions = [j for j, p in enumerate(pos) if selected[p][1] == 1]
                if det_positions:
                    oi = np.asarray([selected[pos[j]][2] for j in det_positions])
                    reencoded = np.asarray(outputs['fish_post'][pos[det_positions]])
                    parity['rows'] += len(oi)
                    parity['maximum_component_error'] = max(parity['maximum_component_error'], float(np.max(np.abs(reencoded - original[oi]))))
                    known = original_valid[oi]
                    if known.any():
                        parity['maximum_cosine_error'] = max(parity['maximum_cosine_error'], float(np.max(np.abs(1 - (reencoded[known] * original[oi][known]).sum(1)))))
                    parity['validity_equal'] &= bool(np.array_equal(valid.numpy()[det_positions], original_valid[oi]))
                if next_position % 2048 < config['batch_size']:
                    if shutil.disk_usage(args.output).free < config['reserve_gib'] * 2**30:
                        raise OSError('40GiB reserve')
                    print(video, next_position, '/', n, flush=True)
        if next_position != n or not parity['validity_equal'] or parity['maximum_component_error'] > .01 or parity['maximum_cosine_error'] > .001:
            raise ValueError('Incomplete output or inherited-coordinate numerical drift')
        artifacts = {}
        for name, array in outputs.items():
            array.flush()
            artifacts[name] = dict(shape=list(array.shape), dtype=str(array.dtype), sha256=sha256(directory / (name + '.npy')))
        records = [dict(frame=frame, view='gt' if view == 0 else 'detector', source_index=index, identity=identity) for frame, view, index, identity in selected]
        write_json(directory / 'records.json', records)
        report['videos'][video] = dict(observations=n, gt=sum(r[1] == 0 for r in selected), detector=sum(r[1] == 1 for r in selected),
            artifacts=artifacts, records_sha256=sha256(directory / 'records.json'), numeric_parity=parity,
            extraction_seconds=time.monotonic() - start)
        write_json(args.output / 'manifest.json', report)
        print(video, 'COMPLETE', report['videos'][video]['numeric_parity'], flush=True)
    report.update(status='COMPLETE_SPARSE_TWO_CROP_FEATURES', bytes=bytes_written,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
