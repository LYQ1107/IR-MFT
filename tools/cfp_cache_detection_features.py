#!/usr/bin/env python3
"""Encode every frozen proposal without GT crops or proposal filtering."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from cfp.identity_data import crop_to_tensor
from cfp.provenance import sha256, write_json
from cfp.reid import FishReID


def worker_init(_):
    cv2.setNumThreads(1)


class DetectionCrops(torch.utils.data.Dataset):
    def __init__(self, root, split, images, cache, paired, auxiliary_root=None):
        self.root = root / split
        self.auxiliary_root = (auxiliary_root or root) / split
        self.images = images
        self.frames = cache['frame_ids']
        self.offsets = cache['offsets']
        self.boxes = cache['boxes']
        self.paired = paired
        self.last_frame, self.last_images = None, None

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, position):
        frame_position = int(np.searchsorted(self.offsets, position, side='right') - 1)
        frame = int(self.frames[frame_position])
        if frame != self.last_frame:
            record = self.images[frame]
            rgb = cv2.imread(str(self.root / record['vi_file_name']))
            ir = cv2.imread(str(self.auxiliary_root / record['ir_file_name'])) if self.paired else None
            if rgb is None or (self.paired and (ir is None or ir.shape != rgb.shape)):
                raise ValueError('Missing or unaligned current-frame inputs')
            self.last_images = rgb, ir
            self.last_frame = frame
        box = self.boxes[position]
        xywh = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
        crops, valid = [], []
        for image in self.last_images:
            if image is None:
                crops.append(torch.zeros((3, 128, 256)))
                valid.append(False)
                continue
            try:
                crop = crop_to_tensor(image, xywh)
            except ValueError:
                # Preserve the proposal; mark unavailable appearance explicitly.
                crops.append(torch.zeros((3, 128, 256)))
                valid.append(False)
            else:
                crops.append(crop)
                valid.append(True)
        return crops[0], crops[1], valid[0], valid[1], position


def load_encoder(path):
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint['epoch'] != checkpoint['config']['epochs']:
        raise ValueError('Declared fish representation uses its fixed last epoch')
    state = checkpoint['model']
    encoder = FishReID(state['classifier.weight'].shape[0], state['projection.weight'].shape[0])
    encoder.load_state_dict(state, strict=True)
    encoder.eval().requires_grad_(False).cuda()
    return encoder, dict(checkpoint_sha256=sha256(path), checkpoint_epoch=checkpoint['epoch'],
                         seed=checkpoint['seed'], training_metadata=checkpoint['metadata'])


def declared_encoder_role(protocol_path, partition, teacher_path=None):
    protocol = json.loads(protocol_path.read_text())
    expected = sha256(protocol_path)
    if teacher_path is None:
        return protocol[partition], expected
    teacher = json.loads(teacher_path.read_text())
    if partition != 'train' or teacher['state'] != 'FROZEN' or teacher['parent_protocol_sha256'] != expected:
        raise ValueError('OOF teacher protocol must derive from canonical association train split')
    selected = teacher['teacher_heldout']
    if (set(selected) & set(teacher['train']) or
            set(selected + teacher['train']) != set(protocol['train']) or
            set(protocol['dev']) & set(teacher['train'])):
        raise ValueError('OOF teacher has heldout/dev training exposure')
    return selected, sha256(teacher_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--partition', choices=['train', 'dev', 'historical_test'], required=True)
    p.add_argument('--rgb-checkpoint', type=Path, required=True)
    p.add_argument('--ir-checkpoint', type=Path)
    p.add_argument('--auxiliary-root', type=Path, help='Declared RGB control view; detector inputs remain data-root')
    p.add_argument('--embedding-stage', choices=['pre_neck', 'post_neck'], default='pre_neck')
    p.add_argument('--teacher-protocol', type=Path, help='Explicit out-of-fold teacher protocol; encode only its heldout training videos')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--final-freeze', type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Detection feature banks are immutable')
    if args.partition == 'historical_test':
        if args.final_freeze is None or json.loads(args.final_freeze.read_text())['state'] != 'FROZEN':
            raise ValueError('Historical test appearance evaluation requires a declared final freeze')
    generation_path = args.data_root / 'manifest.json'
    generation = json.loads(generation_path.read_text())
    if generation['status'] != 'COMPLETE' or generation.get('probe_frames', 0):
        raise ValueError('Canonical proposal features require complete causal inputs')
    detection_manifest = json.loads((args.detections / 'manifest.json').read_text())
    if detection_manifest['status'] != 'COMPLETE' or detection_manifest['probe']:
        raise ValueError('Incomplete/diagnostic detection cache')
    if detection_manifest['config'].get('input_regime') != 'causal':
        raise ValueError('Use canonical causal fixed proposals for new appearance comparisons')
    if detection_manifest['config']['generation_manifest_sha256'] != sha256(generation_path):
        raise ValueError('Detection inputs changed')
    protocol = json.loads(args.protocol.read_text())
    if detection_manifest['protocol_sha256'] != sha256(args.protocol):
        raise ValueError('Detection split protocol changed')
    split = 'test' if args.partition == 'historical_test' else 'train'
    annotation = json.loads((args.data_root / 'annotations' / (split + '.json')).read_text())
    names = {video['id']: video['file_name'] for video in annotation['videos']}
    selected_sequences, expected_training_protocol = declared_encoder_role(args.protocol, args.partition, args.teacher_protocol)
    rgb_model, rgb_metadata = load_encoder(args.rgb_checkpoint)
    ir_model, ir_metadata = load_encoder(args.ir_checkpoint) if args.ir_checkpoint else (None, None)
    for metadata in (rgb_metadata, ir_metadata):
        if metadata and metadata['training_metadata']['protocol_sha256'] != expected_training_protocol:
            raise ValueError('Encoder training split differs from its declared main/OOF role')
    auxiliary_mode, auxiliary_generation_sha256 = 'constructed_ir', None
    if args.auxiliary_root:
        if ir_model is None:
            raise ValueError('Auxiliary view requires its trained encoder')
        auxiliary_path = args.auxiliary_root / 'manifest.json'
        auxiliary = json.loads(auxiliary_path.read_text())
        auxiliary_mode = auxiliary['config']['mode']
        if auxiliary_mode not in ('rgb_contrast', 'rgb_background_difference'):
            raise ValueError('Unknown auxiliary view')
        if (auxiliary['status'] != 'COMPLETE' or auxiliary.get('probe_frames', 0) or
                auxiliary['canonical_generation_manifest_sha256'] != sha256(generation_path)):
            raise ValueError('Control view does not match the complete canonical input history')
        if sha256(args.auxiliary_root / 'annotations' / (split + '.json')) != sha256(args.data_root / 'annotations' / (split + '.json')):
            raise ValueError('Control frame indexing changed')
        auxiliary_generation_sha256 = sha256(auxiliary_path)
    expected_regime = ('complete causal constructed IR' if auxiliary_mode == 'constructed_ir' else
                       f'complete causal RGB auxiliary control: {auxiliary_mode}')
    if ir_metadata and ir_metadata['training_metadata']['input_regime'] != expected_regime:
        raise ValueError('Auxiliary encoder training inputs disagree with selected view')
    if ir_metadata and ir_metadata['seed'] != rgb_metadata['seed']:
        raise ValueError('Main paired baseline uses matched encoder seed indices')
    args.output.mkdir(parents=True)
    manifest = dict(status='RUNNING', input_regime='causal', partition=args.partition,
                    protocol_sha256=sha256(args.protocol), generation_manifest_sha256=sha256(generation_path),
                    detection_manifest_sha256=sha256(args.detections / 'manifest.json'),
                    rgb_encoder=rgb_metadata, ir_encoder=ir_metadata, fp16=True,
                    auxiliary_view=auxiliary_mode if ir_model else None,
                    auxiliary_generation_manifest_sha256=auxiliary_generation_sha256,
                    storage_schema='ir column denotes the auxiliary view; inspect auxiliary_view, never infer a thermal sensor',
                    embedding_stage=args.embedding_stage,
                    encoder_role='oof_teacher' if args.teacher_protocol else 'main_baseline',
                    teacher_protocol_sha256=sha256(args.teacher_protocol) if args.teacher_protocol else None,
                    selected_sequences=selected_sequences,
                    crop_hw=[128, 256], margin=.05, normalize='ImageNet', batch_size=args.batch_size,
                    source_sha256=sha256(__file__), crop_source_sha256=sha256('cfp/identity_data.py'),
                    model_source_sha256=sha256('cfp/reid.py'),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    scope='Current-frame detector crops, aligned to unchanged proposals; no GT crop or online oracle labels')
    write_json(args.output / 'manifest.json', manifest)
    reports = {}
    start_all = time.perf_counter()
    for sequence in selected_sequences:
        cache_path = args.detections / (sequence + '.npz')
        expected = detection_manifest['caches'][sequence]['sha256']
        if sha256(cache_path) != expected:
            raise ValueError('Detection bank changed')
        cache = np.load(cache_path)
        if np.any(cache['boxes'][:, 2:] <= cache['boxes'][:, :2]) or not np.isfinite(cache['boxes']).all():
            raise ValueError('Invalid frozen proposal geometry')
        images = {int(image['frame_id']): image for image in annotation['images'] if names[image['video_id']] == sequence}
        if not np.array_equal(cache['frame_ids'], np.arange(1, len(images) + 1)):
            raise ValueError('Proposal bank does not cover full selected video')
        dataset = DetectionCrops(args.data_root, split, images, cache, ir_model is not None, args.auxiliary_root)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init)
        rgb_rows, ir_rows, valid_rgb_rows, valid_ir_rows, latencies = [], [], [], [], []
        next_position = 0
        with torch.inference_mode():
            for rgb, ir, valid_rgb, valid_ir, positions in loader:
                if positions.tolist() != list(range(next_position, next_position + len(positions))):
                    raise ValueError('Feature-to-proposal index correspondence lost')
                next_position += len(positions)
                rgb = rgb.cuda(non_blocking=True)
                ir = ir.cuda(non_blocking=True) if ir_model else None
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.cuda.amp.autocast():
                    rgb_feature, _ = rgb_model(rgb, embedding_stage=args.embedding_stage)
                    ir_feature, _ = ir_model(ir, embedding_stage=args.embedding_stage) if ir_model else (None, None)
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t)
                if not torch.isfinite(rgb_feature).all() or (ir_feature is not None and not torch.isfinite(ir_feature).all()):
                    raise ValueError('Nonfinite proposal features')
                rgb_feature = rgb_feature.cpu().numpy()
                rgb_feature[~valid_rgb.numpy()] = 0
                rgb_rows.append(rgb_feature)
                valid_rgb_rows.append(valid_rgb.numpy())
                if ir_feature is not None:
                    ir_feature = ir_feature.cpu().numpy()
                    ir_feature[~valid_ir.numpy()] = 0
                    ir_rows.append(ir_feature)
                    valid_ir_rows.append(valid_ir.numpy())
        if next_position != len(dataset) or len(dataset) == 0:
            raise ValueError('Missing proposal features')
        values = dict(rgb=np.concatenate(rgb_rows), valid_rgb=np.concatenate(valid_rgb_rows))
        if ir_rows:
            values.update(ir=np.concatenate(ir_rows), valid_ir=np.concatenate(valid_ir_rows))
        path = args.output / (sequence + '.npz')
        np.savez_compressed(path, **values)
        report = dict(manifest, status='COMPLETE', sequence=sequence, detection_sha256=expected,
                      features_sha256=sha256(path), detections=len(dataset),
                      invalid_rgb_crops=int(np.sum(~values['valid_rgb'])),
                      invalid_ir_crops=int(np.sum(~values['valid_ir'])) if ir_rows else None,
                      batch_encoder_latency_p50_p95_ms=(np.quantile(latencies, [.5, .95]) * 1000).tolist(),
                      latency_scope='Crop encoder(s), excludes decode, cropping, generator, detector and tracker')
        write_json(path.with_suffix('.provenance.json'), report)
        reports[sequence] = {key: report[key] for key in ['features_sha256', 'detections', 'invalid_rgb_crops', 'invalid_ir_crops', 'batch_encoder_latency_p50_p95_ms']}
        print(sequence, reports[sequence], flush=True)
    manifest.update(status='COMPLETE', videos=reports, total_seconds=time.perf_counter() - start_all)
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
