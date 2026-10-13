"""Inspect current-crop raw/pre-neck/neck information on fit and inner only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from cfp.crop_retrieval import embedding_crop_retrieval
from cfp.local_identity_encoder import resnet_spatial
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.trainable_local_identity import TrainableLocalFishReID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, required=True)
    ap.add_argument('--crops', type=Path, required=True)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    videos = config['fit'] + [config['inner']]
    if (config['state'] != 'FROZEN' or config['excluded_outer'] in videos
            or len(set(videos)) != len(videos) or args.output.exists()
            or not os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('Frozen fit/inner only, explicit idle GPU, new output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.4)
    fm = json.loads((args.features / 'manifest.json').read_text())
    cm = json.loads((args.crops / 'manifest.json').read_text())
    if cm['feature_manifest_sha256'] != sha256(args.features / 'manifest.json'):
        raise ValueError('Crop feature provenance differs')
    models = {}
    model_proofs = {}
    for mode in config['models']:
        path = args.runs / f'visual_identity_{mode}_SN-015_s0_v1'
        report = json.loads((path / 'manifest.json').read_text())
        if report['status'] != 'COMPLETE' or report['config']['fit'] != config['fit'] or report['config']['inner'] != config['inner']:
            raise ValueError('Only the registered completed fit checkpoint is supported')
        if sha256(path / 'epoch8.pt') != report['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        # Exact executable architecture matches, while evaluator improvements are
        # versioned independently rather than pretending historical source equality.
        for source in ('cfp/trainable_local_identity.py', 'cfp/local_identity_encoder.py'):
            if sha256(source) != report['guard']['source_sha256'][source]:
                raise ValueError('Model architecture source changed')
        saved = torch.load(path / 'epoch8.pt', map_location='cpu', weights_only=False)
        model = TrainableLocalFishReID(len(saved['classifier_keys']), report['config']['dimension'], mode, report['config']['local_count'])
        model.load_state_dict(saved['model'], strict=True)
        models[mode] = model.cuda().eval()
        model_proofs[mode] = {'path': str(path / 'epoch8.pt'), 'sha256': report['checkpoint_sha256'], 'manifest_sha256': sha256(path / 'manifest.json')}
    # Random orthogonal projection has no labels, no fitting or model selection.
    rng = np.random.default_rng(config['orthogonal_projection_seed'])
    projection = torch.from_numpy(np.linalg.qr(rng.standard_normal((2048, 256)))[0].astype(np.float32)).cuda()
    result = dict(status='RUNNING', config=config, model_proofs=model_proofs,
        config_sha256=sha256(args.config), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        feature_manifest_sha256=sha256(args.features / 'manifest.json'), crop_manifest_sha256=sha256(args.crops / 'manifest.json'),
        physical_devices=os.environ['CUDA_VISIBLE_DEVICES'], videos={}, scope=config['scope'])
    sources = [__file__, 'cfp/crop_retrieval.py', 'cfp/trainable_local_identity.py', 'cfp/local_identity_encoder.py', args.config]
    result['source_snapshots'] = snapshot_sources(args.output.with_suffix('.sources'), sources)
    write_json(args.output, result)
    for video in videos:
        start_time = time.monotonic()
        folder = args.features / video
        records_path = folder / 'records.json'
        if sha256(records_path) != fm['videos'][video]['records_sha256']:
            raise ValueError('Records changed')
        records = json.loads(records_path.read_text())
        arrays = {}
        for key in ('valid', 'visibility8', 'imagenet_global'):
            p = folder / (key + '.npy')
            if sha256(p) != fm['videos'][video]['artifacts'][key]['sha256']:
                raise ValueError('Source array changed')
            arrays[key] = np.load(p, mmap_mode='r')
        p = args.crops / (video + '.npy')
        if sha256(p) != cm['videos'][video]['sha256']:
            raise ValueError('Crop input changed')
        crops = np.load(p, mmap_mode='r')
        if len(crops) != len(records):
            raise ValueError('Crop order changed')
        frozen = torch.from_numpy(np.array(arrays['imagenet_global'])).cuda().float()
        controls = {'imagenet_raw2048': F.normalize(frozen, dim=-1), 'imagenet_orthogonal_projection256': F.normalize(frozen @ projection, dim=-1)}
        def evaluate(values):
            return embedding_crop_retrieval(values, records, np.array(arrays['valid']), 'cuda', config['query_cap_per_identity_per_view'], config['minimum_past_gaps'])
        measurements = {k: evaluate(z.cpu().numpy()) for k, z in controls.items()}
        del controls, frozen
        for mode, model in models.items():
            outputs = {key: [] for key in config['stages']}
            for start in range(0, len(crops), config['batch_size']):
                end = start + config['batch_size']
                images = torch.from_numpy(np.array(crops[start:end])).cuda().float()
                visible = torch.from_numpy(np.array(arrays['visibility8'][start:end])).cuda().float()
                cache = {}
                hook = model.head.neck.register_forward_pre_hook(lambda module, inputs: cache.update(pre=inputs[0].detach()))
                try:
                    with torch.inference_mode(), torch.cuda.amp.autocast():
                        spatial = resnet_spatial(model.backbone, images)
                        out = model.forward_spatial(spatial, visible)
                        raw = F.adaptive_avg_pool2d(spatial, 1).flatten(1)
                finally:
                    hook.remove()
                outputs['backbone_raw2048'].append(F.normalize(raw.float(), dim=-1).cpu().numpy())
                outputs['head_pre_layernorm256'].append(F.normalize(cache['pre'].float(), dim=-1).cpu().numpy())
                outputs['head_layernorm256'].append(out['embedding'].cpu().numpy())
            measurements[mode] = {key: evaluate(np.concatenate(parts)) for key, parts in outputs.items()}
        result['videos'][video] = {'measurements': measurements, 'seconds': time.monotonic() - start_time}
        write_json(args.output, result)
        print(json.dumps({'video': video, 'seconds': result['videos'][video]['seconds']}), flush=True)
    result.update(status='COMPLETE_FIT_INNER_VISUAL_STAGE_DIAGNOSIS', peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(args.output, result)


if __name__ == '__main__':
    main()
