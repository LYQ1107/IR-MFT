#!/usr/bin/env python3
"""One bounded aligned crop cache for real outer/inner-excluded visual fitting."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from cfp.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing crop cache')
    fm = json.loads((args.features / 'manifest.json').read_text())
    index = json.loads((args.index / 'manifest.json').read_text())
    if fm['status'] != 'COMPLETE_SPARSE_TWO_CROP_FEATURES' or fm['index_manifest_sha256'] != sha256(args.index / 'manifest.json'):
        raise ValueError('Exact completed sparse current feature/index source required')
    planned = sum(e['observations'] for e in fm['videos'].values()) * 3 * 128 * 256 * 2
    if planned > 16 * 2**30 or shutil.disk_usage(args.output.parent).free - planned < 40 * 2**30:
        raise OSError('Bounded cache budget or40GiB reserve')
    spec = importlib.util.spec_from_file_location('crop_source_helper', Path(__file__).with_name('cfp_extract_two_crop_identity_features.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    args.output.mkdir(parents=True)
    report = dict(status='RUNNING', feature_manifest_sha256=sha256(args.features / 'manifest.json'),
        index_manifest_sha256=sha256(args.index / 'manifest.json'), source_sha256=sha256(__file__),
        crop_source_sha256=sha256('cfp/identity_data.py'), videos={},
        scope='Same registered sparse GT/detector current crops/order storedfloat16 for newImageNet-initialized visual fitting. No new image or detector proposals. Outer/inner exclusions enforced by trainers; noIR read.')
    write_json(args.output / 'manifest.json', report)
    torch.set_num_threads(2)
    for video, e in fm['videos'].items():
        records_path = args.features / video / 'records.json'
        if sha256(records_path) != e['records_sha256']:
            raise ValueError('Crop records drift')
        records = json.loads(records_path.read_text())
        ip, paths = args.index / (video + '.npz'), args.index / (video + '.frames.json')
        if sha256(ip) != index['videos'][video]['index_sha256'] or sha256(paths) != index['videos'][video]['frame_paths_sha256']:
            raise ValueError('Crop geometry/frame path drift')
        with np.load(ip, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        selected = [(row['frame'], 0 if row['view'] == 'gt' else 1, row['source_index'], row['identity']) for row in records]
        dataset = helper.CurrentCropDataset(index['data_root'], json.loads(paths.read_text()), arrays, selected)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, worker_init_fn=helper.worker_init)
        output = args.output / (video + '.npy')
        mapped = np.lib.format.open_memmap(output, mode='w+', dtype=np.float16, shape=(len(records), 3, 128, 256))
        original_valid = np.load(args.features / video / 'valid.npy')
        written = 0
        for crops, visibility, valid, positions in loader:
            if positions.tolist() != list(range(written, written + len(positions))) or not np.array_equal(valid.numpy(), original_valid[positions]):
                raise ValueError('Current crop alignment/validity changed')
            mapped[positions.numpy()] = crops.numpy().astype(np.float16)
            written += len(positions)
            if shutil.disk_usage(args.output).free < 40 * 2**30:
                raise OSError('40GiB reserve')
        mapped.flush()
        if written != len(records):
            raise ValueError('Incomplete crop cache')
        del mapped
        report['videos'][video] = dict(rows=written, records_sha256=e['records_sha256'], sha256=sha256(output), bytes=output.stat().st_size)
        write_json(args.output / 'manifest.json', report)
        print(video, 'COMPLETE', written, flush=True)
    report['status'] = 'COMPLETE_ALIGNED_PILOT_CROPS'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
