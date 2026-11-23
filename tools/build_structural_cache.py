#!/usr/bin/env python3
"""Small CPU-only Train descriptors, reusing exact existing frame IDs."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from tools.structural_grid import structural_grid
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    spec_path = ROOT/'configs/perceptual_structural_relative_20261123.json'
    plan = json.loads(plan_path.read_text()); spec = json.loads(spec_path.read_text())
    assert sha256(plan_path) == spec['main_plan_sha256']
    assert sha256(ROOT/'configs/perceptual_correlation_repair_20261123.json') == spec['parent_plan_sha256']
    diagnostic = json.loads((ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURE_TRAIN_AUDIT.json').read_text())
    assert diagnostic['prerequisite_for_further_registration']
    protocol = json.loads((ROOT/plan['protocol']).read_text()); work = Path(plan['work_root'])
    out = work/spec['cache']['directory']; out.mkdir(exist_ok=False)
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),
        ROOT/'tools/structural_grid.py', ROOT/'tools/structural_correspondence.py']}
    state = dict(status='PRODUCING_TRAIN_SCALAR_DESCRIPTORS_NO_GT', specification_sha256=sha256(spec_path),
                 source_hashes=sources, physical_gpu=None, GT_read=False, Test_read=False, sequences={})
    dump(out/'state.json', state); cv2.setNumThreads(1); started = time.perf_counter()
    physical_bytes = 0
    for base, dirs, files in os.walk(work, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(base)/name).is_symlink()]
        physical_bytes += sum((Path(base)/name).lstat().st_size for name in files if not (Path(base)/name).is_symlink())
    if physical_bytes+int(.3*2**30) > 64*2**30 or shutil.disk_usage(work).free < 40*2**30:
        raise RuntimeError('Respect work cap and independent-disk reserve')
    for sequence in protocol['train']:
        cache_path = work/'correlation_cache'/sequence/'features/complete.json'
        cache = json.loads(cache_path.read_text()); assert cache['config_sha256'] == spec['parent_plan_sha256']
        frames = cache['frame_ids']; assert len(frames) == 192 and cache['Dev_rows'] == cache['Test_rows'] == 0
        folder = out/sequence; folder.mkdir(); arrays = {}
        for cue in ['rgb', 'ir']:
            arrays[cue+'_descriptor'] = np.lib.format.open_memmap(folder/(cue+'_descriptor.npy'), mode='w+', dtype=np.float16, shape=(192, 8, 60, 108))
            arrays[cue+'_valid'] = np.lib.format.open_memmap(folder/(cue+'_valid.npy'), mode='w+', dtype=np.bool_, shape=(192, 1, 60, 108))
        for index, frame in enumerate(frames):
            rgb_path = Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'
            ir_path = Path(plan['generated_ir_root'])/'train'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
            rgb = cv2.imread(str(rgb_path)); ir = cv2.imread(str(ir_path), -1)
            assert rgb is not None and ir is not None and ir.shape == rgb.shape[:2] and ir.dtype == np.uint8
            for cue, image in [('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)), ('ir', ir)]:
                values, valid = structural_grid(image, spec['training_input_hw'])
                arrays[cue+'_descriptor'][index] = values; arrays[cue+'_valid'][index] = valid
            if (index+1) % 32 == 0:
                print(json.dumps(dict(sequence=sequence, frames=index+1, total=192, seconds=time.perf_counter()-started)), flush=True)
        outputs = {}
        for name, array in arrays.items():
            array.flush(); path = folder/(name+'.npy')
            outputs[name] = dict(path=str(path), shape=list(array.shape), bytes=path.stat().st_size, sha256=sha256(path))
        meta = dict(status='COMPLETE_TRAIN_SCALAR_STRUCTURE', frames=192, frame_ids=frames,
            outputs=outputs, GT_read=False, Dev_rows=0, Test_rows=0, source_hashes=sources,
            specification_sha256=sha256(spec_path), parent_cache_manifest_sha256=sha256(cache_path))
        dump(folder/'complete.json', meta)
        state['sequences'][sequence] = dict(manifest=str(folder/'complete.json'), sha256=sha256(folder/'complete.json'))
        dump(out/'state.json', state)
    assert all(sha256(ROOT/path) == value for path, value in sources.items())
    assert sha256(spec_path) == state['specification_sha256']
    state.update(status='COMPLETE', frames=1152, seconds=time.perf_counter()-started)
    dump(out/'state.json', state)
    print(json.dumps(dict(status='COMPLETE_TRAIN_SCALAR_STRUCTURE', frames=1152, physical_gpu=None)))


if __name__ == '__main__':
    main()
