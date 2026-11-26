#!/usr/bin/env python3
"""Bounded GPU4 engineering probe of the unchanged November image generator.

Eight Train calls test prefix invariance, sequence resets and sampled cache
pixel parity. Original PNGs and weights remain read-only. The active MFT
training process is never stopped or signalled.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/perceptual_fixed_November_causality_probe_20261126.json'
WORK = Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def original_live(pin):
    folder = Path('/proc')/str(pin['PID'])
    stat = (folder/'stat').read_text()
    fields = stat[stat.rfind(')')+2:].split()
    command = (folder/'cmdline').read_bytes()
    require(fields[0] not in ('Z', 'T') and int(fields[19]) == pin['start_ticks'] and
            hashlib.sha256(command).hexdigest() == pin['raw_argv_sha256'], 'Original process identity changed')


def guard(plan):
    original_live(plan['original_controller'])
    original_live(plan['original_IR_worker'])
    state = read(WORK/'native_pyramid_study_state.json')
    require(state['status'] == plan['required_original_phase'] and
            state['phases']['NORMAL_TRAIN_09']['temporal_ir']['status'] == 'RUNNING',
            'Abort only the auxiliary probe: original native phase advanced')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    plan = read(PLAN)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '4' and
            os.environ.get('PROBE_NVML_PHYSICAL_GPU') == '4', 'This independent probe uses physical GPU4 only')
    guard(plan)
    info = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True)
    device = next(line.split(',') for line in info.stdout.splitlines() if line.split(',')[0].strip() == '4')
    require(device[1].strip() == plan['physical_gpu_uuid'] and
            int(device[2]) >= plan['minimum_free_gpu_MiB_before_start'], 'GPU4 memory reserve unavailable')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    require(int(memory['MemAvailable'].split()[0])*1024 >= plan['minimum_available_host_GiB']*1024**3,
            'Host memory reserve unavailable')
    output = args.output_directory
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    archive = ROOT/plan['generator_archive']
    configuration = read(ROOT/plan['generator_configuration'])
    mft = read(ROOT/'configs/perceptual_mft_20261122.json')
    require(configuration['output_root'] == mft['generated_ir_root'] and
            configuration['run_BLIP'] is False and configuration['seed_every_frame'] == 20261101,
            'Wrong fixed November image source')
    files = {str(p): sha(p) for p in [PLAN, Path(__file__).resolve(), ROOT/plan['generator_configuration'],
        ROOT/'configs/perceptual_mft_20261122.json', archive/'scripts/diff_worker.py',
        archive/'scripts/final_aux_worker.py', archive/'scripts/production_common.py',
        archive/'DiffV2IR/configs/generate.yaml']}
    weights = {name: archive/'cache/ram'/relative for name, relative in [
        ('DiffV2IR', 'DiffV2IR/IR-500k/finetuned_checkpoints/M3FD.ckpt'),
        ('SAM', 'SAM/sam_vit_b_01ec64.pth')]}
    weight_receipts = {}
    for name, path in weights.items():
        guard(plan)
        stat = path.stat()
        print(json.dumps(dict(status='READ_ONLY_WEIGHT_SHA256_BEGIN', weight=name, bytes=stat.st_size)), flush=True)
        weight_receipts[name] = dict(path=str(path.resolve()), bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns, sha256=sha(path))
        print(json.dumps(dict(status='READ_ONLY_WEIGHT_SHA256_COMPLETE', weight=name,
                              sha256=weight_receipts[name]['sha256'])), flush=True)
    require(weight_receipts['DiffV2IR']['sha256'] == configuration['weight_sha256'], 'Actual generator weight digest differs')
    write(output/'before_GPU_provenance.json', dict(source_hashes=files, weight_receipts=weight_receipts,
        original_GPU4_free_MiB=int(device[2]), original_process_pins={k: plan[k] for k in ('original_controller','original_IR_worker')}))
    guard(plan)
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    require(not torch.cuda.is_initialized(), 'CUDA must not initialize before provenance and original-phase guards')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(plan['maximum_cuda_memory_fraction'])
    context = output/'runtime_context'
    (context/'cache').mkdir(parents=True)
    (context/'evidence').mkdir()
    (context/'cache/ram').symlink_to(archive/'cache/ram', target_is_directory=True)
    (context/'DiffV2IR').symlink_to(archive/'DiffV2IR', target_is_directory=True)
    fixture_roots = {name: context/('input_'+name) for name in ('original','changed_future','reset_interlude')}
    fixture_rows = []
    for chain in ('original','changed_future'):
        folder = fixture_roots[chain]/'BT-003/visible'
        folder.mkdir(parents=True)
        for frame in (1,2,3):
            source_sequence = 'BT-001' if chain == 'changed_future' and frame == 3 else 'BT-003'
            source_path = Path(mft['data_root'])/'train'/source_sequence/'visible'/f'{frame:06d}.jpg'
            link = folder/f'{frame:06d}.jpg'
            link.symlink_to(source_path)
            fixture_rows.append(dict(chain=chain, declared_frame=frame, path=str(link),
                original_source=str(source_path), source_sha256=sha(source_path)))
    interlude = fixture_roots['reset_interlude']/'BT-001/visible'
    interlude.mkdir(parents=True)
    (interlude/'000001.jpg').symlink_to(Path(mft['data_root'])/'train/BT-001/visible/000001.jpg')
    require(all(sha(fixture_roots['original']/'BT-003/visible'/f'{f:06d}.jpg') ==
                sha(fixture_roots['changed_future']/'BT-003/visible'/f'{f:06d}.jpg') for f in (1,2)),
            'Fixture prefixes differ before generation')
    future_a = (fixture_roots['original']/'BT-003/visible/000003.jpg').read_bytes()
    future_b = (fixture_roots['changed_future']/'BT-003/visible/000003.jpg').read_bytes()
    require(future_a != future_b, 'Fixture future was not changed')
    lookahead_control = []
    for frame in (1,2):
        current = (fixture_roots['original']/'BT-003/visible'/f'{frame:06d}.jpg').read_bytes()
        a, b = hashlib.sha256(current+future_a).hexdigest(), hashlib.sha256(current+future_b).hexdigest()
        require(a != b, 'Fixture cannot detect deliberate lookahead')
        lookahead_control.append(dict(frame=frame, original=a, changed_future=b, differs=True))
    write(output/'input_fixture_before_prefix.json', dict(sequence_links=fixture_rows,
        both_full_suffixes_prepared_before_prefix=True, prefixes_identical=True, future_changed=True,
        deliberate_CPU_lookahead_control=lookahead_control, original_RGB_bytes_copied=False))
    sys.path[:0] = [str(archive/'scripts'), str(archive/'DiffV2IR'),
                    str(archive/'DiffV2IR/stable_diffusion')]
    import diff_worker
    diff_worker.R = context
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    loaded_start = time.monotonic()
    guard(plan)
    sam = sam_model_registry['vit_b'](checkpoint=str(weights['SAM'])).cuda().eval()
    generator = SamAutomaticMaskGenerator(sam, points_per_side=32, points_per_batch=64,
        pred_iou_thresh=.88, stability_score_thresh=.95, crop_n_layers=0, min_mask_region_area=0)
    guard(plan)
    loaded = diff_worker.load_model('M3FD')
    package_roots = [Path(diff_worker.__file__).resolve().parent,
                     (archive/'DiffV2IR').resolve()]
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if not path or not str(path).endswith('.py'):
            continue
        path = Path(path).resolve()
        if any(path.is_relative_to(base) for base in package_roots) or any(
                part in path.parts for part in ('segment_anything', 'k_diffusion')):
            files[str(path)] = sha(path)
    # Original-phase and wall checks also run inside every diffusion step.
    def step_guard(*_):
        guard(plan)
        require(time.monotonic()-started < plan['maximum_child_wall_seconds'], 'Auxiliary probe wall limit')
    handle = loaded[1].register_forward_pre_hook(step_guard)
    torch.cuda.synchronize()
    model_loading_seconds = time.monotonic()-loaded_start
    print(json.dumps(dict(status='ORIGINAL_GENERATOR_AND_SAM_LOADED_GPU4', model_loading_seconds=model_loading_seconds)), flush=True)
    rows, arrays = [], {}
    calls = [('original', 'BT-003', f, 'BT-003', f) for f in (1,2,3)]
    calls += [('changed_future', 'BT-003', f, 'BT-003' if f<3 else 'BT-001', f) for f in (1,2,3)]
    calls += [('reset_interlude', 'BT-001', 1, 'BT-001', 1), ('reset_return', 'BT-003', 1, 'BT-003', 1)]
    for index, (chain, sequence, frame, rgb_sequence, rgb_frame) in enumerate(calls):
        guard(plan)
        begin = time.monotonic()
        torch.manual_seed(20261101);torch.cuda.manual_seed_all(20261101);np.random.seed(20261101)
        torch.cuda.reset_peak_memory_stats()
        fixture = fixture_roots['original' if chain == 'reset_return' else chain]
        rgbpath = fixture/sequence/'visible'/f'{frame:06d}.jpg'
        rgb_sha = sha(rgbpath)
        files[str(rgbpath)] = rgb_sha
        image = Image.open(rgbpath).convert('RGB')
        rgb = np.asarray(image)
        generator.predictor.reset_image()
        torch.cuda.synchronize()
        sam_start = time.monotonic()
        with torch.inference_mode():
            masks = generator.generate(rgb)
        torch.cuda.synchronize()
        sam_seconds = time.monotonic()-sam_start
        union = np.zeros(rgb.shape[:2], dtype=bool)
        for mask in masks:
            union |= mask['segmentation']
        mask_array = (union*255).astype(np.uint8)
        maskpath = output/f'{index:02d}_mask.png'
        Image.fromarray(mask_array).save(maskpath, format='PNG')
        auxpath = output/f'{index:02d}_aux.json'
        write(auxpath, dict(sequence=sequence, frame=frame, rgb=str(rgbpath), mask_path=str(maskpath),
            caption=configuration['fixed_caption'], edit='turn the visible image of '+configuration['fixed_caption']+' into infrared',
            mask_union_fraction=float(union.mean())))
        task = dict(sequence=sequence, frame=frame, rgb=str(rgbpath), aux=str(auxpath), native=512,
            mode='letterbox', input_size=[512,512], seed=20261101, steps=25,
            cfg_text=7.5, cfg_image=1.5, cfg_seg=1.5, fixed_caption=configuration['fixed_caption'])
        torch.cuda.synchronize();inference_start=time.monotonic()
        generated, preprocessing = diff_worker.run(loaded, image, task)
        torch.cuda.synchronize();inference_seconds=time.monotonic()-inference_start
        gray = cv2.cvtColor(np.asarray(generated.convert('RGB')), cv2.COLOR_RGB2GRAY)
        irpath = output/f'{index:02d}_IR.png'
        Image.fromarray(gray).save(irpath, format='PNG', compress_level=6)
        cachepath = Path(mft['generated_ir_root'])/'train'/rgb_sequence/'infrared_diffv2ir'/f'{rgb_frame:06d}.png'
        samcache = Path(mft['generated_ir_root'])/'train'/rgb_sequence/'sam_mask'/f'{rgb_frame:06d}.png'
        # Timer ends before this extra read-only cache comparison.
        generation_with_local_IO_seconds = time.monotonic()-begin
        old = np.asarray(Image.open(cachepath).convert('L'))
        oldmask = np.asarray(Image.open(samcache).convert('L'))
        difference = np.abs(gray.astype(np.int16)-old.astype(np.int16))
        files[str(cachepath)], files[str(samcache)] = sha(cachepath), sha(samcache)
        arrays[chain,frame] = (gray.copy(), mask_array.copy())
        row = dict(index=index, chain=chain, declared_sequence=sequence, frame=frame,
            actual_RGB_sequence=rgb_sequence, actual_RGB_frame=rgb_frame, RGB_sha256=rgb_sha,
            IR_png_sha256=sha(irpath), IR_pixel_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
            SAM_png_sha256=sha(maskpath), SAM_pixel_sha256=hashlib.sha256(mask_array.tobytes()).hexdigest(),
            original_fixed_IR_sha256=files[str(cachepath)], original_fixed_SAM_sha256=files[str(samcache)],
            exact_cached_IR_pixels=bool(np.array_equal(gray,old)), exact_cached_SAM_pixels=bool(np.array_equal(mask_array,oldmask)),
            cache_IR_changed_pixels=int(np.count_nonzero(difference)), cache_IR_max_abs_difference=int(difference.max()),
            cache_IR_mean_abs_difference=float(difference.mean()), SAM_seconds=sam_seconds,
            DiffV2IR_seconds=inference_seconds, generation_with_local_IO_seconds=generation_with_local_IO_seconds,
            peak_own_allocated_GPU_bytes=torch.cuda.max_memory_allocated(), preprocessing=preprocessing)
        rows.append(row)
        write(output/f'{index:02d}_record.json', row)
        print(json.dumps(dict(status='GENERATED_TRAIN_ENGINEERING_FRAME', index=index, chain=chain,
            cache_pixel_parity=row['exact_cached_IR_pixels'], seconds=generation_with_local_IO_seconds)), flush=True)
    handle.remove()
    prefix = [dict(frame=f, exact_IR_pixels=bool(np.array_equal(arrays['original',f][0],arrays['changed_future',f][0])),
        exact_SAM_pixels=bool(np.array_equal(arrays['original',f][1],arrays['changed_future',f][1]))) for f in (1,2)]
    reset = dict(exact_IR_pixels=bool(np.array_equal(arrays['original',1][0],arrays['reset_return',1][0])),
        exact_SAM_pixels=bool(np.array_equal(arrays['original',1][1],arrays['reset_return',1][1])))
    guard(plan)
    require(all(sha(p) == h for p,h in files.items()), 'Original source or sampled dataset bytes changed')
    for name,path in weights.items():
        receipt=weight_receipts[name];stat=path.stat()
        require(stat.st_size==receipt['bytes'] and stat.st_mtime_ns==receipt['mtime_ns'], 'Original weight file changed')
    value=dict(status='COMPLETE_EIGHT_TRAIN_GENERATOR_CAUSALITY_PROBE_NOT_WHOLE_CACHE_REPRODUCTION_OR_MFT_ACCURACY',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows, prefix_comparisons=prefix,
        all_two_prefix_IR_and_SAM_equal=all(r['exact_IR_pixels'] and r['exact_SAM_pixels'] for r in prefix),
        sequence_reset=reset, sampled_original3_cache_IR_pixels_equal=all(r['exact_cached_IR_pixels'] for r in rows[:3]),
        sampled_original3_cache_SAM_pixels_equal=all(r['exact_cached_SAM_pixels'] for r in rows[:3]),
        source_hashes=files, weight_receipts=weight_receipts, model_loading_seconds=model_loading_seconds,
        full_future_suffix_set_on_disk_before_prefix=True, deliberate_CPU_lookahead_control=lookahead_control,
        physical_gpu='4', physical_gpu_uuid=plan['physical_gpu_uuid'], CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        actual_PID=os.getpid(), versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','Pillow','k-diffusion','transformers','segment-anything']},
        original_training_signalled=False, original_dataset_outputs_modified=False, large_weights_or_RGB_copied=False,
        total_generation_calls=8, fixed_generator_seed=20261101, additional_training_seeds=False, new_MFT_training=False,
        Test_GT_or_accuracy_read=False, supported_innovations=0, goal_met=False,
        seconds=time.monotonic()-started, peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        scope='Eight Train frames only, including counterfactual and reset repetitions. Negative cache parity is retained. Not evidence of all48070cached frames, calibrated uncertainty, formal FPS, generation-inclusive MFT runtime or tracking improvement. Local write costs differ from original NAS production. Fixed scalar PNGs used by ongoing MFT are never replaced.')
    write(output/'complete.json', value)
    print(json.dumps({k:value[k] for k in ['status','all_two_prefix_IR_and_SAM_equal','sequence_reset',
        'sampled_original3_cache_IR_pixels_equal','sampled_original3_cache_SAM_pixels_equal','seconds']}), flush=True)


if __name__ == '__main__':
    main()
