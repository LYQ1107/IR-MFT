#!/usr/bin/env python3
"""Per-component throughput, checked against immutable37-stream Dev prefixes."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from tools.build_correlation_cache import auxiliary_input
from tools.extract_structural_relative import detections
from tools.structural_grid import structural_grid
from tools.structural_relative_inference import decoded_relative_response
from tools.indexed_postprocess import indexed_postprocess
from yolox.models.frozen_head_repair import head_predictions
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.exp import get_exp
from yolox.models.structural_factorial_repair import StructuralFactorial,FLAGS
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.foveated_regions import exact_half_preproc
from yolox.utils import postprocess


def inputs(plan, config, calibration, sequence, frames, cue, image_cache=None):
    values = {'rgb': []}; available = []; ratios = []
    descriptors = {} if cue == 'baseline' else {'rgb': []}
    validity = {key: [] for key in descriptors}
    if cue not in ['rgb', 'baseline']:
        values[cue] = []
    if cue == 'ir':
        descriptors['ir'] = []; validity['ir'] = []
    for frame in frames:
        rgb_path = Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'
        rgb = cv2.imread(str(rgb_path)) if image_cache is None else image_cache[sequence,frame,'rgb']
        assert rgb is not None
        value, ratio = exact_half_preproc(rgb, config['inference_input_hw'],
                                        (.485,.456,.406), (.229,.224,.225))
        values['rgb'].append(value); ratios.append(ratio)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if cue != 'baseline' else None
        if cue == 'ir':
            path = Path(plan['generated_ir_root'])/'dev'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
            image = cv2.imread(str(path), -1) if image_cache is None else image_cache[sequence,frame,'ir']
            assert image is not None and image.ndim == 2 and image.dtype == np.uint8 and image.shape == rgb.shape[:2]
        else:
            image = gray
        if cue in ['baseline', 'rgb']:
            available.append(True)
        else:
            norm = calibration['statistics'][cue]
            value, _ = auxiliary_input(image, config['inference_input_hw'], norm['mean'], norm['std'])
            values[cue].append(value); available.append(bool(image.max() != image.min()))
        if cue != 'baseline':
            for view in descriptors:
                desc, valid = structural_grid(image if view == 'ir' else gray, config['inference_input_hw'])
                descriptors[view].append(desc); validity[view].append(valid)
    def transfer(collection):
        return {key: torch.from_numpy(np.stack(values)).cuda() for key, values in collection.items()}
    return dict(inputs=transfer(values), descriptors=transfer(descriptors), validity=transfer(validity)), ratios, available


def predict(model, adapter, arrays, ratios, available, cue, role):
    features = {name: model.backbone(value) for name,value in arrays['inputs'].items()}
    teacher = model.head(features['rgb'])
    if adapter is None:
        outputs = postprocess(teacher, 1, .1, .7)
    else:
        teacher_raw = head_predictions(model.head, features['rgb'])[0]
        scores = []; offset = 0
        for value in features['rgb']:
            count = value.shape[-2]*value.shape[-1]
            score = teacher[:,offset:offset+count,4]*teacher[:,offset:offset+count,5]
            scores.append(score.reshape(len(ratios),1,*value.shape[-2:])); offset += count
        view = 'ir' if cue == 'ir' else 'rgb'
        d, v = arrays['descriptors'], arrays['validity']
        common = (features['rgb'], features[cue], scores, d['rgb'], d[view], v['rgb'], v[view],
                  torch.tensor(available, device='cuda'))
        current = head_predictions(model.head, adapter(*common))[0]
        null = head_predictions(model.head, adapter(*common, null=True))[0]
        output = decoded_relative_response(current, null, teacher_raw, teacher, centered=adapter.centered,
                                            bound=2.)
        assert torch.equal(output[:,:,:4], teacher[:,:,:4])
        outputs = indexed_postprocess(output,1,.1,.7)[0]
    return [detections(value,ratio) for value,ratio in zip(outputs,ratios)]


def measure(model, adapter, plan, config, calibration, cue, role, source_key, image_cache=None):
    work = Path(plan['work_root']); sequence_rows = []
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for sequence in ['BT-005', 'SN-013']:
        source = work/'factorial_observations/structural_relative_predictions/dev'/sequence/'prefix_check_32'/(source_key+'.npz')
        with np.load(source) as z:
            reference = {k: z[k] for k in ['frame_ids','offsets','boxes','scores']}
        tracker = BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),frame_rate=25)
        elapsed = 0.; input_elapsed=0.; compute_elapsed=0.; tracking_elapsed=0.
        count = 0; output_rows = 0; checked = 0
        # Frames1..8 warm up and initialize tracking;9..32 are timed. Batches
        # exactly match the previously frozen producer's independent B4 layout.
        for first in range(1,33,4):
            frames = list(range(first,first+4)); torch.cuda.synchronize(); start = time.perf_counter()
            arrays, ratios, available = inputs(plan,config,calibration,sequence,frames,cue,image_cache)
            torch.cuda.synchronize(); input_end=time.perf_counter()
            values = predict(model,adapter,arrays,ratios,available,cue,role)
            torch.cuda.synchronize(); compute_end=time.perf_counter()
            for value in values:
                tracks = tracker.update(value,(1,1),(1,1))
                output_rows += sum(int(t.tlwh[2]*t.tlwh[3]>100) for t in tracks)
            torch.cuda.synchronize(); end=time.perf_counter();duration=end-start
            if first > 8:
                elapsed += duration; count += len(frames)
                input_elapsed+=input_end-start;compute_elapsed+=compute_end-input_end
                tracking_elapsed+=end-compute_end
            # Correctness comparison is outside the timed section. No GT or
            # metric evaluation is used by the benchmark.
            for frame,value in zip(frames,values):
                lo,hi = reference['offsets'][frame-1:frame+1]
                original = np.c_[reference['boxes'][lo:hi],reference['scores'][lo:hi]]
                np.testing.assert_array_equal(value,original); checked += 1
            del arrays, values
        assert count == 24 and checked == 32
        sequence_rows.append(dict(sequence=sequence,timed_frames=count,seconds=elapsed,
            FPS=count/elapsed,warmup_frames=8,source32_arrays_exact=True,
            source_packet_sha256=sha256(source),tracking_output_rows_including_warmup=output_rows,
            input_preparation_transfer_seconds=input_elapsed,NN_NMS_transfer_seconds=compute_elapsed,
            tracking_seconds=tracking_elapsed,NN_NMS_FPS=count/compute_elapsed))
    seconds = sum(r['seconds'] for r in sequence_rows)
    compute_seconds=sum(r['NN_NMS_transfer_seconds']for r in sequence_rows)
    return dict(per_sequence=sequence_rows,timed_frames=48,seconds=seconds,FPS=48/seconds,
                NN_NMS_FPS=48/compute_seconds,
                peak_own_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_own_reserved_GPU_GiB=torch.cuda.max_memory_reserved()/2**30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory-inputs-and-components', action='store_true')
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    from tools.perceptual_gpu_handoff import POLICY_PATH, future_gpus, previous_gpu_work_complete
    assert physical == future_gpus()[0]
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    fits = json.loads((work/'structural_factorial_fit_state.json').read_text())
    dev = json.loads((work/'structural_factorial_dev_state.json').read_text())
    assert fits['status'] == 'COMPLETE'
    assert dev['producers']['BT-005']['status'] == 'COMPLETE'
    next_stage = json.loads((work/'structural_factorial_next_stage_state.json').read_text())
    assert next_stage['status'] == 'WAITING_FOR_COMPLETE32DEV_AND_ANALYSIS_AND_RUNTIME'
    config_path = ROOT/'configs/perceptual_structural_relative_20261123.json'
    config = json.loads(config_path.read_text())
    phase_path = ROOT/'configs/perceptual_structural_factorial_20261123.json'
    phase = json.loads(phase_path.read_text())
    assert phase['base_config_sha256'] == sha256(config_path)
    parent_path = ROOT/'configs/perceptual_correlation_repair_20261123.json'
    parent = json.loads(parent_path.read_text())
    calibration = json.loads((work/'correlation_cache/calibration.json').read_text())
    assert calibration['config_sha256'] == sha256(parent_path)
    suffix = '_MEMORY_COMPONENTS' if args.memory_inputs_and_components else ''
    output = ROOT/('docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_FACTORIAL_SINGLE_METHOD_RUNTIME'+suffix+'.json')
    assert not output.exists()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    checkpoint = Path(parent['source_RGB_checkpoint'])
    assert sha256(checkpoint) == parent['source_RGB_sha256']
    model = get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'), None).get_model()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True); del saved
    model = model.cuda().half().eval().requires_grad_(False)
    base_parameters = sum(p.numel() for p in model.parameters())
    image_cache = None; preload_seconds = 0.
    if args.memory_inputs_and_components:
        start = time.perf_counter(); image_cache = {}
        for sequence in ['BT-005', 'SN-013']:
            for frame in range(1,33):
                rgb = cv2.imread(str(Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'))
                ir = cv2.imread(str(Path(plan['generated_ir_root'])/'dev'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'),-1)
                assert rgb is not None and ir is not None and ir.ndim == 2 and ir.dtype == np.uint8 and ir.shape == rgb.shape[:2]
                image_cache[sequence,frame,'rgb'] = rgb; image_cache[sequence,frame,'ir'] = ir
        preload_seconds = time.perf_counter()-start
    selected = [('RGB_fresh_baseline','baseline',None,'CG3_source_RGB')]
    selected += [(f'FCT_{bits}_e06_raw','ir',bits,f'FCT_ir_{bits}_s0_e06_raw') for bits in ['000','001','010','100']]
    selected += [('FCT_111_e06_raw','ir','111','FCT_primary_IR111_raw')]
    records = []
    with torch.inference_mode():
        for identifier, cue, variant, key in selected:
            if variant is None:
                adapter = None; extra = 0; weight_hash = None
            else:
                a = config['architecture']
                common = dict(latent=a['latent_channels'],radius=a['local_radius'],temperature=a['structural_temperature'],
                    bound=a['feature_residual_bound_RMS_fraction'],risk_threshold=a['teacher_risk_threshold'],
                    tolerance=a['forward_backward_tolerance_cells'])
                adapter = StructuralFactorial(components=variant,**common)
                if variant == '111':
                    old_phase = json.loads((ROOT/'configs/perceptual_structural_nocycle_20261123.json').read_text())
                    path = Path(old_phase['primary_IR']['checkpoint'])
                else:
                    path = work/'structural_factorial_fit'/f'ir_{variant}_s0/epoch06.pt'
                saved = torch.load(path, map_location='cpu', weights_only=False)
                assert saved['config_sha256'] == sha256(config_path) and saved['completed_epochs'] == 6
                assert saved['metadata']['seed'] == 0 and saved['metadata']['cue'] == cue
                assert saved['metadata']['source_RGB_sha256'] == parent['source_RGB_sha256']
                if variant != '111':
                    assert saved['metadata']['component_flags'] == FLAGS[variant]
                    assert saved['metadata']['factorial_plan_sha256'] == sha256(phase_path)
                adapter.load_state_dict(saved['adapter'], strict=True); del saved
                adapter = adapter.cuda().float().eval().requires_grad_(False)
                extra = sum(p.numel() for p in adapter.parameters()); assert extra == 145600
                weight_hash = sha256(path)
            result = measure(model,adapter,plan,config,calibration,cue,'relative_semantics_raw',key,image_cache)
            records.append(dict(id=identifier,cue=cue,variant=variant,epoch=6 if variant else None,
                base_parameters=base_parameters,adapter_parameters=extra,total_parameters=base_parameters+extra,
                encoder_passes=1 if cue in ['baseline','rgb'] else 2,checkpoint_sha256=weight_hash,**result))
            print(json.dumps(records[-1]),flush=True)
            del adapter; gc.collect(); torch.cuda.empty_cache()
    dump(output,dict(status='COMPLETE_SINGLE_METHOD_STRUCTURAL_B4_THROUGHPUT',rows=records,
        physical_gpu=physical,gpu_handoff_policy_sha256=sha256(POLICY_PATH),
        gpu_handoff_helper_sha256=sha256(ROOT/'tools/perceptual_gpu_handoff.py'),
        fixed_fitted_seed=0,additional_seed_work=False,
        raw_images_preloaded_in_RAM=args.memory_inputs_and_components,preload_seconds_excluded=preload_seconds,
        raw_image_RAM_GiB=0. if image_cache is None else sum(v.nbytes for v in image_cache.values())/2**30,
        unique_Dev_frames=64,frame_method_comparisons=384,
        scope='Six separate methods, each2Devvideos x32frames;8warmup+24timed/video. B4throughput, notB1latency. '
              'Includes original image preprocessing, live descriptor extraction, transfers, requiredshared encoder passes, '
              'current/null/raw teacher heads, NMS and ByteTrack. Generation excluded. Memory-input mode excludes fileIO/preload. '
              'Per-method inference includes oneadapter only. Other project load and filesystem caches are uncontrolled. '
              'Every raw frame packet bit-exact to immutable prefixes or completed original IR packets; '
              'prefix/full37agreement is independently required by the Dev controller before accuracy. '
              'All adapters are deliberately profiled at6epochs for architecture cost, not selected using accuracy.',
        source_sha256=sha256(__file__),base_config_sha256=sha256(config_path),factorial_plan_sha256=sha256(phase_path),
        GT_read=False,accuracy_selection=False))


if __name__ == '__main__':
    main()
