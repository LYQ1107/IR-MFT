#!/usr/bin/env python3
"""One-method batch4 throughput, verified against frozen complete Dev packets."""
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
from tools.extract_single_seed_geometry import detections
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.exp import get_exp
from yolox.models.local_correlation_repair import CorrelationRepair
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.foveated_regions import exact_half_preproc
from yolox.utils import postprocess


def inputs(plan, config, calibration, sequence, frames, cue):
    values = {'rgb': []}; available = []; ratios = []
    if cue not in ['rgb', 'baseline']:
        values[cue] = []
    for frame in frames:
        path = Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'
        rgb = cv2.imread(str(path)); assert rgb is not None
        array, ratio = exact_half_preproc(rgb, config['inference_input_hw'],
                                        (.485, .456, .406), (.229, .224, .225))
        values['rgb'].append(array); ratios.append(ratio)
        if cue in ['rgb', 'baseline']:
            available.append(True)
        else:
            if cue == 'ir':
                path = Path(plan['generated_ir_root'])/'dev'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
                image = cv2.imread(str(path), -1)
                assert image is not None and image.dtype == np.uint8 and image.shape == rgb.shape[:2]
            else:
                image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            norm = calibration['statistics'][cue]
            array, _ = auxiliary_input(image, config['inference_input_hw'], norm['mean'], norm['std'])
            values[cue].append(array); available.append(bool(image.max() != image.min()))
    return {k: torch.from_numpy(np.stack(v)).cuda() for k,v in values.items()}, ratios, available


def predict(model, adapter, arrays, ratios, available, cue, role):
    features = {name: model.backbone(value) for name, value in arrays.items()}
    teacher = model.head(features['rgb'])
    if adapter is None:
        output = teacher
    else:
        scores = []; offset = 0
        for value in features['rgb']:
            count = value.shape[-2]*value.shape[-1]
            p = teacher[:,offset:offset+count,4]*teacher[:,offset:offset+count,5]
            scores.append(p.reshape(len(ratios),1,*value.shape[-2:])); offset += count
        repaired = adapter(features['rgb'], features[cue], scores,
                           torch.tensor(available,device='cuda'))
        output = model.head(repaired)
        if role == 'rgb_boxes':
            output[:,:,:4] = teacher[:,:,:4]
        elif role == 'rgb_scores':
            output[:,:,4:] = teacher[:,:,4:]
        else:
            assert role == 'full'
    return [detections(out, ratio) for out,ratio in zip(postprocess(output,1,.1,.7),ratios)]


def measure(model, adapter, plan, config, calibration, cue, role, source_key):
    work = Path(plan['work_root']); sequence_rows = []
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for sequence in ['BT-005', 'SN-013']:
        source = work/'single_seed_geometry_predictions/dev'/sequence/(source_key+'.npz')
        with np.load(source) as z:
            reference = {k: z[k] for k in ['frame_ids','offsets','boxes','scores']}
        tracker = BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),frame_rate=25)
        elapsed = 0.; count = 0; output_rows = 0; checked = 0
        # Frames1..8 warm up and initialize tracking;9..40 are timed. Batches
        # exactly match the previously frozen producer's independent B4 layout.
        for first in range(1,41,4):
            frames = list(range(first,first+4)); torch.cuda.synchronize(); start = time.perf_counter()
            arrays, ratios, available = inputs(plan,config,calibration,sequence,frames,cue)
            values = predict(model,adapter,arrays,ratios,available,cue,role)
            for value in values:
                tracks = tracker.update(value,(1,1),(1,1))
                output_rows += sum(t.tlwh[2]*t.tlwh[3]>100 for t in tracks)
            torch.cuda.synchronize(); duration = time.perf_counter()-start
            if first > 8:
                elapsed += duration; count += len(frames)
            # Correctness comparison is outside the timed section. No GT or
            # metric evaluation is used by the benchmark.
            for frame,value in zip(frames,values):
                lo,hi = reference['offsets'][frame-1:frame+1]
                original = np.c_[reference['boxes'][lo:hi],reference['scores'][lo:hi]]
                np.testing.assert_array_equal(value,original); checked += 1
            del arrays, values
        assert count == 32 and checked == 40
        sequence_rows.append(dict(sequence=sequence,timed_frames=count,seconds=elapsed,
            FPS=count/elapsed,warmup_frames=8,source40_arrays_exact=True,
            source_packet_sha256=sha256(source),tracking_output_rows_including_warmup=output_rows))
    seconds = sum(r['seconds'] for r in sequence_rows)
    return dict(per_sequence=sequence_rows,timed_frames=64,seconds=seconds,FPS=64/seconds,
                peak_own_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_own_reserved_GPU_GiB=torch.cuda.max_memory_reserved()/2**30)


def main():
    physical = os.environ.get('CUDA_VISIBLE_DEVICES'); assert physical in ['2','3']
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work = Path(plan['work_root'])
    assert json.loads((work/'CGA_formal_test_state.json').read_text())['status'] == 'COMPLETE'
    assert json.loads((work/'structural_relative_dev_state.json').read_text())['status'] == 'COMPLETE'
    spec_path = ROOT/'configs/perceptual_CGA_formal_test_20261123.json';spec = json.loads(spec_path.read_text())
    config_path = ROOT/'configs/perceptual_correlation_repair_20261123.json';config = json.loads(config_path.read_text())
    assert sha256(config_path) == spec['config_sha256']
    calibration = json.loads((work/'correlation_cache/calibration.json').read_text())
    assert calibration['config_sha256'] == sha256(config_path)
    output = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_CGA_SINGLE_METHOD_RUNTIME.json'
    assert not output.exists()
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.45,0)
    checkpoint = Path(config['source_RGB_checkpoint']);assert sha256(checkpoint) == config['source_RGB_sha256']
    model = get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'),None).get_model()
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True);del saved
    model = model.cuda().half().eval().requires_grad_(False)
    base_parameters = sum(p.numel() for p in model.parameters()); records = []
    selected = [None]+[next(r for r in spec['cases'] if r['id']=='CGA_'+cue+'_Dev_selected')
                       for cue in ['rgb','gray','ir']]
    models = {r['id']:r for r in spec['models']}
    with torch.inference_mode():
        for case in selected:
            if case is None:
                adapter=None;cue='baseline';role='full';key='CG3_source_RGB';identifier='RGB_fresh_baseline'
                extra=0;weight_hash=None
            else:
                row=models[case['model_id']];assert row['variant']=='full' and row['seed']==0
                a=config['architecture'];adapter=CorrelationRepair(latent=a['latent_channels'],radius=a['local_radius'],
                    temperature=a['temperature'],risk_gate=True,bound=a['maximum_residual_RMS_fraction'])
                path=Path(row['checkpoint']);saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['config_sha256']==sha256(config_path) and saved['metadata']['seed']==0
                adapter.load_state_dict(saved['adapter'],strict=True);del saved
                adapter=adapter.cuda().float().eval().requires_grad_(False)
                cue=case['cue'];role=case['head_role'];key=case['proposal_key'];identifier=case['id']
                extra=sum(p.numel() for p in adapter.parameters());weight_hash=sha256(path)
            result=measure(model,adapter,plan,config,calibration,cue,role,key)
            records.append(dict(id=identifier,cue=cue,head_role=role,base_parameters=base_parameters,
                adapter_parameters=extra,total_parameters=base_parameters+extra,
                encoder_passes=1 if cue in ['rgb','baseline']else 2,
                checkpoint_sha256=weight_hash,**result))
            print(json.dumps(records[-1]),flush=True)
            del adapter;gc.collect();torch.cuda.empty_cache()
    dump(output,dict(status='COMPLETE_SINGLE_METHOD_VERIFIED_B4_THROUGHPUT',rows=records,
        physical_gpu=physical,fixed_fitted_seed=0,additional_seed_work=False,
        scope='Each method separately: current-image read/preprocess/transfer, required '
              'shared-weight encoder passes, one adapter, actual raw head role, NMS, CPU '
              'ByteTrack and device synchronization. B4 throughput, not B1 online latency. '
              '8warmup+32timed frames per Dev video; all40 raw packets bit-exact to frozen '
              'full Dev. Comparison outside timing. File cache/hardware warmed; other '
              'project load is not controlled. One pass per method, not repeated seed fits.',
        source_sha256=sha256(Path(__file__)),selection_sha256=sha256(spec_path),
        GT_read=False,accuracy_selection=False))


if __name__=='__main__':
    main()
