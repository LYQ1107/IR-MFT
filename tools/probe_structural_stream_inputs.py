#!/usr/bin/env python3
"""Verify live Train inputs against existing cache slices, without any fitting."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from tools.build_correlation_cache import read_pair,auxiliary_input
from tools.perceptual_gpu_handoff import future_gpus,previous_gpu_work_complete
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.structural_grid import structural_grid
from yolox.tracker.foveated_regions import exact_half_preproc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-encoder',action='store_true')
    args = parser.parse_args()
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    work = Path(plan['work_root'])
    config_path = ROOT/'configs/perceptual_structural_relative_20261123.json'
    config = json.loads(config_path.read_text())
    parent_path = ROOT/'configs/perceptual_correlation_repair_20261123.json'
    parent = json.loads(parent_path.read_text())
    calibration_path = work/'correlation_cache/calibration.json'
    calibration = json.loads(calibration_path.read_text())
    assert calibration['config_sha256'] == config['parent_plan_sha256'] == sha256(parent_path)
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.frozen_encoder:
        assert physical == future_gpus()[1]
        assert previous_gpu_work_complete(
            json.loads((work/'structural_nocycle_dev_state.json').read_text()),
            json.loads((work/'structural_nocycle_fit_state.json').read_text()))
    else:
        assert physical == ''
    suffix = 'GPU'+str(physical) if args.frozen_encoder else 'CPU'
    out = work/('structural_stream_inputs_'+suffix)
    out.mkdir(exist_ok=False)
    state = dict(status='REGISTERED_TRAIN_INPUT_PARITY_PROBE',indices=[0,96],
        source_sha256=sha256(__file__),base_config_sha256=sha256(config_path),
        calibration_sha256=sha256(calibration_path),physical_gpu=physical,
        frozen_encoder_check=args.frozen_encoder,no_fitting=True,Test_read=False)
    dump(out/'state.json',state)
    started = time.perf_counter()
    try:
        encoder = None
        if args.frozen_encoder:
            from yolox.exp import get_exp
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(.2,0)
            source = Path(parent['source_RGB_checkpoint'])
            assert sha256(source) == parent['source_RGB_sha256']
            with torch.random.fork_rng(devices=[]):
                model = get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'),None).get_model()
            saved = torch.load(source,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'],strict=True)
            del saved
            encoder = model.backbone.cuda().half().eval().requires_grad_(False)
            del model
            torch.cuda.reset_peak_memory_stats()
        rows = []
        with torch.no_grad():
            for sequence in protocol['train']:
                folder = work/'correlation_cache'/sequence/'features'
                meta_path = folder/'complete.json'
                meta = json.loads(meta_path.read_text())
                struct_path = work/'structural_cache'/sequence/'complete.json'
                structure = json.loads(struct_path.read_text())
                assert meta['config_sha256'] == sha256(parent_path)
                assert meta['source_RGB_sha256'] == parent['source_RGB_sha256']
                assert structure['parent_cache_manifest_sha256'] == sha256(meta_path)
                assert structure['specification_sha256'] == sha256(config_path)
                gt_path = Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
                assert sha256(gt_path) == meta['GT_sha256'] == protocol['derived_gt_sha256']['train/'+sequence]
                gt = np.loadtxt(gt_path,delimiter=',',ndmin=2)
                gt = gt[(gt[:,6] > 0)&(gt[:,7] == 1)]
                targets = np.load(meta['targets']['path'],mmap_mode='r')
                assert sha256(meta['targets']['path']) == meta['targets']['sha256']
                for index in state['indices']:
                    frame = meta['frame_ids'][index]
                    rgb,ir = read_pair(plan,sequence,frame)
                    gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
                    value,ratio = exact_half_preproc(rgb,config['training_input_hw'],
                                                    (.485,.456,.406),(.229,.224,.225))
                    views = {'rgb':value}
                    for cue,image in [('gray',gray),('ir',ir)]:
                        norm = calibration['statistics'][cue]
                        views[cue],_ = auxiliary_input(image,config['training_input_hw'],norm['mean'],norm['std'])
                    rebuilt = np.zeros((500,5),np.float32)
                    boxes = gt[gt[:,0] == frame,2:6].astype(np.float32).copy()
                    boxes[:,:2] += boxes[:,2:]/2
                    boxes *= ratio
                    assert len(boxes) <= 500
                    rebuilt[:len(boxes),1:] = boxes
                    np.testing.assert_array_equal(rebuilt,targets[index])
                    for cue,image in [('rgb',gray),('ir',ir)]:
                        descriptor,valid = structural_grid(image,config['training_input_hw'])
                        for key,array in [('descriptor',descriptor),('valid',valid)]:
                            cached = np.load(structure['outputs'][cue+'_'+key]['path'],mmap_mode='r')
                            np.testing.assert_array_equal(array,cached[index])
                    if encoder is not None:
                        for cue,value in views.items():
                            features = encoder(torch.from_numpy(value[None]).cuda())
                            for level,feature in enumerate(features,3):
                                cached = np.load(meta['outputs'][f'{cue}_P{level}']['path'],mmap_mode='r')
                                np.testing.assert_array_equal(feature[0].cpu().numpy(),cached[index])
                    rows.append(dict(sequence=sequence,frame=frame,index=index,
                        targets_all500rows_exact=True,RGB_IR_descriptors_and_validity_exact=True,
                        all3cue_3level_B1_FPNs_exact=args.frozen_encoder,
                        cache_manifest_sha256=sha256(meta_path),structure_manifest_sha256=sha256(struct_path)))
                    print(json.dumps(rows[-1]),flush=True)
        assert len(rows) == 12
        report = dict(state,status='COMPLETE_TRAIN_INPUT_PARITY_PROBE',rows=rows,seconds=time.perf_counter()-started,
            scope='Actual12Trainframes:2previouslycached frames/video,all6Train videos. Exact500row labels, '
                  'RGB/actualIR descriptors and validity; GPU mode also verifies3cues x3levels x12frames using '
                  'the original B1 frozen encoder. Compared cache slices are actual arrays, not synthetic fixtures. '
                  'Complete FPN banks are not rehashed by this probe; manifests and declared source checkpoint are checked.',
            pending_before_fitting='Streaming training-loop, initialization and gradient-step parity; '
                'registered matched modality controls and formal Test plan. This probe performs no optimization.',
            no_image_or_FPN_copy=True,source_RGB_sha256=parent['source_RGB_sha256'],
            GT_read='Only existing Train labels to reproduce cached targets',Dev_read=False,
            peak_own_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30 if encoder is not None else None,
            goal_met=False,supported_innovations=0)
        dump(out/'state.json',report)
        dump(ROOT/f'docs/fusion_research/PERCEPTUAL_MFT_20261123_TRAIN_STREAM_INPUT_PARITY_{suffix}.json',report)
        print(json.dumps(dict(status=report['status'],frames=12,CUDA=physical,no_fitting=True)),flush=True)
    except Exception as error:
        state.update(status='FAILED',error=repr(error))
        dump(out/'state.json',state)
        raise


if __name__ == '__main__':
    main()
