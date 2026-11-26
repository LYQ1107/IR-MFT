#!/usr/bin/env python3
"""Leave-one-Train-video-out descriptor diagnosis; no Dev/Test or GPU input."""
import datetime
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
from tools.run_pseudo_ir_comparison import sha256
from tools.train_roi_identity import identities_and_boxes


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    started = time.monotonic()
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    plan = read(plan_path)
    protocol = read(ROOT/plan['protocol'])
    work = Path(plan['work_root'])
    base = work/'fisher_memory_20261127'
    path = base/'Train_leave_one_video_out_transfer.json'
    assert not path.exists()
    fit = read(base/'fit_complete.json')
    assert fit['Dev_fit_rows'] == fit['Test_fit_rows'] == 0
    source_hashes = {str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),
        ROOT/'tools/mft_identity_projection.py',ROOT/'tools/fusion_memory_features.py',
        ROOT/'tools/train_roi_identity.py',plan_path,ROOT/plan['protocol']]}
    arrays = {k:[]for k in ['rgb','ir','video','identity','frame']}
    for video,seq in enumerate(protocol['train']):
        folder = work/'correlation_cache'/seq/'features'
        meta = read(folder/'complete.json')
        assert sha256(folder/'complete.json') == fit['Train_sources'][seq]['metadata_sha256']
        gt_path = Path(plan['evaluation']['GT_root'])/'train'/seq/'gt/gt.txt'
        assert sha256(gt_path) == fit['Train_sources'][seq]['GT_sha256']
        gt = np.loadtxt(gt_path,delimiter=',',ndmin=2)
        targets = np.load(folder/'targets.npy',mmap_mode='r')
        assert sha256(folder/'targets.npy') == fit['Train_sources'][seq]['targets_sha256']
        maps = {}
        for cue in ['rgb','ir']:
            p = folder/(cue+'_P3.npy')
            assert sha256(p) == fit['Train_sources'][seq]['P3_sha256'][cue]
            maps[cue] = np.load(p,mmap_mode='r')
        for i,frame in enumerate(meta['frame_ids']):
            labels,boxes = identities_and_boxes(gt,frame,meta['ratios'][i],targets[i])
            if not len(labels):
                continue
            for cue in maps:
                feature = torch.from_numpy(np.asarray(maps[cue][i],dtype=np.float32)[None])
                values,valid = pool_p3(feature,boxes,tuple(meta['image_shapes'][i]),(480,864),meta['ratios'][i])
                assert bool(valid.all())
                arrays[cue].append(values.numpy().copy())
            arrays['video'].append(np.full(len(labels),video,np.int64))
            arrays['identity'].append(labels)
            arrays['frame'].append(np.full(len(labels),frame,np.int64))
        print(json.dumps(dict(stage='Train_ROI_reuse',sequence=seq,seconds=time.monotonic()-started)),flush=True)
    arrays = {k:np.concatenate(v,axis=0)for k,v in arrays.items()}
    keep = np.zeros(len(arrays['video']),bool)
    for video in np.unique(arrays['video']):
        for label in np.unique(arrays['identity'][arrays['video']==video]):
            selected = (arrays['video']==video)&(arrays['identity']==label)
            if selected.sum() >= 3:
                keep |= selected
    assert int(keep.sum()) == fit['actual_fitted_ROI_observations']
    arrays = {k:v[keep]for k,v in arrays.items()}
    features = dict(raw_RGB=arrays['rgb'],raw_IR=arrays['ir'],
                    raw_paired=paired_features(arrays['rgb'],arrays['ir']))
    rows = []
    for held,sequence in enumerate(protocol['train']):
        train = arrays['video'] != held
        test = ~train
        views = {key:value[test]for key,value in features.items()}
        fitting = {}
        for name,key in [('LOO_RGB_Fisher','raw_RGB'),('LOO_IR_Fisher','raw_IR'),('LOO_paired_Fisher','raw_paired')]:
            projection,details = fit_fisher(features[key][train],arrays['video'][train],
                arrays['identity'][train],dimensions=64,ridge_fraction=.01)
            assert held not in [d['video']for d in details['video_counts']]
            views[name] = transform(features[key][test],projection)
            fitting[name] = dict(fit_videos=[protocol['train'][int(i)]for i in np.unique(arrays['video'][train])],
                held_video_features_or_labels_enter_fit=False,fit_observations=int(train.sum()),
                dimensions=details['dimensions'],fitted_identity_classes=details['video_local_identity_classes'])
        result = {name:temporal_retrieval(value,arrays['video'][test],arrays['identity'][test],arrays['frame'][test])
                  for name,value in views.items()}
        rows.append(dict(held_Train_video=sequence,held_observations=int(test.sum()),
                         early_gallery_late_query_retrieval=result,fitting=fitting))
        print(json.dumps(dict(stage='LOO_complete',held_video=sequence,
            top1={k:v['equal_video_top1_percent']for k,v in result.items()})),flush=True)
    summary = {key:float(np.mean([row['early_gallery_late_query_retrieval'][key]['equal_video_top1_percent']for row in rows]))
               for key in rows[0]['early_gallery_late_query_retrieval']}
    for source,expected in source_hashes.items():
        assert sha256(ROOT/source) == expected
    value = dict(status='COMPLETE_SIX_FOLD_TRAIN_VIDEO_TRANSFER_DIAGNOSIS',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,equal_video_top1_percent=summary,
        source_hashes=source_hashes,original_deployed_Train_fit_sha256=sha256(base/'fit_complete.json'),
        deployed_parameters_or_association_grid_changed=False,source_images_or_FPN_copied=False,
        descriptor_arrays_saved=False,fit_model_states_saved=False,GPU_work=False,Dev_rows=0,Test_rows=0,
        scope='Held video descriptors and IDs excluded from transform fit; only the held Train video early templates and late queries are used to compute retrieval. Six Train videos only; not independent Test, not tracking HOTA, not a native-IR novelty claim.',
        no_additional_random_seeds=True,supported_innovations=0,goal_met=False,seconds=time.monotonic()-started)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],equal_video_top1_percent=summary,seconds=value['seconds'])),flush=True)


if __name__=='__main__':
    main()
