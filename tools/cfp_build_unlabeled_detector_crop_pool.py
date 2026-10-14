#!/usr/bin/env python3
"""Four-fit-video detector crop pool; never select/read physical identity values."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--config',type=Path,default=Path('configs/cfp/phase_vii_unlabeled_detector_crop_pool_v1.json'));a=ap.parse_args()
    if a.output.exists():raise ValueError('New immutable fit-only unlabeled pool required')
    cfg=a.config;c=json.loads(cfg.read_text());r=a.root;crop_root=r/'identity_pilot_crop_cache_v1';feature_root=r/'dinov2_two_crop_evidence_v1';index=r/'identity_two_crop_index_v1';detection=Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections');cm=json.loads((crop_root/'manifest.json').read_text());fm=json.loads((feature_root/'manifest.json').read_text());im=json.loads((index/'manifest.json').read_text());dm=json.loads((detection/'manifest.json').read_text())
    if cm['status']!='COMPLETE_ALIGNED_PILOT_CROPS' or fm['status']!='COMPLETE_FROZEN_DINOV2_TWO_CROP_EVIDENCE' or im['status']!='COMPLETE_INDEX_ONLY' or dm['status']!='COMPLETE' or set(c['fit_videos'])&set(c['excluded_videos']):raise ValueError('Exact previouslyverified currentcrop/framepath sources and excludedinner/outer required')
    a.output.mkdir();report=dict(status='RUNNING',config_sha256=sha256(cfg),fit_videos=c['fit_videos'],excluded_videos=c['excluded_videos'],videos={},sources=snapshot_sources(a.output/'sources',[__file__,cfg]),scope='Onlyactualvalid detector-view crops fromfourfitvideos; records view/source_index/frame used, physical identity values neveraccessed/filtered/sampled. GTcrop rows unselected. CurrentRGB headers providegeometry, noannotation JSON/GTboxes/GTphysical IDs read. Detector/foundation historicalexposure remains explicit; legal existing studycrop sources, notexternalFishMAE data or officialreproduction.')
    for video in c['fit_videos']:
        source=feature_root/video;records=json.loads((source/'records.json').read_text());valid=np.load(source/'valid.npy');paths=json.loads((index/(video+'.frames.json')).read_text())
        if sha256(source/'records.json')!=fm['videos'][video]['records_sha256'] or sha256(source/'valid.npy')!=fm['videos'][video]['artifacts']['valid']['sha256'] or sha256(crop_root/(video+'.npy'))!=cm['videos'][video]['sha256'] or sha256(index/(video+'.frames.json'))!=im['videos'][video]['frame_paths_sha256'] or sha256(detection/(video+'.npz'))!=dm['caches'][video]['sha256']:raise ValueError('Immutablecurrent sourcebytes drift')
        with np.load(detection/(video+'.npz'),allow_pickle=False) as det:boxes=det['boxes'].copy();frame_of_proposal=np.repeat(det['frame_ids'],np.diff(det['offsets']))
        indices=[i for i,x in enumerate(records) if x['view']=='detector' and valid[i]];proposal=np.asarray([records[i]['source_index'] for i in indices],np.int64);frames=np.asarray([records[i]['frame'] for i in indices],np.int64)
        if not len(indices) or not np.array_equal(frame_of_proposal[proposal],frames):raise ValueError('Exact actualdetectorframe timestamps required')
        dimensions={}
        for frame in np.unique(frames):
            row=paths[int(frame)-1];assert row['frame_id']==frame
            relative=Path(row['rgb_file'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Currentimage path outside declaredsource')
            with Image.open(Path(im['data_root'])/'train'/relative) as image:w,h=image.size
            dimensions[int(frame)]=[h,w]
        output=a.output/(video+'.npz');np.savez_compressed(output,crop_indices=np.asarray(indices,np.int64),boxes=boxes[proposal],frames=frames,proposal_indices=proposal,dimensions=np.asarray([dimensions[int(f)] for f in frames],np.int32))
        report['videos'][video]=dict(crops=len(indices),frames=len(dimensions),pool_sha256=sha256(output),source_crops=str(crop_root/(video+'.npy')),source_crops_sha256=cm['videos'][video]['sha256'],records_sha256=sha256(source/'records.json'),validity_sha256=sha256(source/'valid.npy'),detection_sha256=sha256(detection/(video+'.npz')),frame_paths_sha256=sha256(index/(video+'.frames.json')))
        write_json(a.output/'manifest.json',report)
    report['status']='COMPLETE_FIT_ONLY_UNLABELED_DETECTOR_CROP_POOL';report['physical_identity_values_used']=False;report['GT_crop_values_used']=False;write_json(a.output/'manifest.json',report);print(json.dumps({v:e['crops'] for v,e in report['videos'].items()}))

if __name__=='__main__':main()
