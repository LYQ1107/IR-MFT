#!/usr/bin/env python3
"""Isolated current-only JPM deployment/pilot pixel and coordinate agreement."""
import argparse,json,os
from pathlib import Path
import cv2,numpy as np,torch
from cfp.transreid_native_encoder import CurrentJpmIdentityEncoder
from cfp.dinov2_native_encoder import current_proposal_crops
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU-only exact completedJPM model deployment proof required')
    torch.set_num_threads(2);r=a.root;training=r/'transreid_dino_jpm_SN-015_s0_v1';results=[]
    for video in ['BT-001','SN-015']:
        encoder=CurrentJpmIdentityEncoder('/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2',r/'external_models/dinov2_vits14_v1/dinov2_vits14_pretrain.pth',r/'dinov2_small_inference_source_smoke_v1.json',training,video,device='cpu')
        records=json.loads((training/video/'records.json').read_text());valid=np.load(r/'dinov2_two_crop_evidence_v1'/video/'valid.npy');frame=min(x['frame'] for i,x in enumerate(records) if x['view']=='detector' and valid[i]);chosen=[i for i,x in enumerate(records) if x['view']=='detector' and x['frame']==frame and valid[i]][:4]
        with np.load('/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections/'+video+'.npz') as det:boxes=det['boxes'][[records[i]['source_index'] for i in chosen]].copy()
        path=json.loads((r/'identity_two_crop_index_v1'/(video+'.frames.json')).read_text())[frame-1]['rgb_file'];rgb=cv2.imread('/data3/liuyeqiang/datasets/mft26-ir-causal-v1/train/'+path);assert rgb is not None
        crop,_,_=current_proposal_crops(rgb,boxes);pilot=np.load(r/'identity_pilot_crop_cache_v1'/(video+'.npy'),mmap_mode='r');assert np.array_equal(crop.numpy(),np.array(pilot[chosen]).astype(np.float32))
        observed=encoder.encode(rgb,np.r_[boxes,np.array([[-10,-10,-5,-5]],np.float32)]);expected=np.load(training/video/'embedding.npy',mmap_mode='r')[chosen];current=observed['embedding'][:-1];maximum=float(np.abs(current-expected).max());cosine=float(np.abs(1-(current*expected).sum(-1)).max());assert maximum<3e-5 and cosine<1e-5,(maximum,cosine)
        assert not observed['valid'][-1] and np.count_nonzero(observed['embedding'][-1])==0
        results.append(dict(video=video,current_frame=frame,current_proposals=len(chosen),invalid_proposals_preserved=1,pilot_pixel_bytes_equal=True,max_component_difference=maximum,max_cosine_difference=cosine,coordinate_sha256=encoder.metadata['coordinate_sha256']))
    write_json(a.output,dict(status='VERIFIED_ACTUAL_CURRENT_JPM_ENCODER_AND_EXACT_PILOT_PIXELS',results=results,training_manifest_sha256=sha256(training/'manifest.json'),sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,'cfp/transreid_native_encoder.py','cfp/transreid_dino_jpm.py','third_party/transreid_jpm/shuffle.py','cfp/dinov2_native_encoder.py','cfp/masked_identity_checkpoint.py']),scope='Separate aftertraining numericalproof, pilotrecords/validity/crops usedonlyascomparison targets, never deploymentinputs. CurrentonlyRGB/actualproposals andfixedfit-onlyBNneck forward withpad128, no GTlabels/histories/future. Fourorlesssampledactualproposals pervideo plusinvalidrow, notallframe/nativeMOT/rawFPS success; CPUvsrecordedGPUFP32 numericalagreement thresholds3e-5component/1e-5cosine explicit, no byte-equaldescriptor claim.'))
    print(json.dumps(results))

if __name__=='__main__':main()
