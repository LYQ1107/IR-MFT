#!/usr/bin/env python3
"""Same current pixels, old final cache batch shape vs online current padding128."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from cfp.online import CurrentFrameFrontend,proposal_crop_arrays
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('Newimmutable numeric proof required')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2);r=a.root;data=Path('/data3/liuyeqiang/datasets/mft26-ir-causal-v1');model,meta=CurrentFrameFrontend._load_encoder('/data3/liuyeqiang/IR-MFT_assets/cfp/phase1/reid_rgb_seed0/epoch12.pt',sha256('configs/cfp/protocol_v2.json'))
    with np.load('/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections/SN-001.npz') as source:offsets,boxes=source['offsets'],source['boxes']
    with np.load('/data3/liuyeqiang/IR-MFT_assets/cfp/phase2/causal_train_detection_features_postneck_seed0/SN-001.npz') as source:expected=source['rgb']
    rows=json.loads((r/'identity_two_crop_index_v1/SN-001.frames.json').read_text());size=len(boxes)%128;start=len(boxes)-size;measures=[]
    for frame in range(680,685):
        lo,hi=offsets[frame-1:frame+1];rgb=cv2.imread(str(data/'train'/rows[frame-1]['rgb_file']));arrays,valid=proposal_crop_arrays(rgb,None,boxes[lo:hi],rgb_only=True);full=CurrentFrameFrontend.encode_crops(type('Padding',(),{'encoder_batch':128})(),arrays[0],valid[0],model);ids=np.arange(lo,hi);selected=ids>=start;indices=ids[selected]-start;padded=torch.zeros(size,3,128,256);padded[indices]=arrays[0][selected]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):tail,_=model(padded.cuda(),embedding_stage='post_neck')
        tail=tail[indices].float().cpu().numpy();reference=expected[lo:hi][selected];error=float(np.max(np.abs(tail-reference))) if len(tail) else 0;assert error<3e-5
        current=full[selected];cos=np.sum(current*reference,axis=1)/(np.linalg.norm(current,axis=1)*np.linalg.norm(reference,axis=1));measures.append(dict(frame=frame,current_crops=len(full),old_cache_tail_rows=len(tail),tail_batch=size,tail_reconstruction_component_error=error,current128_vs_reference_component_error=float(np.max(np.abs(current-reference))) if len(current) else 0,current128_vs_reference_cosine_error=float(np.max(np.abs(1-cos))) if len(current) else 0))
    write_json(a.output,dict(status='VERIFIED_CURRENT_ONLY_OLD_TAIL_BATCH_RECONSTRUCTION',video='SN-001',proposals=len(boxes),old_tail_start=int(start),old_tail_batch=size,measurements=measures,current_only_RGB_no_future_images=True,checkpoint_sha256=meta['checkpoint_sha256'],cache_source_sha256=sha256('tools/cfp_cache_detection_features.py'),sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,'tools/cfp_cache_detection_features.py']),scope='Validationonly: samecurrentRGB/crops placedinoldcachefinalbatch38, otherpositionszero. ReconstructsoldcacheFPvalues; runtimecurrentpad128unchanged. Numericalbatchshape issue, notnewweights/semanticcoordinate/detection. No model/threshold tuning orfullrawFPSclaim.'))
    print(json.dumps(measures),flush=True)

if __name__=='__main__':main()
