#!/usr/bin/env python3
"""Unlabeled Dev pixel registration audit of actual RGB/new pseudo-IR pairs."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from tools.run_pseudo_ir_comparison import dump,sha256


def gradients(image):
    if image.ndim==2:image=image[:,:,None]
    values=[]
    for channel in range(image.shape[2]):
        plane=image[:,:,channel].astype(np.float32)/255
        x=cv2.Scharr(plane,cv2.CV_32F,1,0);y=cv2.Scharr(plane,cv2.CV_32F,0,1)
        values.append(cv2.magnitude(x,y))
    result=np.maximum.reduce(values)
    return np.clip(result/max(float(np.percentile(result,95)),1e-5),0,2).astype(np.float32)


def estimate(first,second):
    h,w=first.shape;window=cv2.createHanningWindow((w,h),cv2.CV_32F)
    (dx,dy),response=cv2.phaseCorrelate(first,second,window)
    return float(dx),float(dy),float(response)


def main():
    cv2.setNumThreads(1)
    rng=np.random.default_rng(10);signal=rng.random((64,96),dtype=np.float32)
    shifted=cv2.warpAffine(signal,np.float32([[1,0,5],[0,1,-3]]),(96,64))
    check=estimate(signal,shifted)
    if np.linalg.norm(np.array(check[:2])-[5,-3])>.5:raise ValueError('Registration direction smoke failed')
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root']);rows=[]
    for sequence in protocol['dev']:
        source=Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(sequence+'.npz')
        with np.load(source) as z:data={k:z[k] for k in z.files}
        samples=np.linspace(1,len(data['frame_ids']),20,dtype=int)
        for frame in samples:
            rgb_path=Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'
            ir_path=Path(plan['generated_ir_root'])/'dev'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
            rgb=cv2.imread(str(rgb_path));ir=cv2.imread(str(ir_path),cv2.IMREAD_UNCHANGED)
            if rgb is None or ir is None or rgb.shape[:2]!=ir.shape:raise ValueError('Invalid actual pair')
            small_rgb=cv2.resize(rgb,(960,540));small_ir=cv2.resize(ir,(960,540))
            global_shift=estimate(gradients(small_rgb),gradients(small_ir))
            a,b=data['offsets'][frame-1:frame+1];boxes=data['boxes'][a:b][data['scores'][a:b]>.6]
            for box in boxes:
                extent=box[2:]-box[:2];lower=np.maximum(np.floor(box[:2]-.15*extent),0).astype(int);upper=np.minimum(np.ceil(box[2:]+.15*extent),[rgb.shape[1],rgb.shape[0]]).astype(int)
                x1,y1=lower;x2,y2=upper
                if x2-x1<8 or y2-y1<8:continue
                first=gradients(cv2.resize(rgb[y1:y2,x1:x2],(96,64)))
                second=gradients(cv2.resize(ir[y1:y2,x1:x2],(96,64)))
                dx,dy,response=estimate(first,second)
                aligned=cv2.warpAffine(second,np.float32([[1,0,-dx],[0,1,-dy]]),(96,64))
                def corr(x,y):return float(np.corrcoef(x.ravel(),y.ravel())[0,1]) if x.std()>1e-6 and y.std()>1e-6 else 0.
                rows.append(dict(sequence=sequence,frame=int(frame),box=box.tolist(),dx_box_fraction=dx/96*1.3,dy_box_fraction=dy/64*1.3,response=response,correlation_before=corr(first,second),correlation_after=corr(first,aligned),global_shift_half_resolution=list(global_shift)))
    summaries={}
    for sequence in protocol['dev']:
        subset=[r for r in rows if r['sequence']==sequence];reliable=[r for r in subset if r['response']>=.15]
        summaries[sequence]=dict(ROIs=len(subset),response_ge_015=len(reliable),median_abs_shift_fraction=np.median([[abs(r['dx_box_fraction']),abs(r['dy_box_fraction'])] for r in reliable],axis=0).tolist(),median_correlation_before=float(np.median([r['correlation_before'] for r in reliable])),median_correlation_after=float(np.median([r['correlation_after'] for r in reliable])),large_shift_fraction=float(np.mean([max(abs(r['dx_box_fraction']),abs(r['dy_box_fraction']))>.1 for r in reliable])))
    dump(w/'crossmodal_alignment_audit.json',dict(status='COMPLETE_UNLABELED_DEV_ALIGNMENT_AUDIT',scope='Current RGB and actual new pseudo-IR gradient phase correlation only; no GT, Test labels or tracking scores. Increased registration correlation is not proof of HOTA improvement.',direction_smoke=check,summaries=summaries,rows=rows,source_sha256=sha256(__file__)))
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
