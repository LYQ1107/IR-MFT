#!/usr/bin/env python3
"""Display actual new pseudo-IR and fixed crop masks without identity labels."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from yolox.tracker.foreground_transport import region_masks


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root'])
    examples=[('BT-005',300),('BT-005',1200),('SN-013',800),('SN-013',2200)]
    fig,axes=plt.subplots(4,4,figsize=(12,7),constrained_layout=True)
    for row,(sequence,frame) in enumerate(examples):
        with np.load(w/f'features/dev/{sequence}/new.npz') as data:
            a,b=data['offsets'][frame-1:frame+1];boxes=data['boxes'][a:b];scores=data['scores'][a:b]
        rgb=cv2.imread(str(Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'))
        ir=cv2.imread(str(Path(plan['generated_ir_root'])/'dev'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'),0)
        if rgb is None or ir is None:raise ValueError('Missing actual RGB/new-IR frame')
        valid=np.flatnonzero(scores>.6)
        intersection=np.maximum(np.minimum(boxes[valid,None,2:],boxes[None,valid,2:])-np.maximum(boxes[valid,None,:2],boxes[None,valid,:2]),0).prod(-1)
        area=np.maximum(boxes[valid,2:]-boxes[valid,:2],0).prod(-1)
        overlap=intersection/(area[:,None]+area[None]-intersection+1e-8);np.fill_diagonal(overlap,-1)
        first=int(valid[np.unravel_index(overlap.argmax(),overlap.shape)[0]]) if len(valid)>1 else int(valid[0])
        box=boxes[first];masks,quality=region_masks(rgb,ir,box)
        x1,y1=np.maximum(box[:2].astype(int),0);x2,y2=np.minimum(np.ceil(box[2:]).astype(int),[rgb.shape[1],rgb.shape[0]])
        values=[cv2.cvtColor(rgb[y1:y2,x1:x2],cv2.COLOR_BGR2RGB),ir[y1:y2,x1:x2],masks['rgb'],masks['ir']]
        for col,value in enumerate(values):
            axes[row,col].imshow(value,cmap='gray',vmin=0,vmax=1 if col>=2 else 255);axes[row,col].axis('off')
            if row==0:axes[row,col].set_title(['RGB crop','Actual new pseudo-IR','RGB contrast mask','IR contrast mask'][col])
        axes[row,0].text(0,-.08,f'{sequence} f{frame}; score{scores[first]:.2f}',transform=axes[row,0].transAxes,fontsize=9)
        axes[row,3].text(0,-.08,f'qRGB{quality[0]:.2f}, qIR{quality[1]:.2f}',transform=axes[row,3].transAxes,fontsize=9)
    fig.suptitle('Unlabeled Dev crop diagnosis: same detector box, actual November DiffV2IR')
    path=w/'foreground_crop_diagnosis.png';fig.savefig(path,dpi=140);plt.close(fig);print(path)


if __name__=='__main__':main()
