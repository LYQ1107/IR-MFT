#!/usr/bin/env python3
"""Small posthoc Train-only retrieval diagnosis; never an accuracy result."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.models.local_correlation_repair import local_readout


def support(target,height,width,image_hw):
    mask=np.zeros((height,width),bool)
    for row in target:
        if min(row[3:5])<=0:continue
        cx,cy,bw,bh=row[1:5]
        x1=max(0,int(np.floor((cx-bw/2)*width/image_hw[1])));x2=min(width,int(np.ceil((cx+bw/2)*width/image_hw[1])))
        y1=max(0,int(np.floor((cy-bh/2)*height/image_hw[0])));y2=min(height,int(np.ceil((cy+bh/2)*height/image_hw[0])))
        mask[y1:y2,x1:x2]=True
    return mask


def main():
    torch.set_num_threads(2)
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_correlation_repair_20261123.json';config=json.loads(config_path.read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());rows=[];sources={}
    for cue in ['rgb','gray','ir']:
        checkpoint=work/'correlation_fit'/f'{cue}_full_s0/epoch24.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['config_sha256']!=sha256(config_path):raise ValueError('Original matched model required')
        sources[cue]=dict(checkpoint_sha256=sha256(checkpoint));weights=saved['adapter']
        for seq in protocol['train']:
            directory=work/'correlation_cache'/seq/'features';meta=json.loads((directory/'complete.json').read_text())
            if meta['config_sha256']!=sha256(config_path)or meta['Dev_rows']or meta['Test_rows']:raise ValueError('Train only')
            target=np.load(directory/'targets.npy',mmap_mode='r')
            for level in [3,4,5]:
                rgb=np.load(directory/f'rgb_P{level}.npy',mmap_mode='r')
                auxiliary=rgb if cue=='rgb'else np.load(directory/f'{cue}_P{level}.npy',mmap_mode='r')
                weight=weights[f'levels.{level-3}.project.weight']
                with torch.no_grad():
                    for current,wrong in [(0,96),(96,0)]:
                        q=F.conv2d(torch.tensor(np.array(rgb[current:current+1]),dtype=torch.float32),weight)
                        values={}
                        for key,index in [('current',current),('wrong_same_video',wrong)]:
                            k=F.conv2d(torch.tensor(np.array(auxiliary[index:index+1]),dtype=torch.float32),weight)
                            _,diagnostics=local_readout(q,k,k,radius=2,temperature=.1)
                            values[key]=diagnostics[0,0].numpy()
                        h,w=values['current'].shape;mask=support(target[current],h,w,config['training_input_hw'])
                        invalid=support(target[wrong],h,w,config['training_input_hw']);valid=mask&~invalid
                        for scope,selection in [('current_GT_support',mask),('current_GT_without_wrong_GT_support',valid)]:
                            if not selection.any():continue
                            a=values['current'][selection];b=values['wrong_same_video'][selection]
                            rows.append(dict(cue=cue,sequence=seq,frame=int(meta['frame_ids'][current]),wrong_frame=int(meta['frame_ids'][wrong]),
                                level=level,scope=scope,cells=int(selection.sum()),current_max_cosine=float(a.mean()),
                                wrong_max_cosine=float(b.mean()),current_minus_wrong=float((a-b).mean()),fraction_current_greater=float((a>b).mean())))
    report=dict(status='COMPLETE_POSTHOC_SMALL_TRAIN_READOUT_DIAGNOSTIC',rows=rows,sources=sources,source_sha256=sha256(__file__),
        config_sha256=sha256(config_path),unique_Train_frames=12,Dev_rows=0,Test_read=False,
        scope='Already-fitted seed0 epoch24 projections, fixed first and middle cached Train frames per video. Shared-projection local maximum cosine only; not whole-model causal effect, heldout correspondence, identity accuracy or HOTA.')
    path=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_CORRELATION_READOUT_AUDIT.json';dump(path,report)
    for cue in ['rgb','gray','ir']:
        r=[x for x in rows if x['cue']==cue and x['scope']=='current_GT_without_wrong_GT_support'];n=sum(x['cells']for x in r)
        print(cue,'cells',n,'weighted_current_minus_wrong',sum(x['current_minus_wrong']*x['cells']for x in r)/max(n,1))


if __name__=='__main__':main()
