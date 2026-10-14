"""Only detector pixel rows and actual box geometry; no identity label API."""
import json
from pathlib import Path
import numpy as np
import torch
from .local_identity_encoder import letterbox_visibility
from .provenance import sha256


def load_unlabeled_pool(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if m['status']!='COMPLETE_FIT_ONLY_UNLABELED_DETECTOR_CROP_POOL' or m['physical_identity_values_used'] or m['GT_crop_values_used']:raise ValueError('Actual fit-only unlabeleddetector pool required')
    banks={}
    for video,e in m['videos'].items():
        p=root/(video+'.npz')
        if sha256(p)!=e['pool_sha256'] or sha256(e['source_crops'])!=e['source_crops_sha256']:raise ValueError('Unlabeledpixel/geometry source drift')
        with np.load(p,allow_pickle=False) as data:bank={k:data[k].copy() for k in data.files}
        bank['pixels']=np.load(e['source_crops'],mmap_mode='r');banks[video]=bank
    return banks,m


def unlabeled_pixel_batch(observations,banks,device='cpu'):
    images,masks=[],[]
    for video,index in observations:
        b=banks[video];images.append(b['pixels'][b['crop_indices'][index]]);x1,y1,x2,y2=b['boxes'][index];masks.append(letterbox_visibility(b['dimensions'][index],[x1,y1,x2-x1,y2-y1]))
    return torch.from_numpy(np.asarray(images)).to(device).float(),torch.stack(masks).to(device)
