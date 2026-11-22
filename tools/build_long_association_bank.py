#!/usr/bin/env python3
"""Appearance-only Train identity candidate sets, no extra image dataset."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from yolox.tracker.association_features import pair_features
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='configs/perceptual_strong_rgb_residual_20261123.json');parser.add_argument('--output-name',default='association_long_train');args=parser.parse_args()
    if not args.output_name.replace('_','').isalnum():raise ValueError('Simple bank directory name required')
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    config_path=ROOT/args.plan;config=json.loads(config_path.read_text())
    source=w/'train_bank/train.npz';metadata=json.loads((source.parent/'complete.json').read_text())
    if metadata['status']!='COMPLETE_TRAIN_ONLY' or metadata['Dev_rows_used'] or metadata['Test_rows_used'] or sha256(source)!=metadata['feature_bank_sha256']:raise ValueError('Train-only feature provenance')
    with np.load(source) as z:bank={k:z[k] for k in z.files}
    output=w/args.output_name;output.mkdir(exist_ok=True)
    if (output/'complete.json').exists():raise ValueError('Completed long bank cannot be overwritten')
    reports={}
    for video,sequence in enumerate(protocol['train']):
        indices=np.flatnonzero(bank['sequence']==video);identities=np.unique(bank['label'][indices])
        by_id={identity:indices[bank['label'][indices]==identity] for identity in identities}
        arrays={k:[] for k in ['base','reference','rgb','ir','IR_available','label','query','actual_gap']};query=0
        def data(pick):
            fields=['boxes','embedding','rgb_mask_embedding','ir_mask_embedding','color','rgb_hog','ir_hog','quality']
            result={k:bank[k][pick].copy() for k in fields};result['boxes'][:,2:]+=result['boxes'][:,:2]
            result['point_quality']=result['quality'].copy()
            return result
        for identity,source_indices in by_id.items():
            source_indices=source_indices[np.argsort(bank['frame'][source_indices],kind='stable')]
            samples=np.unique(np.linspace(0,max(0,len(source_indices)//2-1),config['long_source_observations_per_identity'],dtype=int))
            for position in samples:
                i=int(source_indices[position]);frame=int(bank['frame'][i])
                for gap in config['long_desired_gaps']:
                    target_time=frame+gap;window=max(5,gap//4);targets=[]
                    for candidate,available in by_id.items():
                        eligible=available[(bank['frame'][available]>=frame+8)&(np.abs(bank['frame'][available]-target_time)<=window)]
                        if not len(eligible):continue
                        j=int(eligible[np.argmin(np.abs(bank['frame'][eligible]-target_time))]);targets.append(j)
                    if not targets:continue
                    targets=np.array(targets,int);past=data(np.array([i]));current=data(targets);n=len(targets)
                    matrices={(s,t):np.zeros((1,n),np.float32) for s in ['bbox','rgb','joint'] for t in ['bbox','rgb','joint']}
                    good={s:np.zeros(1,np.uint8) for s in ['bbox','rgb','joint']}
                    base,rgb,ref=pair_features(past,current,matrices,good,good,8,'rgb');_,ir,other_ref=pair_features(past,current,matrices,good,good,8,'ir')
                    np.testing.assert_array_equal(ref,other_ref)
                    base=base.reshape(n,23);base[:,:3]=0;base[:,12:14]=0
                    rgb=rgb.reshape(n,17);ir=ir.reshape(n,17);ref=ref.reshape(n,17)
                    for value in [rgb,ir,ref]:value[:,15:17]=0
                    valid=(past['quality'][0,1]>=.15)&(current['quality'][:,1]>=.15)
                    for name,value in [('base',base),('reference',ref),('rgb',rgb),('ir',ir),('IR_available',valid),('label',bank['label'][targets]==identity),('query',np.full(n,query,np.int64)),('actual_gap',bank['frame'][targets]-frame)]:arrays[name].append(value)
                    query+=1
        if not arrays['label']:raise ValueError('Train video has no valid chronological appearance sets')
        saved={k:np.concatenate(v).astype(np.float16 if k in ['base','reference','rgb','ir'] else np.int64 if k in ['query','actual_gap'] else np.uint8) for k,v in arrays.items()}
        path=output/(sequence+'.npz');tmp=output/(sequence+'.tmp.npz');np.savez_compressed(tmp,**saved);tmp.replace(path)
        reports[sequence]=dict(path=str(path),sha256=sha256(path),rows=len(saved['label']),positive_rows=int(saved['label'].sum()),queries=query,bytes=path.stat().st_size,actual_gap_range=[int(saved['actual_gap'].min()),int(saved['actual_gap'].max())])
        print(json.dumps(dict(sequence=sequence,**reports[sequence])),flush=True)
    dump(output/'complete.json',dict(status='COMPLETE_TRAIN_ONLY_APPEARANCE_SETS',videos=protocol['train'],source_bank_sha256=sha256(source),plan_sha256=sha256(config_path),outputs=reports,Dev_rows=0,Test_rows=0,scope='Chronological Train identity candidate sets, not simultaneous detector outputs. Unknown motion features explicitly masked; identities are labels only.',source_sha256=sha256(__file__)))


if __name__=='__main__':main()
