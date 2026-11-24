#!/usr/bin/env python3
"""Physically mismatched native cues with intact RGB and deletion-safe training."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.neck72_runtime import require_storage, require_host_memory
from yolox.tracker.native_query_features import identity_pair_features
from yolox.tracker.native_cue_corruption import permuted_current_packet


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_native_neck_robust_query_20261123.json'
    if a.sequence not in protocol['train']:raise ValueError('Only Train query construction')
    require_storage(w,2**29)
    require_host_memory(8*2**30)
    directory=w/'native_neck_mismatch_query_bank'/a.sequence;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():raise ValueError('Completed native query bank exists')
    alignment=json.loads((w/'native_neck_alignment/complete.json').read_text())
    if alignment['Dev_rows'] or alignment['Test_rows']:raise ValueError('Alignment partition mismatch')
    names=[f'{cue}_s{seed}' for cue in ['rgb','ir'] for seed in [0]]
    rotations={}
    for name in names:
        info=alignment['outputs'][name];path=Path(info['path'])
        if sha256(path)!=info['sha256']:raise ValueError('Alignment map changed')
        with np.load(path) as z:rotations[name]=z['rotation'].copy()
    outputs={};sources={}
    def feature_source(path,expected,field):
        meta=json.loads((path.parent/'complete.json').read_text())
        if meta['status']!=expected or meta.get('Dev_rows',meta.get('Dev_fit_rows',0)) or meta.get('Test_rows',meta.get('Test_fit_rows',0)) or sha256(path)!=meta[field]:raise ValueError('Native Train feature provenance')
        sources[str(path)]=sha256(path);return meta
    def write(regime,reference,native):
        files={}
        for name,arrays in [('reference',reference)]+[(name,dict(cue7=value,quality=qualities[name],mismatch_valid=mismatch_valid)) for name,value in native.items()]:
            path=directory/(regime+'_'+name+'.npz');temporary=directory/(regime+'_'+name+'.tmp.npz');np.savez_compressed(temporary,**arrays);os.replace(temporary,path)
            files[name]=dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size)
        outputs[regime]=files
    # Local geometry and labels were already proved identical for every row.
    index_path=w/'native_candidate_index'/a.sequence/'index.npz';index_meta=json.loads((index_path.parent/'complete.json').read_text())
    if not index_meta['every_saved_geometry_field_exact'] or sha256(index_path)!=index_meta['index_sha256']:raise ValueError('Exact native proposal index missing')
    with np.load(index_path) as z:index={k:z[k].copy() for k in ['stream','edge_source','edge_target','edge_query','edge_label','edge_lag']}
    rgb_path=w/'native_neck_candidate_features'/a.sequence/'rgb/features.npz';rm=feature_source(rgb_path,'COMPLETE_EXACT_TRAIN_NATIVE_FEATURES','feature_sha256')
    ir_path=w/'native_neck_candidate_features'/a.sequence/'ir/features.npz';im=feature_source(ir_path,'COMPLETE_EXACT_TRAIN_NATIVE_FEATURES','feature_sha256')
    if rm['index_sha256']!=index_meta['index_sha256'] or im['index_sha256']!=index_meta['index_sha256']:raise ValueError('Native feature/proposal index mismatch')
    with np.load(rgb_path) as z:reference=z['reference_post'].copy();native={f'rgb_s{seed}':z[f'native_s{seed}']@rotations[f'rgb_s{seed}'] for seed in [0]}
    with np.load(ir_path) as z:native.update({f'ir_s{seed}':z[f'native_s{seed}']@rotations[f'ir_s{seed}'] for seed in [0]})
    starts=np.r_[0,np.flatnonzero(np.diff(index['stream'])!=0)+1];ends=np.r_[starts[1:],len(index['stream'])]
    group=np.repeat(np.arange(len(starts)),ends-starts);sg=group[index['edge_source']];tg=group[index['edge_target']]
    blocks=np.r_[0,np.flatnonzero((np.diff(sg)!=0)|(np.diff(tg)!=0))+1,len(sg)]
    count=len(sg);rgb3=np.zeros((count,3),np.float32);ref7=np.zeros((count,7),np.float32);cue7={name:np.zeros((count,7),np.float32) for name in names};qualities={name:np.zeros((count,2),np.float32) for name in names};mismatch_valid=np.zeros(count,bool)
    with np.load(w/'association_train'/a.sequence/'pairs.npz') as z:quality_by_cue=dict(rgb=z['base'][:,10:12].astype(np.float32),ir=z['ir'][:,2:4].astype(np.float32))
    for lo,hi in zip(blocks[:-1],blocks[1:]):
        s0,s1=starts[sg[lo]],ends[sg[lo]];t0,t1=starts[tg[lo]],ends[tg[lo]]
        rows=index['edge_source'][lo:hi]-s0;cols=index['edge_target'][lo:hi]-t0
        first,_,third=identity_pair_features(reference[s0:s1],reference[t0:t1]);rgb3[lo:hi]=first[rows,cols];ref7[lo:hi]=third[rows,cols]
        m=t1-t0;permutation=np.roll(np.arange(m),1);mismatch_valid[lo:hi]=m>1
        for name in names:
            second=permuted_current_packet(reference[s0:s1],reference[t0:t1],native[name][s0:s1],native[name][t0:t1],permutation);cue7[name][lo:hi]=second[rows,cols]
            q=quality_by_cue[name.split('_')[0]][lo:hi];target_quality=np.empty(m,np.float32);target_quality[cols]=q[:,1]
            qualities[name][lo:hi]=np.c_[q[:,0],target_quality[permutation[cols]]]
    local_path=w/'association_train'/a.sequence/'pairs.npz'
    if sha256(local_path)!=index_meta['source_bank_sha256']:raise ValueError('Original local bank changed')
    with np.load(local_path) as z:
        for old,new in [('query','edge_query'),('label','edge_label'),('lag','edge_lag')]:np.testing.assert_array_equal(z[old],index[new])
    write('local',dict(RGB3=rgb3,reference7=ref7),cue7)
    sources[str(local_path)]=sha256(local_path);sources[str(index_path)]=sha256(index_path)
    del index,native,reference,rgb3,ref7,cue7,sg,tg,group
    # Reconstruct the preserved chronological candidate sets exactly as well.
    teacher_path=w/'native_train_features/reference_s0/features.npz';feature_source(teacher_path,'COMPLETE_TRAIN_ONLY_NATIVE_FEATURES','feature_sha256')
    with np.load(teacher_path) as z:reference=z['post_neck'].copy()
    native={}
    for name in names:
        path=w/'native_neck_train_features'/name/'features.npz';feature_source(path,'COMPLETE_TRAIN_ONLY_NATIVE_FEATURES','feature_sha256')
        with np.load(path) as z:native[name]=z['post_neck']@rotations[name]
    with np.load(w/'train_bank/train.npz') as z:labels=z['label'].copy();frames=z['frame'].copy();video=z['sequence'].copy()
    source_path=w/'association_long_train_v2'/(a.sequence+'.npz')
    manifest=json.loads((source_path.parent/'complete.json').read_text())
    if sha256(source_path)!=manifest['outputs'][a.sequence]['sha256']:raise ValueError('Original long bank changed')
    with np.load(source_path) as z:
        old={k:z[k].copy() for k in ['query','label','actual_gap']}
        quality_by_cue=dict(rgb=z['base'][:,10:12].astype(np.float32),ir=z['ir'][:,2:4].astype(np.float32))
    long_plan=json.loads((ROOT/'configs/perceptual_strong_rgb_residual_v2_20261123.json').read_text())
    indices=np.flatnonzero(video==protocol['train'].index(a.sequence));by_id={identity:indices[labels[indices]==identity] for identity in np.unique(labels[indices])}
    count=len(old['label']);rgb3=np.zeros((count,3),np.float32);ref7=np.zeros((count,7),np.float32);cue7={name:np.zeros((count,7),np.float32) for name in names};qualities={name:np.zeros((count,2),np.float32) for name in names};mismatch_valid=np.zeros(count,bool);offset=query=0
    for identity,source_indices in by_id.items():
        source_indices=source_indices[np.argsort(frames[source_indices],kind='stable')]
        positions=np.unique(np.linspace(0,max(0,len(source_indices)//2-1),long_plan['long_source_observations_per_identity'],dtype=int))
        for position in positions:
            i=int(source_indices[position]);frame=int(frames[i])
            for gap in long_plan['long_desired_gaps']:
                target_time=frame+gap;window=max(5,gap//4);targets=[]
                for candidate,available in by_id.items():
                    eligible=available[(frames[available]>=frame+8)&(np.abs(frames[available]-target_time)<=window)]
                    if len(eligible):targets.append(int(eligible[np.argmin(abs(frames[eligible]-target_time))]))
                if not targets:continue
                targets=np.array(targets,int);m=len(targets);sl=slice(offset,offset+m)
                np.testing.assert_array_equal(old['query'][sl],np.full(m,query));np.testing.assert_array_equal(old['label'][sl],labels[targets]==identity);np.testing.assert_array_equal(old['actual_gap'][sl],frames[targets]-frame)
                first,_,third=identity_pair_features(reference[i:i+1],reference[targets]);rgb3[sl]=first[0];ref7[sl]=third[0]
                permutation=np.roll(np.arange(m),1);mismatch_valid[sl]=m>1
                for name in names:
                    second=permuted_current_packet(reference[i:i+1],reference[targets],native[name][i:i+1],native[name][targets],permutation);cue7[name][sl]=second[0]
                    q=quality_by_cue[name.split('_')[0]][sl];qualities[name][sl]=np.c_[q[:,0],q[permutation,1]]
                offset+=m;query+=1
    if offset!=count:raise ValueError('Long native query coverage mismatch')
    write('long',dict(RGB3=rgb3,reference7=ref7),cue7);sources[str(source_path)]=sha256(source_path)
    dump(directory/'complete.json',dict(status='COMPLETE_TRAIN_ONLY_MISMATCH_QUERY_BANK',sequence=a.sequence,config_sha256=sha256(config_path),sources=sources,outputs=outputs,local_rows=index_meta['pair_rows'],long_rows=count,long_queries=query,alignment_sha256=sha256(w/'native_neck_alignment/complete.json'),Dev_rows=0,Test_rows=0,identity_labels_used_as_features=False,source_sha256=sha256(__file__),feature_source_sha256=sha256(ROOT/'yolox/tracker/native_query_features.py')))
    print(json.dumps(dict(sequence=a.sequence,status='COMPLETE_TRAIN_ONLY_MISMATCH_QUERY_BANK',local_rows=index_meta['pair_rows'],long_rows=count)),flush=True)
    complete=json.loads((directory/'complete.json').read_text());complete.update(selected_encoder_seeds=[0],single_seed_user_policy_sha256=sha256(ROOT/'configs/perceptual_single_seed_policy_20261123.json'));dump(directory/'complete.json',complete);require_storage(w)


if __name__=='__main__':main()
