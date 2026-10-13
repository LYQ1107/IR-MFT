#!/usr/bin/env python3
"""All six K4 outer/inner folds, three seeds, eight distinct controls."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser()
    for name in ['root','state']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--worker',type=int,choices=range(4),required=True);a=ap.parse_args()
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.root.resolve().is_relative_to(Path.cwd().resolve()):raise ValueError('New outsideGit CPU-only allfold driver required')
    config=Path('configs/cfp/phase_vii_local_dense_context_identity_v2.json');c=json.loads(config.read_text());assignments={0:['full','local_contrastive'],1:['global','global_gru'],2:['local_attention','local_temporal'],3:['no_ema','no_id']};modes=assignments[a.worker]
    tasks=[(v,m,s) for s in c['seed_plan'] for v in c['videos'] for m in modes]
    paths=[__file__,'tools/cfp_train_local_dense_context_identity.py','cfp/local_dense_context_identity.py','cfp/local_dense_jepa.py','cfp/local_dense_data.py','cfp/local_dense_training.py','cfp/local_dense_evaluation.py','cfp/identity_evidence.py','cfp/provenance.py',str(config)];hashes={p:sha256(p) for p in paths}
    state=dict(status='RUNNING',pid=os.getpid(),worker=a.worker,tasks=tasks,completed=[],active=None,sources=snapshot_sources(a.state.with_suffix('.sources'),paths),source_guards=hashes,alias={'no_prediction':'local_temporal'},scope='Sixouter/innerfolds x three seeds x eight distinctcontextV2controls, K4. Originalfirstfold27 actualfits retained; noPrediction aliases localTemporal and is not refitted/counted as independent allfold. K8 remains registered futurestage, no finalgain/novelty/MOT/goalcomplete claim.')
    write_json(a.state,state)
    try:
        for video,mode,seed in tasks:
            if any(sha256(p)!=v for p,v in hashes.items()):raise ValueError('Active registered sources changed')
            if shutil.disk_usage(a.root).free<42*2**30:raise OSError('40GiBreserve plus parallel checkpoint margin')
            out=a.root/f'local_dense_context_{mode}_{video}_k4_s{seed}_v2';state['active']=dict(output=str(out));write_json(a.state,state)
            if not out.exists():
                cmd=[sys.executable,'tools/cfp_train_local_dense_context_identity.py','--features',str(a.root/'dinov2_two_crop_evidence_v1'),'--index',str(a.root/'identity_two_crop_index_v1'),'--legacy-features',str(a.root/'identity_two_crop_features_pilot_s0_v1'),'--config',str(config),'--output',str(out),'--outer',video,'--mode',mode,'--seed',str(seed),'--local-count','4']
                with Path(str(out)+'.log').open('x') as log:
                    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);state['active']['pid']=p.pid;write_json(a.state,state);code=p.wait()
                if code:raise RuntimeError('Actual child failure '+str(code))
            m=json.loads((out/'manifest.json').read_text());g=m['guard']
            if m['status']!='COMPLETE' or g['outer']!=video or g['mode']!=mode or g['seed']!=seed or g['local_count']!=4 or g['config_sha256']!=hashes[str(config)] or sha256(out/'final.pt')!=m['checkpoint_sha256'] or any(sha256(p)!=v for p,v in g['source_sha256'].items()):raise ValueError('Existing result incomplete/changed; inspect actual producer rather than restart')
            if g['inner']==video or video in g['fit'] or g['inner'] in g['fit'] or len(g['fit'])!=4 or m['cuda_initialized']:raise ValueError('Exact outer+inner excluded from everyCPU fit')
            state['completed'].append(dict(output=str(out),manifest_sha256=sha256(out/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256']));state['active']=None;write_json(a.state,state);print(json.dumps(dict(video=video,mode=mode,seed=seed,completed=len(state['completed']))),flush=True)
        state['status']='COMPLETE_ALL_SIX_CONTEXT_IDENTITY_K4_FOLDS';write_json(a.state,state)
    except Exception as e:state.update(status='HALTED_ACTUAL_FAILURE',error=str(e));write_json(a.state,state);raise

if __name__=='__main__':main()
