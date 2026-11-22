#!/usr/bin/env python3
"""Same fixed Train pair draws for RGB and spatially bound appearance controls."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.audit_long_identity_evidence import auc
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    while True:
        state=json.loads((w/'spatial_train_state.json').read_text())
        if state['status']=='COMPLETE':break
        if state['status']=='FAILED':raise RuntimeError('Spatial Train producer failed')
        time.sleep(5)
    path=w/'spatial_identity_evidence_audit.json'
    if path.exists():raise ValueError('Completed spatial audit exists')
    source=w/'train_bank/train.npz';meta=json.loads((source.parent/'complete.json').read_text())
    if sha256(source)!=meta['feature_bank_sha256'] or meta['Dev_rows_used'] or meta['Test_rows_used']:raise ValueError('Train provenance')
    with np.load(source) as bank:data={k:bank[k] for k in ['embedding','label','frame','sequence']}
    raw=data['embedding'].astype(np.float32);raw/=np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-8)
    rng=np.random.default_rng(20261123);rows=[];sources={}
    for video,sequence in enumerate(protocol['train']):
        directory=w/'spatial_tokens/train'/sequence;metadata=json.loads((directory/'complete.json').read_text());tokens=directory/'tokens.npz'
        if metadata['status']!='COMPLETE_TRAIN_ONLY' or metadata['source_bank_sha256']!=sha256(source) or sha256(tokens)!=metadata['token_sha256']:raise ValueError('Spatial Train provenance')
        indices=np.flatnonzero(data['sequence']==video);local=np.full(len(raw),-1,int);local[indices]=np.arange(len(indices))
        with np.load(tokens) as z:
            if not np.array_equal(z['indices'],indices) or not np.array_equal(z['label'],data['label'][indices]) or not np.array_equal(z['frame'],data['frame'][indices]):raise ValueError('Spatial observation index changed')
            features={mode:z[mode+'_tokens'].astype(np.float32) for mode in ['box','rgb','ir']}
            quality={mode:z[mode+'_quality'].astype(np.float32) for mode in ['box','rgb','ir']}
        for value in features.values():value/=np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-8)
        sources[sequence]=metadata
        def scores(pairs):
            p,q=local[pairs[:,0]],local[pairs[:,1]]
            out=dict(raw_RGB=(raw[pairs[:,0]]*raw[pairs[:,1]]).sum(1))
            for mode,value in features.items():
                first,second=value[p],value[q]
                matrix=np.einsum('nac,nbc->nab',first[:,1:],second[:,1:])
                out[mode+'_global']=(first[:,0]*second[:,0]).sum(1)
                out[mode+'_parts']=np.maximum((matrix[:,0,0]+matrix[:,1,1])/2,(matrix[:,0,1]+matrix[:,1,0])/2)
            gate=np.minimum(quality['ir'][p,5],quality['ir'][q,5])*np.minimum(quality['ir'][p,6],quality['ir'][q,6])
            out['RGB_box_counterfactual']=out['raw_RGB']+.5*(out['box_parts']-out['rgb_parts'])
            out['IR_counterfactual']=out['raw_RGB']+.5*(out['ir_parts']-out['rgb_parts'])
            out['IR_structure_guarded_counterfactual']=out['raw_RGB']+.5*gate*(out['ir_parts']-out['rgb_parts'])
            return out
        for gap in [25,128]:
            positive=[];negative=[]
            for _ in range(8000):
                i=int(rng.choice(indices));later=indices[data['frame'][indices]>=data['frame'][i]+gap]
                if not len(later):continue
                same=later[data['label'][later]==data['label'][i]];different=later[data['label'][later]!=data['label'][i]]
                if not len(same) or not len(different):continue
                positive.append((i,int(rng.choice(same))));negative.append((i,int(rng.choice(different))))
                if len(positive)>=2000:break
            if not positive:continue
            p=scores(np.array(positive));n=scores(np.array(negative))
            statistics={k:dict(AUC=auc(p[k],n[k]),positive_median=float(np.median(p[k])),negative_median=float(np.median(n[k]))) for k in p}
            row=dict(sequence=sequence,minimum_gap=gap,positive_pairs=len(positive),negative_pairs=len(negative),statistics=statistics);rows.append(row)
            print(json.dumps(row),flush=True)
    dump(path,dict(status='COMPLETE_TRAIN_ONLY_SPATIAL_FEASIBILITY_AUDIT',scope='Descriptive same/different identity AUC, not tracking HOTA or held-out accuracy. The frozen RGB encoder saw these Train identities, so evidence is optimistic. Same pair draws as the prior Train-only audit; all controls retained.',source_bank_sha256=sha256(source),sources=sources,rows=rows,source_sha256=sha256(__file__),Dev_rows=0,Test_rows=0))


if __name__=='__main__':main()
