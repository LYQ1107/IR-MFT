#!/usr/bin/env python3
"""Offline official scoring of completed, hash-checked current-only replays."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from cfp.provenance import sha256,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU scoring only')
    parent=Path('configs/cfp/phase_vii_joint_evaluation_subset_v1.json');p=json.loads(parent.read_text());rows=[]
    for path in sorted(a.root.glob('native_identity_*/manifest.json')):
        m=json.loads(path.read_text())
        if m['status']!='COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY':continue
        video=m['video'];pred=path.parent/(video+'.txt')
        if sha256(pred)!=m['prediction_sha256']:raise ValueError('Prediction drift')
        output=Path(str(path.parent)+'.trackeval')
        if not (output/'metrics.json').exists():
            output.mkdir(exist_ok=False);protocol=dict(p,train=[video],dev=[],historical_test=[],role='Offline completed current-only native replay scoring; parameters not selected from metrics',parent_protocol_sha256=sha256(parent));write_json(output/'protocol.json',protocol)
            cmd=[sys.executable,'tools/cfp_evaluate.py','--data-root','/data3/liuyeqiang/datasets/mft26-ir-causal-v1','--derived-gt-root','/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt','--protocol',str(output/'protocol.json'),'--partition','train','--predictions',str(path.parent),'--output',str(output),'--trackeval-root','/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
            with (output/'evaluation.log').open('x') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Actual official scoring failure: '+str(output))
        metrics=json.loads((output/'metrics.json').read_text())
        if metrics['artifacts'][video]['prediction_sha256']!=sha256(pred) or metrics['artifacts'][video]['gt_sha256']!=p['derived_gt_sha256']['train/'+video]:raise ValueError('Official scoring provenance differs')
        rows.append(dict(name=path.parent.name,video=video,memory=m['memory'],context_identity=m['context_identity'],model_checkpoint_sha256=m['model_checkpoint_sha256'],replay_manifest_sha256=sha256(path),metrics_sha256=sha256(output/'metrics.json'),official_metrics_path=str(output/'metrics.json'),metrics=metrics['metrics'][video],association_seconds=m['association_seconds']))
        print(json.dumps(dict(name=path.parent.name,**{k:metrics['metrics'][video][k] for k in ['HOTA','AssA','IDF1','MOTA','IDSW']})),flush=True)
    write_json(a.output,dict(status='COMPLETE_SCORING_OF_AVAILABLE_COMPLETED_NATIVE_REPLAYS',rows=rows,scope='Only completed fullnative video runs included; cached replay association time is not whole raw online FPS. Fullmatrix/three-seed/strongsource/C/D/rawtiming pending. Official allalpha/raw metrics remain external linked by SHA. No tuning from these metrics.'))

if __name__=='__main__':main()
