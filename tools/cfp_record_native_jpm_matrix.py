#!/usr/bin/env python3
"""Verified full-video source-JPM/native matrix, mark incomplete groups explicitly."""
import argparse,json
from pathlib import Path
import numpy as np
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--require-complete',action='store_true');a=ap.parse_args()
    if a.output.exists():raise ValueError('New immutable native matrix required')
    root=a.root;cfg=Path('configs/cfp/phase_vii_native_current_identity_replay_v1.json');protocol=Path('configs/cfp/phase_vii_joint_evaluation_subset_v1.json');p=json.loads(protocol.read_text());rows={};missing=[];features={};summary={};baseline={}
    for mode in ['global','jpm','cls_capacity']:
        for seed in range(3):
            training=root/f'transreid_dino_{mode}_SN-015_s{seed}_v1';tm=json.loads((training/'manifest.json').read_text());modelsha=tm['checkpoint_sha256']
            if tm['status']!='COMPLETE' or sha256(training/'final.pt')!=modelsha:raise ValueError('Exactcomplete sourceJPM model required')
            for video in ['BT-001','SN-015']:
                f=root/f'native_jpm_features_{mode}_{video}_s{seed}_v1';fm=json.loads((f/'manifest.json').read_text()) if (f/'manifest.json').exists() else {}
                for memory in ['last','fifo','ema']:
                    name=f'native_jpm_identity_{mode}_{video}_{memory}_s{seed}_v1';out=root/name;metrics=Path(str(out)+'.trackeval')/'metrics.json'
                    if not metrics.exists():missing.append(name);continue
                    m=json.loads((out/'manifest.json').read_text());official=json.loads(metrics.read_text())
                    if (fm.get('status')!='COMPLETE_NATIVE_CURRENT_JPM_FEATURES' or fm['encoder']['coordinate_sha256']!=modelsha or video in tm['guard']['fit'] or m['status']!='COMPLETE_NATIVE_JPM_IDENTITY_REPLAY' or m['completed_frames']!=fm['frames'] or m['feature_manifest_sha256']!=sha256(f/'manifest.json') or m['config_sha256']!=sha256(cfg) or sha256(out/(video+'.txt'))!=m['prediction_sha256'] or official['artifacts'][video]['prediction_sha256']!=m['prediction_sha256'] or official['artifacts'][video]['gt_sha256']!=p['derived_gt_sha256']['train/'+video]):raise ValueError('Nativefullframe/model/GT/prediction/protocol mismatch')
                    rows[name]=dict(mode=mode,seed=seed,video=video,memory=memory,output=str(out),manifest_sha256=sha256(out/'manifest.json'),feature_manifest_sha256=sha256(f/'manifest.json'),checkpoint_sha256=modelsha,official_metrics_path=str(metrics),official_metrics_sha256=sha256(metrics),metrics=official['metrics'][video],frames=fm['frames'])
                    if str(f) not in features:
                        for field,e in fm['artifacts'].items():
                            if sha256(f/(field+'.npy'))!=e['sha256']:raise ValueError('Actualnativefeature data drift')
                        features[str(f)]=dict(manifest_sha256=sha256(f/'manifest.json'),encoder=fm['encoder'])
    if a.require_complete and missing:raise ValueError('All54required; actualmatrix incomplete')
    for video in ['BT-001','SN-015']:
        summary[video]={};baseline[video]={}
        for memory in ['last','fifo','ema']:
            original=root/f'native_identity_frozen_{video}_{memory}_s0_v1.trackeval/metrics.json';b=json.loads(original.read_text());baseline[video][memory]=dict(path=str(original),sha256=sha256(original),metrics=b['metrics'][video])
        for mode in ['global','jpm','cls_capacity']:
            summary[video][mode]={}
            for memory in ['last','fifo','ema']:
                vals=[r for r in rows.values() if r['video']==video and r['mode']==mode and r['memory']==memory];vals.sort(key=lambda r:r['seed']);summary[video][mode][memory]=dict(completed_seeds=[v['seed'] for v in vals],complete_three_seed=len(vals)==3,metrics={k:dict(values=[v['metrics'][k] for v in vals],mean=float(np.mean([v['metrics'][k] for v in vals])) if len(vals)==3 else None,sample_sd=float(np.std([v['metrics'][k] for v in vals],ddof=1)) if len(vals)==3 else None) for k in ['HOTA','DetA','AssA','IDF1','MOTA','IDSW']})
    result=dict(status='COMPLETE_54_NATIVE_SOURCE_JPM_CONTROLS' if not missing else 'PARTIAL_VERIFIED_AVAILABLE_NATIVE_SOURCE_JPM_CONTROLS',completed_full_videos=len(rows),planned_full_videos=54,missing=missing,rows=rows,features=features,baseline=baseline,summary=summary,sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,cfg,protocol]),scope='Firstfoldthreeencoderfamilies/threefitseeds/twoexcludedwholevideos/threeordinarymemories. ExactsameRGB/proposals/settings/officialGT; learnedhistories endogenous. Complete-seedgroups only getmeans/SD; no luckyseed/memory selection. FrozenCLS384/global384 versusJPMandmatchedCLScapacity1920, privatefoundation/detector exposure unresolved. Notallsixouterfolds/finalmethod/rawFPS/newnovelty.')
    write_json(a.output,result);print(json.dumps(dict(status=result['status'],full_videos=len(rows),missing=len(missing))))

if __name__=='__main__':main()
