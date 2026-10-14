#!/usr/bin/env python3
"""All42 completed native observable-proxy controls and exact source/prediction scores."""
import argparse,json
from pathlib import Path
import numpy as np
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('New immutable all42 report required')
    driver=a.root/'observable_occlusion_all42_driver_v1.json';d=json.loads(driver.read_text());config=Path('configs/cfp/phase_vii_observable_occlusion_controls_v1.json');c=json.loads(config.read_text())
    if d['status']!='COMPLETE_42_NATIVE_OBSERVABLE_OCCLUSION_CONTROLS' or len(d['completed'])!=7:raise ValueError('Allseven completeconfigs/sixvideos required')
    result=dict(status='COMPLETE_42_NATIVE_OBSERVABLE_OCCLUSION_CONTROL_SCORES',config=c,driver_sha256=sha256(driver),runs={},macro_by_configuration={},videos={},sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,config]),scope=c['scope'])
    for row in d['completed']:
        p=Path(row['output']);m=json.loads((p/'manifest.json').read_text());official=json.loads(Path(row['metrics']).read_text());key=row['mode']+str(row['seed'])
        if m['status']!='COMPLETE' or sha256(p/'manifest.json')!=row['manifest_sha256'] or sha256(row['metrics'])!=row['metrics_sha256'] or set(m['videos'])!=set(c['sequences']):raise ValueError('Exactcompleted control/source/scores required')
        for video,v in m['videos'].items():
            if sha256(p/(video+'.txt'))!=v['prediction_sha256'] or official['artifacts'][video]['prediction_sha256']!=v['prediction_sha256']:raise ValueError('Fullnativeofficial prediction source drift')
            if row['mode']=='native' and not v['baseline_byte_equal']:raise ValueError('Nativeall6 mustbyteequaloriginal reference')
        result['runs'][key]=dict(path=str(p),manifest_sha256=row['manifest_sha256'],metrics_path=row['metrics'],metrics_sha256=row['metrics_sha256'],manifest=m,official_metrics=official['metrics'])
        result['macro_by_configuration'][key]={k:float(np.mean([official['metrics'][v][k] for v in c['sequences']])) for k in ['HOTA','DetA','AssA','IDF1','MOTA','IDSW']}
    for video in c['sequences']:
        result['videos'][video]={key:{k:row['official_metrics'][video][k] for k in ['HOTA','DetA','AssA','IDF1','MOTA','IDSW']} for key,row in result['runs'].items()}
        result['videos'][video]['permutation_placebo_three_seed']={metric:dict(values=[result['runs']['permuted_overlap'+str(seed)]['official_metrics'][video][metric] for seed in range(3)],mean=float(np.mean([result['runs']['permuted_overlap'+str(seed)]['official_metrics'][video][metric] for seed in range(3)])),sample_sd=float(np.std([result['runs']['permuted_overlap'+str(seed)]['official_metrics'][video][metric] for seed in range(3)],ddof=1))) for metric in ['HOTA','AssA','IDF1','MOTA','IDSW']}
    result.update(full_native_videos=42,unique_control_configs=7,all_six_disabled_prediction_bytes_equal=True);write_json(a.output,result);print(json.dumps(result['macro_by_configuration']))

if __name__=='__main__':main()
