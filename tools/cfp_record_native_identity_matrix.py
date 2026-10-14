#!/usr/bin/env python3
"""All126 complete same-source ordinary native controls, no best seed selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from cfp.provenance import sha256, snapshot_sources, write_json


def stats(values):return dict(values=values,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--scored',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('New immutable complete matrix report required')
    rows={v['name']:v for v in json.loads(a.scored.read_text())['rows']};config_path=Path('configs/cfp/phase_vii_native_current_identity_replay_v1.json');c=json.loads(config_path.read_text())
    report=dict(status='COMPLETE_ALL126_NATIVE_ORDINARY_IDENTITY_CONTROLS',runs={},summary={},scored_sha256=sha256(a.scored),
        sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,config_path]),
        scope='All6native videos/frozen3deterministic memory controls andtwo trainedencoder modes ×three seeds ×three matchingmemory rules=126complete native runs. Every adaptedencoder excludes itsouterandnextinner video. SamefixedcausalV3proposals/native lifecycle/output; nativecounter inherits detector/privatefoundation exposure. CacheProtocolA historicaldetector finalglobalunpad3tail differs fromalways-pad8 rawvalidation; no endtoendFPS/allmethod/novelty claim.')
    for video in c['videos']:
        report['summary'][video]={}
        for memory in c['ordinary_controls']:
            entry=dict(frozen={},models={})
            frozen=rows[f'native_identity_frozen_{video}_{memory}_s0_v1'];entry['frozen']=frozen['metrics']
            for mode in ['global','patch_attention']:
                values=[]
                for seed in c['seed_plan']:
                    name=f'native_identity_{mode}_{video}_{memory}_s{seed}_v1';row=rows[name];m=json.loads((a.root/name/'manifest.json').read_text())
                    model=a.root/f'dinov2_fish_{mode}_{video}_k4_s{seed}_v2';trained=json.loads((model/'manifest.json').read_text())
                    if (m['status']!='COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY' or m['online_GT_teacher_future_used']
                            or trained['guard']['outer']!=video or video in trained['guard']['fit'] or trained['guard']['inner'] in trained['guard']['fit']
                            or sha256(a.root/name/'manifest.json')!=row['replay_manifest_sha256']
                            or sha256(row['official_metrics_path'])!=row['metrics_sha256']):raise ValueError('Exact completed native/excluded-encoder/scored provenance required')
                    feature_root=a.root/f'native_dino_{mode}_{video}_k4_s{seed}_v1'
                    fm=json.loads((feature_root/'manifest.json').read_text());encoded=fm['encoder']['trained_encoder']
                    if (sha256(feature_root/'manifest.json')!=m['feature_manifest_sha256'] or encoded is None
                            or encoded['checkpoint_sha256']!=trained['checkpoint_sha256'] or sha256(model/'final.pt')!=trained['checkpoint_sha256']
                            or encoded['outer']!=video or encoded['seed']!=seed or encoded['mode']!=mode):raise ValueError('Actualencoder-source and checkpoint coordinates differ')
                    report['runs'][name]=dict(row,encoder_training=str(model),encoder_checkpoint_sha256=trained['checkpoint_sha256'],encoder_feature_manifest_sha256=m['feature_manifest_sha256']);values.append(row['metrics'])
                entry['models'][mode]={k:stats([v[k] for v in values]) for k,v in values[0].items() if isinstance(v,(int,float))}
            name=f'native_identity_frozen_{video}_{memory}_s0_v1'
            if sha256(frozen['official_metrics_path'])!=frozen['metrics_sha256'] or sha256(a.root/name/'manifest.json')!=frozen['replay_manifest_sha256']:raise ValueError('Frozen native reference drift')
            report['runs'][name]=frozen;report['summary'][video][memory]=entry
    if len(report['runs'])!=126:raise ValueError('Allregistered controls required')
    report['macro']={}
    for memory in c['ordinary_controls']:
        entry=dict(frozen={},models={});metrics=['HOTA','AssA','IDF1','MOTA','IDSW']
        for metric in metrics:entry['frozen'][metric]=float(np.mean([report['summary'][v][memory]['frozen'][metric] for v in c['videos']]))
        for mode in ['global','patch_attention']:
            entry['models'][mode]={metric:stats([float(np.mean([report['summary'][v][memory]['models'][mode][metric]['values'][s] for v in c['videos']])) for s in c['seed_plan']]) for metric in metrics}
        report['macro'][memory]=entry
    write_json(a.output,report);print(json.dumps(dict(status=report['status'],runs=len(report['runs']),macro=report['macro'])))


if __name__=='__main__':main()
