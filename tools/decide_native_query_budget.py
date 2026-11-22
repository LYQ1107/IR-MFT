#!/usr/bin/env python3
"""Complete common-budget lookback audit, using only both full Dev videos."""
import argparse
import json
from pathlib import Path

from tools.run_pseudo_ir_comparison import dump,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=int,choices=[18,24,30,36],required=True);args=parser.parse_args()
    work=Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    destination=work/f'native_query_common_e{args.epoch}_decision.json'
    if destination.exists():raise ValueError('Completed budget decision already exists')
    groups={}
    for label in ['robust','consensus']:
        stages=[label+'_query_dev']+[label+str(epoch)+'_dev' for epoch in range(24,args.epoch+1,6)]
        for stage in stages:
            state=json.loads((work/(stage+'_state.json')).read_text())
            if state['status']!='COMPLETE':raise ValueError('Every registered curve must finish: '+stage)
            for name in state['jobs']:
                path=work/stage/name/'metrics.json';v=json.loads(path.read_text());m=v['manifest'];c=m['config']
                if c.get('disable_native_increment'):continue
                if m['split']!='dev' or m['frames']!=4834 or v['status']!='COMPLETE':raise ValueError('Full Dev only')
                key=(label,c['detection_arm'],c['native_cue'],c['native_seed'])
                groups.setdefault(key,[]).append(dict(epoch=m['native_query']['epoch'],HOTA=v['trackeval']['HOTA'],path=str(path),sha256=sha256(path)))
    if len(groups)!=24:raise ValueError('Both algorithms, frontends, cues and all3 seeds required')
    rows=[]
    for key,curve in sorted(groups.items()):
        if sorted(row['epoch'] for row in curve)!=list(range(6,args.epoch+1,6)):raise ValueError('Incomplete/duplicate checkpoint curve')
        earlier=[row for row in curve if row['epoch']<=args.epoch-12]
        best=min(curve,key=lambda row:(-row['HOTA'],row['epoch']));old=min(earlier,key=lambda row:(-row['HOTA'],row['epoch']))
        rows.append(dict(algorithm=key[0],frontend=key[1],cue=key[2],seed=key[3],curve=sorted(curve,key=lambda row:row['epoch']),best=best,lookback_best=old,gain_pp=best['HOTA']-old['HOTA']))
    witness=max(rows,key=lambda row:row['gain_pp'])
    extend=args.epoch<36 and witness['gain_pp']>.2
    result=dict(status='EXTEND_ALL_MATCHED_HEADS' if extend else 'STOP_ALL_MATCHED_HEADS',epoch=args.epoch,
        next_epoch=args.epoch+6 if extend else None,maximum_gain_pp=witness['gain_pp'],witness=witness,rows=rows,
        rule='Any complete Dev curve improves best by>0.2pp over its best through12 epochs earlier; maximum36. All24 clean/robust fitting controls use the same common budget.',Test_selection=False,
        stopped_at_cap=args.epoch==36,source_sha256=sha256(__file__))
    dump(destination,result);dump(ROOT/f'docs/fusion_research/PERCEPTUAL_MFT_20261123_NATIVE_COMMON{args.epoch}_DECISION.json',result)
    print(json.dumps(dict(status=result['status'],epoch=args.epoch,maximum_gain_pp=witness['gain_pp'],next_epoch=result['next_epoch'],Dev_curves=len(rows))))


if __name__=='__main__':main()
