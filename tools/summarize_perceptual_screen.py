#!/usr/bin/env python3
"""Audit and export the complete registered first Dev screen, including failures."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics


def main():
    plan_path=ROOT/'configs/perceptual_mft_20261122.json';plan=json.loads(plan_path.read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root']);rows=[]
    for stage,cfgdir in [('screen','perceptual_screen_20261122'),('repair1','perceptual_repair1_20261122')]:
        for config_path in sorted((ROOT/'configs'/cfgdir).glob('*.json')):
            folder=w/stage/config_path.stem;metrics_path=folder/'metrics.json'
            data=json.loads(metrics_path.read_text());manifest=data['manifest']
            if data['status']!='COMPLETE' or manifest['frames']!=4834 or manifest['sequences']!=protocol['dev']:raise ValueError('Incomplete Dev experiment')
            if manifest['config_sha256']!=sha256(config_path):raise ValueError('Registered configuration changed')
            per_sequence=detailed_metrics(folder/'trackeval_output/pedestrian_detailed.csv',protocol['dev'],data['trackeval'])
            if sum(r['CLR_Frames'] for r in per_sequence.values())!=4834:raise ValueError('Incorrect GT coverage')
            row=dict(id=config_path.stem,stage=stage,config=manifest['config'],metrics=data['trackeval'],per_sequence=per_sequence,
                     metrics_path=str(metrics_path),metrics_sha256=sha256(metrics_path),config_sha256=sha256(config_path),
                     inference_statistics=manifest['statistics'],replay_FPS=manifest['replay_FPS'],
                     tracker_source_sha256=manifest['tracker_source_sha256'],replay_source_sha256=manifest['replay_source_sha256'])
            rows.append(row)
    if len(rows)!=20:raise ValueError('Expected all 20 registered Dev rows')
    rgb=next(r for r in rows if r['id']=='C00_rgb_ByteTrack');new=next(r for r in rows if r['id']=='C01_new_ByteTrack')
    if rgb['metrics']['HOTA']!=47.579 or new['metrics']['HOTA']!=46.236:raise ValueError('Baseline replay drift')
    for row in rows:row['HOTA_delta_vs_RGB_pp']=row['metrics']['HOTA']-rgb['metrics']['HOTA']
    report=dict(status='COMPLETE_FIRST_DEV_SCREEN_GOAL_NOT_MET',scope='Dev screening only. No new Test accuracy or supported innovation claim. Each row includes all headline fields and official per-sequence values.',
                plan_sha256=sha256(plan_path),rows=rows,
                best_new_input_candidate=max((r for r in rows if r['config']['detection_arm']=='new' and r['config']['backend']!='byte'),key=lambda r:r['metrics']['HOTA'])['id'],
                supported_innovations=0,next_action='Diagnose cost-scale and feasible-graph changes; keep original ByteTrack lifecycle for the next family; evaluate paired learned and classical feature-normalization controls on Dev.')
    docs=ROOT/'docs/fusion_research';dump(docs/'PERCEPTUAL_MFT_20261122_DEV_SCREEN.json',report)
    columns=['HOTA','DetA','AssA','IDF1','MOTA','IDSW']
    text=['# First registered perceptual MFT Dev screen','',report['status'],'',report['scope'],'',
          '| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | HOTA vs RGB |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        vals=[f"{r['metrics'][k]:.3f}" if k!='IDSW' else str(int(r['metrics'][k])) for k in columns]
        text.append('| '+r['id']+' | '+' | '.join(vals)+f" | {r['HOTA_delta_vs_RGB_pp']:+.3f} |")
    text+=['','All 20 Dev experiments completed. No non-control candidate beats the original RGB control. Ordinary appearance, memory or threshold changes receive no multimodal contribution credit.','',
           'The new-input C02 custom backend changes primary/secondary assignment and lifecycle: its age limit is 30 frames, whereas the original ByteTrack buffer is scaled to int(25/30*30)=25. These are experimental backend changes, not isolated modality effects. The next family keeps the original lifecycle and rejection graph.','',
           '![Complete first Dev screen](PERCEPTUAL_MFT_20261122_DEV_SCREEN.png)','',
           'Full headline/per-sequence fields, hashes and source versions are in the adjacent JSON. Historical Test remains excluded from selection.']
    (docs/'PERCEPTUAL_MFT_20261122_DEV_SCREEN.md').write_text('\n'.join(text)+'\n')
    fig,ax=plt.subplots(figsize=(11,8))
    colors=['#777777' if r['config']['backend']=='byte' else '#E69F00' if r['config']['detection_arm']=='rgb' else '#0072B2' for r in rows]
    yy=list(range(len(rows)));ax.barh(yy,[r['metrics']['HOTA'] for r in rows],color=colors)
    ax.set_yticks(yy,[r['id'] for r in rows],fontsize=8);ax.invert_yaxis()
    ax.axvline(47.579,color='#E69F00',linestyle='--',label='Original RGB + ByteTrack')
    ax.axvline(46.236,color='#555555',linestyle=':',label='New IR + unchanged ByteTrack')
    ax.set_xlim(0,55);ax.set_xlabel('Complete Dev HOTA (%)');ax.set_title('First 20 registered Dev controls: no improvement established')
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='lower right',fontsize=8)
    fig.tight_layout();fig.savefig(docs/'PERCEPTUAL_MFT_20261122_DEV_SCREEN.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status=report['status'],experiments=len(rows),supported_innovations=0)))


if __name__=='__main__':main()
