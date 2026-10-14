#!/usr/bin/env python3
"""Measured whole-past-gallery source-JPM plots, distinct from native-MOT metrics."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cfp.provenance import sha256,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--frozen-report',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('New immutable figure bundle required')
    r=json.loads(a.report.read_text());f=json.loads(a.frozen_report.read_text())
    if r['status']!='COMPLETE_FIRST_FOLD_SOURCE_JPM_THREE_SEED_CROP_CONTROLS':raise ValueError('All nine completed actualfits required')
    colors=['#69727c','#2979b8','#d48224','#537c48'];labels=['Frozen CLS\n384D','Global BN\n384D','JPM\n1920D','CLS capacity\n1920D'];fig,axes=plt.subplots(2,2,figsize=(10,7.3),layout='constrained')
    for row,video in enumerate(['BT-001','SN-015']):
        frozen=f['frozen'][video]['detector']['25']
        for col,metric in enumerate(['rank1_percent','mAP_percent']):
            entries=[r['summary'][video][m]['detector']['25'] for m in ['global','jpm','cls_capacity']]
            assert all(e['support']=={k:frozen[k] for k in ['queries','supported_queries','missing_target']} for e in entries)
            means=[frozen[metric]]+[e[metric]['mean'] for e in entries];sd=[0.]+[e[metric]['sample_sd'] for e in entries];ax=axes[row,col];ax.bar(range(4),means,color=colors,alpha=.88);ax.errorbar(range(4),means,yerr=sd,fmt='none',ecolor='#25313b',capsize=4)
            for x,e in enumerate(entries,1):ax.scatter(np.repeat(x,3),e[metric]['values'],s=15,color='#26313d',zorder=3)
            for x,y in enumerate(means):ax.text(x,y+max(sd[x],.1)+.8,f'{y:.2f}',ha='center',fontsize=9)
            ax.set_xticks(range(4),labels,fontsize=9);ax.set_ylim(0,max(means)+max(sd)+7);ax.set_ylabel('Rank-1 (%)' if col==0 else 'mAP (%)');ax.set_title(f'{video} | {"inner" if row==0 else "outer"}, detector crops',fontsize=11);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Source-matched JPM: fixed final models, three seeds\nStrictly past whole-video gallery, gap ≥ 25',fontsize=13)
    a.output.mkdir();artifacts={}
    for extension in ['png','svg','pdf']:
        path=a.output/('source_jpm_retrieval_three_seed.'+extension);fig.savefig(path,dpi=220);artifacts[extension]=dict(path=str(path),sha256=sha256(path))
    plt.close(fig);write_json(a.output/'manifest.json',dict(status='MEASURED_SOURCE_JPM_THREE_SEED_CROP_FIGURE',report_sha256=sha256(a.report),frozen_report_sha256=sha256(a.frozen_report),artifacts=artifacts,caption='Dot=actualseed, whisker=sampleSD for trainedmodels; frozen deterministic/noSD. Supportmatchedsameviewgallery includingunknown candidates, notsingle-current-scene retrieval ornativeMOT. JPMandCLScapacity1920D/equalactiveparameters; global/frozen384D. No bestseed/outer-based selection.'))
    print(a.output)

if __name__=='__main__':main()
