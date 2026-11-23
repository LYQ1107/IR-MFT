#!/usr/bin/env python3
"""Small scientific figure from complete official Dev results; no Test read."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    state=json.loads((work/'correlation_recoverability_pilot_dev_state.json').read_text())
    if state['status']!='COMPLETE'or len(state['jobs'])!=25:raise ValueError('All25 full Dev cases required')
    rows={}
    for name in state['jobs']:
        d=json.loads((work/'correlation_recoverability_pilot_dev'/name/'metrics.json').read_text())
        if d['manifest']['split']!='dev'or d['manifest']['frames']!=4834:raise ValueError('Full Dev only')
        rows[name]=d['trackeval']
    baseline=rows['CGR_original_RGB_ByteTrack'];epochs=[6,12,18,24]
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for row,policy in enumerate(['raw','priority']):
        for col,metric in enumerate(['HOTA','DetA','AssA']):
            ax=axes[row,col]
            for cue,label,color in [('rgb','RGB repeat','#2874a6'),('gray','Grayscale RGB','#d68910'),('ir','Actual new pseudo-IR','#b03a2e')]:
                suffix='_raw'if policy=='raw'else''
                values=[rows[f'CGR_{cue}_full_s0_e{e:02d}{suffix}'][metric]for e in epochs]
                ax.plot(epochs,values,'o-',color=color,label=label,linewidth=1.6,markersize=4)
            ax.axhline(baseline[metric],color='0.35',linestyle='--',label='Original RGB baseline')
            ax.set_title(f'{policy.capitalize()} output: {metric}');ax.set_xticks(epochs);ax.set_xlabel('Training epoch')
            ax.set_ylabel('Official Dev score');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle('Full Dev recovery curriculum + validity, seed 0; shared frozen encoder/head\nNo Test result or established multimodal contribution',fontsize=12)
    target=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_RECOVERABILITY_PILOT.png'
    fig.savefig(target,dpi=130);plt.close(fig)
    if target.stat().st_size>512*1024:raise ValueError('Small figure budget exceeded')
    print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,Test_read=False)))


if __name__=='__main__':main()
